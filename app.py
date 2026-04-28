import argparse
import json
from typing import List, Dict, Optional

from src.preprocess import preprocess_batch
from src.embedding import load_model, embed_texts
from src.similarity import cosine_similarity_matrix, rank_topk, normalize_scores_minmax
from src.drift import analyze_drift
from src.recommender import recommend_alternatives
from src.skill_gap import analyze_skill_gap
from src.ai_planner import generate_career_plan


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def flatten_careers(doc: Dict) -> List[Dict]:
    if isinstance(doc, dict) and 'fields' in doc:
        flat: List[Dict] = []
        for field in doc['fields']:
            for c in field.get('careers', []):
                flat.append({**c, 'field': field.get('name')})
        return flat
    return doc.get('careers', doc) if isinstance(doc, dict) else doc


def main():
    parser = argparse.ArgumentParser(description="Career Path Drift Detection (BERT Embedding)")
    parser.add_argument('--cv', type=str, required=True, help='Path to students JSON file containing key "students"')
    parser.add_argument('--careers', type=str, required=True, help='Path to careers JSON file containing key "careers"')
    parser.add_argument('--model', type=str, default='intfloat/multilingual-e5-base',
                        help='SentenceTransformer model name')
    parser.add_argument('--topk', type=int, default=5, help='Top-K careers to display per student')
    parser.add_argument('--min-sim', type=float, default=0.55, help='Minimum similarity for recommendation')
    parser.add_argument('--tau-high', type=float, default=0.70, help='High suitability threshold (normalized 0-1)')
    parser.add_argument('--tau-mid', type=float, default=0.40, help='Mid suitability threshold (normalized 0-1)')
    parser.add_argument('--delta-minor', type=float, default=0.10, help='Minor drift advantage threshold (normalized 0-1)')
    parser.add_argument('--skill-threshold', type=float, default=0.6, help='Similarity threshold for skill gap detection')
    parser.add_argument('--output', type=str, default=None, help='Optional path to export results as JSON')
    parser.add_argument('--ai-plan', action='store_true',
                        help='Generate interview & learning plan via Gemini (requires GEMINI_API_KEY env var)')
    args = parser.parse_args()

    students_doc: Dict = load_json(args.cv)
    careers_doc: Dict = load_json(args.careers)

    students: List[Dict] = students_doc.get('students', students_doc)
    careers: List[Dict] = flatten_careers(careers_doc)

    career_ids = [c['id'] for c in careers]
    career_titles = {c['id']: c['title'] for c in careers}
    career_fields = {c['id']: c.get('field') for c in careers}
    career_skills = {c['id']: c.get('skills', []) for c in careers}
    career_texts = [f"{c['title']}\n{c['description']}\nSkills: {', '.join(c.get('skills', []))}" for c in careers]

    model = load_model(args.model)

    # Preprocess
    preprocessed_careers = preprocess_batch(career_texts)
    preprocessed_cvs = preprocess_batch([s['cv_text'] for s in students])

    # Embeddings — careers as passages, CVs as queries
    career_emb = embed_texts(model, preprocessed_careers, is_passage=True)
    student_emb = embed_texts(model, preprocessed_cvs, is_passage=False)

    # Similarity
    sim = cosine_similarity_matrix(student_emb, career_emb)

    all_results = []
    for i, student in enumerate(students):
        name = student.get('name', f"student_{i}")
        declared_interest: Optional[str] = student.get('declared_interest')

        print(f"\n{'='*60}")
        print(f"Student: {name} (id={student.get('id')})")
        print(f"Declared interest: {declared_interest} ({career_titles.get(declared_interest, '?')})")
        print(f"{'='*60}")

        norm_row = normalize_scores_minmax(sim[i])
        rankings = rank_topk(sim[i], career_ids, topk=args.topk)
        print("\nTop-K Career Rankings:")
        for rank, (cid, score) in enumerate(rankings, start=1):
            rel = float(norm_row[career_ids.index(cid)])
            field_tag = f" [{career_fields.get(cid)}]" if career_fields.get(cid) else ""
            print(f"  {rank:2d}. {career_titles.get(cid, cid)}{field_tag} [id={cid}] -> sim={score:.4f}  relative={rel:.4f}")

        drift = analyze_drift(
            student_vector=student_emb[i],
            career_vectors=career_emb,
            career_ids=career_ids,
            declared_interest=declared_interest,
            thresholds={
                'tau_high': args.tau_high,
                'tau_mid': args.tau_mid,
                'delta_minor': args.delta_minor,
            }
        )

        print(f"\nDrift Analysis:")
        print(f"  Status:               {drift['status']}")
        print(f"  Declared similarity:  {drift['declared_similarity']} (relative={drift.get('declared_relative_score')})")
        best_alt_title = career_titles.get(drift['best_alt_id'], drift['best_alt_id'])
        print(f"  Best alternative:     {best_alt_title} [id={drift['best_alt_id']}] (sim={drift['best_alt_similarity']}, relative={drift.get('best_alt_relative_score')})")
        print(f"  Advantage:            raw={drift.get('raw_advantage')}, relative={drift.get('relative_advantage')}")
        print(f"  Rationale:            {drift['rationale']}")

        recs = recommend_alternatives(
            sim_row=sim[i],
            career_ids=career_ids,
            topk=args.topk,
            min_similarity=args.min_sim,
        )

        print(f"\nRecommendations (min_sim={args.min_sim}):")
        for rank, (cid, _, rel_score) in enumerate(recs, start=1):
            field_tag = f" [{career_fields.get(cid)}]" if career_fields.get(cid) else ""
            print(f"  {rank:2d}. {career_titles.get(cid, cid)}{field_tag} [id={cid}] -> score={rel_score:.4f}")

        # Career transition context
        top_career_id = rankings[0][0] if rankings else None
        if top_career_id and top_career_id != declared_interest:
            top_title = career_titles.get(top_career_id, top_career_id)
            target_title = career_titles.get(declared_interest, declared_interest or '?')
            print(f"\n[Career Transition Detected]")
            print(f"  CV paling cocok untuk: {top_title} [id={top_career_id}]")
            print(f"  Target karier kamu:    {target_title} [id={declared_interest}]")

        # Skill gap analysis
        skill_gap = None
        if declared_interest and declared_interest in career_skills:
            skill_gap = analyze_skill_gap(
                cv_text=student['cv_text'],
                skills=career_skills[declared_interest],
                model=model,
                threshold=args.skill_threshold,
            )
            print(f"\nSkill Gap (threshold={args.skill_threshold}):")
            print(f"  Match ratio: {skill_gap['match_ratio']:.0%} ({len(skill_gap['matched_skills'])}/{len(skill_gap['matched_skills'])+len(skill_gap['missing_skills'])})")
            if skill_gap['matched_skills']:
                print(f"  Matched: {', '.join(s['skill'] for s in skill_gap['matched_skills'])}")
            if skill_gap['missing_skills']:
                upgrade = [s for s in skill_gap['missing_skills'] if s.get('type') == 'upgrade']
                new_skills = [s for s in skill_gap['missing_skills'] if s.get('type') == 'new']
                if upgrade:
                    print(f"  Perlu di-upgrade ({len(upgrade)}):")
                    for s in upgrade:
                        print(f"    - {s['skill']} (upgrade dari: {', '.join(s.get('upgrade_from', []))})")
                if new_skills:
                    print(f"  Perlu dipelajari dari awal ({len(new_skills)}):")
                    for s in new_skills:
                        print(f"    - {s['skill']}")
            if skill_gap.get('outdated_in_cv'):
                print(f"\n  Skill Jadul di CV (perlu di-modernisasi):")
                for s in skill_gap['outdated_in_cv']:
                    print(f"    - {s['skill']} → {', '.join(s['modern_alternatives'])}")

        student_record = {
            "student_id": student.get('id'),
            "name": name,
            "declared_interest": declared_interest,
            "declared_interest_field": career_fields.get(declared_interest) if declared_interest else None,
            "rankings": [
                {
                    "career_id": cid,
                    "title": career_titles.get(cid, cid),
                    "field": career_fields.get(cid),
                    "score": round(float(norm_row[career_ids.index(cid)]), 4),
                }
                for cid, _ in rankings
            ],
            "drift": drift,
            "skill_gap": skill_gap,
            "recommendations": [
                {
                    "career_id": cid,
                    "title": career_titles.get(cid, cid),
                    "field": career_fields.get(cid),
                    "score": round(rel, 4),
                }
                for cid, _, rel in recs
            ],
        }

        if args.ai_plan:
            ai_payload = {
                "target": {
                    "career_id": declared_interest,
                    "title": career_titles.get(declared_interest, declared_interest),
                    "field": career_fields.get(declared_interest) if declared_interest else None,
                    "score": round(float(norm_row[career_ids.index(declared_interest)]), 4)
                        if declared_interest and declared_interest in career_ids else None,
                },
                "best_alternative": {
                    "career_id": drift.get("best_alt_id"),
                    "title": career_titles.get(drift.get("best_alt_id"), drift.get("best_alt_id")),
                    "field": career_fields.get(drift.get("best_alt_id")),
                    "score": drift.get("best_alt_relative_score"),
                },
                "status": drift.get("status"),
                "rationale": drift.get("rationale"),
                "rankings": student_record["rankings"],
                "recommendations": student_record["recommendations"],
                "skill_gap": skill_gap,
            }
            plan = generate_career_plan(ai_payload)
            student_record["ai_plan"] = plan
            if "text" in plan:
                print(f"\n=== AI Plan (Gemini) ===\n{plan['text']}")
                if plan.get("sources"):
                    print("\nSumber:")
                    for s in plan["sources"]:
                        print(f"  - {s.get('title') or s.get('uri')}: {s.get('uri')}")
            elif "error" in plan:
                print(f"\n[AI Plan error] {plan['error']}")

        all_results.append(student_record)

    # Optional JSON export
    if args.output:
        output_data = {
            "model": args.model,
            "thresholds": {"tau_high": args.tau_high, "tau_mid": args.tau_mid, "delta_minor": args.delta_minor},
            "min_similarity": args.min_sim,
            "count_students": len(students),
            "count_careers": len(careers),
            "results": all_results,
        }
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"\nResults exported to {args.output}")


if __name__ == '__main__':
    main()

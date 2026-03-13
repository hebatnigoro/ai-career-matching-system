import argparse
import json
from typing import List, Dict, Optional

from src.preprocess import preprocess_batch
from src.embedding import load_model, embed_texts
from src.similarity import cosine_similarity_matrix, rank_topk, normalize_scores_minmax
from src.drift import analyze_drift
from src.recommender import recommend_alternatives


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


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
    parser.add_argument('--output', type=str, default=None, help='Optional path to export results as JSON')
    args = parser.parse_args()

    students_doc: Dict = load_json(args.cv)
    careers_doc: Dict = load_json(args.careers)

    students: List[Dict] = students_doc.get('students', students_doc)
    careers: List[Dict] = careers_doc.get('careers', careers_doc)

    career_ids = [c['id'] for c in careers]
    career_titles = {c['id']: c['title'] for c in careers}
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
            print(f"  {rank:2d}. {career_titles.get(cid, cid)} [id={cid}] -> sim={score:.4f}  relative={rel:.4f}")

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
            student_vector=student_emb[i],
            career_vectors=career_emb,
            career_ids=career_ids,
            topk=args.topk,
            min_similarity=args.min_sim
        )

        print(f"\nRecommendations (min_sim={args.min_sim}):")
        for rank, (cid, score) in enumerate(recs, start=1):
            print(f"  {rank:2d}. {career_titles.get(cid, cid)} [id={cid}] -> similarity={score:.4f}")

        all_results.append({
            "student_id": student.get('id'),
            "name": name,
            "declared_interest": declared_interest,
            "rankings": [{"career_id": cid, "title": career_titles.get(cid, cid), "similarity": round(s, 4)} for cid, s in rankings],
            "drift": drift,
            "recommendations": [{"career_id": cid, "title": career_titles.get(cid, cid), "similarity": round(s, 4)} for cid, s in recs],
        })

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

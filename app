import argparse
import json
from typing import List, Dict, Optional

from src.preprocess import preprocess_batch
from src.embedding import load_model, embed_texts
from src.similarity import cosine_similarity_matrix, rank_topk
from src.drift import analyze_drift
from src.recommender import recommend_alternatives


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Career Path Drift Detection (BERT Embedding)")
    parser.add_argument('--cv', type=str, required=True, help='Path to students JSON file containing key "students"')
    parser.add_argument('--careers', type=str, required=True, help='Path to careers JSON file containing key "careers"')
    parser.add_argument('--model', type=str, default='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
                        help='SentenceTransformer model name')
    parser.add_argument('--topk', type=int, default=5, help='Top-K careers to display per student')
    parser.add_argument('--min-sim', type=float, default=0.55, help='Minimum similarity for recommendation')
    parser.add_argument('--tau-high', type=float, default=0.70, help='High suitability threshold for declared interest')
    parser.add_argument('--tau-mid', type=float, default=0.60, help='Mid suitability threshold for declared interest')
    parser.add_argument('--delta-minor', type=float, default=0.08, help='Minor drift advantage threshold for alternative')
    args = parser.parse_args()

    students_doc: Dict = load_json(args.cv)
    careers_doc: Dict = load_json(args.careers)

    students: List[Dict] = students_doc.get('students', students_doc)
    careers: List[Dict] = careers_doc.get('careers', careers_doc)

    career_ids = [c['id'] for c in careers]
    career_texts = [f"{c['title']}\n{c['description']}\nSkills: {', '.join(c.get('skills', []))}" for c in careers]

    model = load_model(args.model)

    # Preprocess
    preprocessed_careers = preprocess_batch(career_texts)
    preprocessed_cvs = preprocess_batch([s['cv_text'] for s in students])

    # Embeddings
    career_emb = embed_texts(model, preprocessed_careers)
    student_emb = embed_texts(model, preprocessed_cvs)

    # Similarity
    sim = cosine_similarity_matrix(student_emb, career_emb)  # shape: [num_students, num_careers]

    for i, student in enumerate(students):
        name = student.get('name', f"student_{i}")
        declared_interest: Optional[str] = student.get('declared_interest')

        print(f"\n=== Student: {name} (id={student.get('id')}) ===")
        print(f"Declared interest: {declared_interest}")

        rankings = rank_topk(sim[i], career_ids, topk=args.topk)
        for rank, (cid, score) in enumerate(rankings, start=1):
            title = next(c['title'] for c in careers if c['id'] == cid)
            print(f"{rank:2d}. {title} [id={cid}] -> similarity={score:.4f}")

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

        print("\nDrift Analysis:")
        print(f"Status: {drift['status']}")
        print(f"Declared interest similarity: {drift['declared_similarity']}")
        print(f"Best alternative id: {drift['best_alt_id']} (sim={drift['best_alt_similarity']})")
        print(f"Advantage (best_alt - declared): {drift['advantage']}")
        print(f"Rationale: {drift['rationale']}")

        recs = recommend_alternatives(
            student_vector=student_emb[i],
            career_vectors=career_emb,
            career_ids=career_ids,
            topk=args.topk,
            min_similarity=args.min_sim
        )

        print("\nRecommended Alternatives (min_sim filter applied):")
        for rank, (cid, score) in enumerate(recs, start=1):
            title = next(c['title'] for c in careers if c['id'] == cid)
            print(f"{rank:2d}. {title} [id={cid}] -> similarity={score:.4f}")


if __name__ == '__main__':
    main()

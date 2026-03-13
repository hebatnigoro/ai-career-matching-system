"""
Evaluation & Descriptive Analysis Script for Thesis.

Generates:
  1. Per-student summary table (rankings, drift status, recommendations)
  2. Similarity distribution statistics (mean, std, min, max per career)
  3. Drift category distribution (count per status)
  4. Similarity heatmap (students x careers) saved as PNG
  5. Similarity histogram saved as PNG
  6. Studi kasus: detailed analysis for selected students

Usage:
  python evaluate.py --cv data/students.json --careers data/careers.json
  python evaluate.py --cv data/students.json --careers data/careers.json --output-dir results
"""

import argparse
import json
import os
from collections import Counter

import numpy as np

from src.preprocess import preprocess_batch
from src.embedding import load_model, embed_texts
from src.similarity import cosine_similarity_matrix, rank_topk
from src.drift import analyze_drift
from src.recommender import recommend_alternatives


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_separator(char='=', width=80):
    print(char * width)


def main():
    parser = argparse.ArgumentParser(description="Evaluation & Analysis for Career Path Drift Detection")
    parser.add_argument('--cv', type=str, default='data/students.json')
    parser.add_argument('--careers', type=str, default='data/careers.json')
    parser.add_argument('--model', type=str, default='intfloat/multilingual-e5-base')
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--min-sim', type=float, default=0.55)
    parser.add_argument('--tau-high', type=float, default=0.70)
    parser.add_argument('--tau-mid', type=float, default=0.40)
    parser.add_argument('--delta-minor', type=float, default=0.10)
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save evaluation outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    students_doc = load_json(args.cv)
    careers_doc = load_json(args.careers)
    students = students_doc.get('students', students_doc)
    careers = careers_doc.get('careers', careers_doc)

    career_ids = [c['id'] for c in careers]
    career_titles = {c['id']: c['title'] for c in careers}
    career_texts = [f"{c['title']}\n{c['description']}\nSkills: {', '.join(c.get('skills', []))}" for c in careers]

    print(f"Model: {args.model}")
    print(f"Students: {len(students)}, Careers: {len(careers)}")
    print(f"Thresholds: tau_high={args.tau_high}, tau_mid={args.tau_mid}, delta_minor={args.delta_minor}")
    print(f"min_similarity={args.min_sim}, topk={args.topk}")

    # Pipeline
    model = load_model(args.model)
    preprocessed_careers = preprocess_batch(career_texts)
    preprocessed_cvs = preprocess_batch([s['cv_text'] for s in students])

    career_emb = embed_texts(model, preprocessed_careers, is_passage=True)
    student_emb = embed_texts(model, preprocessed_cvs, is_passage=False)

    sim = cosine_similarity_matrix(student_emb, career_emb)

    # =========================================================
    # 1. Per-Student Summary Table
    # =========================================================
    print_separator()
    print("1. RINGKASAN PER MAHASISWA")
    print_separator()

    all_drift_statuses = []
    all_results = []

    for i, student in enumerate(students):
        name = student.get('name', f"student_{i}")
        sid = student.get('id', f"stu-{i:03d}")
        declared = student.get('declared_interest')

        rankings = rank_topk(sim[i], career_ids, topk=args.topk)
        drift = analyze_drift(
            student_vector=student_emb[i],
            career_vectors=career_emb,
            career_ids=career_ids,
            declared_interest=declared,
            thresholds={'tau_high': args.tau_high, 'tau_mid': args.tau_mid, 'delta_minor': args.delta_minor},
        )
        recs = recommend_alternatives(
            student_vector=student_emb[i],
            career_vectors=career_emb,
            career_ids=career_ids,
            topk=args.topk,
            min_similarity=args.min_sim,
        )

        status = drift['status']
        all_drift_statuses.append(status)

        declared_title = career_titles.get(declared, declared or '-')
        best_alt_title = career_titles.get(drift.get('best_alt_id', ''), '-')

        print(f"\n  [{sid}] {name}")
        print(f"    Declared Interest : {declared_title} (id={declared})")
        print(f"    Declared Sim      : {drift.get('declared_similarity', '-')}")
        print(f"    Best Alternative  : {best_alt_title} (sim={drift.get('best_alt_similarity', '-')})")
        print(f"    Advantage         : {drift.get('advantage', '-')}")
        print(f"    Status            : {status}")
        print(f"    Top-1 Career      : {career_titles.get(rankings[0][0], '?')} (sim={rankings[0][1]:.4f})")

        all_results.append({
            "student_id": sid,
            "name": name,
            "declared_interest": declared,
            "declared_similarity": drift.get('declared_similarity'),
            "best_alt_id": drift.get('best_alt_id'),
            "best_alt_similarity": drift.get('best_alt_similarity'),
            "advantage": drift.get('advantage'),
            "status": status,
            "rationale": drift.get('rationale'),
            "rankings": [{"career_id": cid, "title": career_titles.get(cid, cid), "similarity": round(s, 4)} for cid, s in rankings],
            "recommendations": [{"career_id": cid, "title": career_titles.get(cid, cid), "similarity": round(s, 4)} for cid, s in recs],
        })

    # =========================================================
    # 2. Distribusi Kategori Drift
    # =========================================================
    print_separator()
    print("\n2. DISTRIBUSI KATEGORI DRIFT")
    print_separator()
    drift_counts = Counter(all_drift_statuses)
    for status, count in sorted(drift_counts.items(), key=lambda x: -x[1]):
        pct = count / len(all_drift_statuses) * 100
        bar = '#' * count
        print(f"  {status:35s} : {count:3d} ({pct:5.1f}%)  {bar}")

    # =========================================================
    # 3. Statistik Distribusi Similarity
    # =========================================================
    print_separator()
    print("\n3. STATISTIK DISTRIBUSI SIMILARITY (semua mahasiswa x semua karier)")
    print_separator()

    all_sims = sim.flatten()
    print(f"  Count   : {len(all_sims)}")
    print(f"  Mean    : {np.mean(all_sims):.4f}")
    print(f"  Std     : {np.std(all_sims):.4f}")
    print(f"  Min     : {np.min(all_sims):.4f}")
    print(f"  Max     : {np.max(all_sims):.4f}")
    print(f"  Median  : {np.median(all_sims):.4f}")
    print(f"  Q1 (25%): {np.percentile(all_sims, 25):.4f}")
    print(f"  Q3 (75%): {np.percentile(all_sims, 75):.4f}")

    # Per-career statistics
    print(f"\n  Per-Career Similarity Statistics:")
    print(f"  {'Career':<35s} {'Mean':>7s} {'Std':>7s} {'Min':>7s} {'Max':>7s}")
    print(f"  {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for j, cid in enumerate(career_ids):
        col = sim[:, j]
        print(f"  {career_titles.get(cid, cid):<35s} {np.mean(col):>7.4f} {np.std(col):>7.4f} {np.min(col):>7.4f} {np.max(col):>7.4f}")

    # =========================================================
    # 4. Visualizations (matplotlib)
    # =========================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        # 4a. Similarity Heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(career_ids) * 0.8), max(6, len(students) * 0.5)))
        im = ax.imshow(sim, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(len(career_ids)))
        ax.set_xticklabels([career_titles.get(cid, cid) for cid in career_ids], rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(students)))
        ax.set_yticklabels([s.get('name', s.get('id', f'stu-{i}')) for i, s in enumerate(students)], fontsize=9)
        ax.set_xlabel('Career Profile')
        ax.set_ylabel('Student')
        ax.set_title('Cosine Similarity Heatmap (Student x Career)')

        # Add text annotations
        for i in range(len(students)):
            for j in range(len(career_ids)):
                ax.text(j, i, f"{sim[i, j]:.2f}", ha='center', va='center', fontsize=6,
                        color='white' if sim[i, j] > 0.6 else 'black')

        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        plt.tight_layout()
        heatmap_path = os.path.join(args.output_dir, 'similarity_heatmap.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"\n  Heatmap saved to: {heatmap_path}")

        # 4b. Similarity Distribution Histogram
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(all_sims, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(args.tau_high, color='green', linestyle='--', linewidth=1.5, label=f'tau_high={args.tau_high}')
        ax.axvline(args.tau_mid, color='orange', linestyle='--', linewidth=1.5, label=f'tau_mid={args.tau_mid}')
        ax.axvline(args.min_sim, color='red', linestyle='--', linewidth=1.5, label=f'min_sim={args.min_sim}')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Cosine Similarity Scores (All Student-Career Pairs)')
        ax.legend()
        plt.tight_layout()
        hist_path = os.path.join(args.output_dir, 'similarity_histogram.png')
        plt.savefig(hist_path, dpi=150)
        plt.close()
        print(f"  Histogram saved to: {hist_path}")

        # 4c. Drift Category Bar Chart
        fig, ax = plt.subplots(figsize=(8, 5))
        statuses = list(drift_counts.keys())
        counts = [drift_counts[s] for s in statuses]
        colors = []
        for s in statuses:
            if 'Aligned' in s or 'Strong' in s:
                colors.append('green')
            elif 'Minor' in s:
                colors.append('orange')
            elif 'Major' in s:
                colors.append('red')
            elif 'Exploration' in s:
                colors.append('gray')
            else:
                colors.append('steelblue')
        ax.barh(statuses, counts, color=colors, edgecolor='black')
        ax.set_xlabel('Count')
        ax.set_title('Drift Category Distribution')
        for i, (cnt, s) in enumerate(zip(counts, statuses)):
            ax.text(cnt + 0.1, i, str(cnt), va='center')
        plt.tight_layout()
        drift_chart_path = os.path.join(args.output_dir, 'drift_distribution.png')
        plt.savefig(drift_chart_path, dpi=150)
        plt.close()
        print(f"  Drift chart saved to: {drift_chart_path}")

    except ImportError:
        print("\n  [WARNING] matplotlib not installed — skipping chart generation.")
        print("  Install with: pip install matplotlib")

    # =========================================================
    # 5. Export Full Results as JSON
    # =========================================================
    export = {
        "model": args.model,
        "thresholds": {
            "tau_high": args.tau_high,
            "tau_mid": args.tau_mid,
            "delta_minor": args.delta_minor,
            "min_similarity": args.min_sim,
        },
        "count_students": len(students),
        "count_careers": len(careers),
        "drift_distribution": dict(drift_counts),
        "similarity_stats": {
            "mean": round(float(np.mean(all_sims)), 4),
            "std": round(float(np.std(all_sims)), 4),
            "min": round(float(np.min(all_sims)), 4),
            "max": round(float(np.max(all_sims)), 4),
            "median": round(float(np.median(all_sims)), 4),
        },
        "results": all_results,
    }
    json_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"\n  Full results exported to: {json_path}")

    # =========================================================
    # 6. Studi Kasus (3 examples)
    # =========================================================
    print_separator()
    print("\n4. STUDI KASUS")
    print_separator()

    # Pick one from each category if possible
    case_studies = {}
    for r in all_results:
        s = r['status']
        if s not in case_studies:
            case_studies[s] = r

    for status, r in case_studies.items():
        print(f"\n  --- {status}: {r['name']} ({r['student_id']}) ---")
        print(f"  CV declared interest: {career_titles.get(r['declared_interest'], r['declared_interest'])}")
        print(f"  Declared similarity:  {r.get('declared_similarity', '-')}")
        print(f"  Best alternative:     {career_titles.get(r.get('best_alt_id', ''), '-')} (sim={r.get('best_alt_similarity', '-')})")
        print(f"  Advantage:            {r.get('advantage', '-')}")
        print(f"  Rationale:            {r.get('rationale', '-')}")
        print(f"  Top-3 Rankings:")
        for rank, rec in enumerate(r['rankings'][:3], 1):
            print(f"    {rank}. {rec['title']} (sim={rec['similarity']})")
        print(f"  Recommendations:")
        for rank, rec in enumerate(r['recommendations'][:3], 1):
            print(f"    {rank}. {rec['title']} (sim={rec['similarity']})")

    print_separator()
    print(f"\nEvaluation complete. All outputs saved in: {args.output_dir}/")
    print_separator()


if __name__ == '__main__':
    main()

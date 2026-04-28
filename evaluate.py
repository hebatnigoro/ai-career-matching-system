"""
Evaluation & Descriptive Analysis Script for Thesis.

Generates:
  1. Per-student summary table (rankings, drift status, recommendations)
  2. Drift category distribution
  3. Similarity distribution statistics (raw cosine & relative score)
  4. Inter-career similarity analysis (discrimination quality)
  5. Expected vs Actual drift validation
  6. Visualizations: heatmap, histogram, drift chart, inter-career heatmap, boxplot
  7. Studi kasus per drift category
  8. Full results exported to JSON

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
from src.similarity import cosine_similarity_matrix, normalize_scores_minmax
from src.drift import analyze_drift
from src.recommender import recommend_alternatives
from src.skill_gap import analyze_skill_gap


def load_json(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _flatten_careers(doc):
    if isinstance(doc, dict) and 'fields' in doc:
        flat = []
        for field in doc['fields']:
            for c in field.get('careers', []):
                flat.append({**c, 'field': field.get('name')})
        return flat
    return doc.get('careers', doc) if isinstance(doc, dict) else doc


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
    parser.add_argument('--skill-threshold', type=float, default=0.6,
                        help='Similarity threshold for skill gap detection')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save evaluation outputs')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    students_doc = load_json(args.cv)
    careers_doc = load_json(args.careers)
    students = students_doc.get('students', students_doc)
    careers = _flatten_careers(careers_doc)

    career_ids = [c['id'] for c in careers]
    career_titles = {c['id']: c['title'] for c in careers}
    career_skills = {c['id']: c.get('skills', []) for c in careers}
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

    # Compute normalized scores per student
    norm_sim = np.zeros_like(sim)
    for i in range(len(students)):
        norm_sim[i] = normalize_scores_minmax(sim[i])

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

        drift = analyze_drift(
            student_vector=student_emb[i],
            career_vectors=career_emb,
            career_ids=career_ids,
            declared_interest=declared,
            thresholds={'tau_high': args.tau_high, 'tau_mid': args.tau_mid, 'delta_minor': args.delta_minor},
        )
        recs = recommend_alternatives(
            sim_row=sim[i],
            career_ids=career_ids,
            topk=args.topk,
            min_similarity=args.min_sim,
        )

        status = drift['status']
        all_drift_statuses.append(status)

        # Skill gap analysis
        skill_gap = None
        if declared and declared in career_skills:
            skill_gap = analyze_skill_gap(
                cv_text=student['cv_text'],
                skills=career_skills[declared],
                model=model,
                threshold=args.skill_threshold,
            )

        declared_title = career_titles.get(declared, declared or '-')
        best_alt_title = career_titles.get(drift.get('best_alt_id', ''), '-')

        # Top-1 by relative score
        top1_idx = int(np.argmax(norm_sim[i]))
        top1_id = career_ids[top1_idx]

        print(f"\n  [{sid}] {name}")
        print(f"    Declared Interest : {declared_title}")
        print(f"    Declared Score    : {drift.get('declared_relative_score', '-')}")
        print(f"    Best Alternative  : {best_alt_title} (score={drift.get('best_alt_relative_score', '-')})")
        print(f"    Advantage         : {drift.get('relative_advantage', '-')}")
        print(f"    Status            : {status}")
        print(f"    Top-1 Career      : {career_titles.get(top1_id, '?')} (score={norm_sim[i][top1_idx]:.4f})")
        if skill_gap:
            print(f"    Skill Match       : {skill_gap['match_ratio']:.0%} ({len(skill_gap['matched_skills'])}/{len(skill_gap['matched_skills'])+len(skill_gap['missing_skills'])})")
            if skill_gap['missing_skills']:
                missing_names = ', '.join(s['skill'] for s in skill_gap['missing_skills'][:5])
                print(f"    Top Missing Skills: {missing_names}")

        all_results.append({
            "student_id": sid,
            "name": name,
            "declared_interest": declared,
            "declared_relative_score": drift.get('declared_relative_score'),
            "best_alt_id": drift.get('best_alt_id'),
            "best_alt_relative_score": drift.get('best_alt_relative_score'),
            "relative_advantage": drift.get('relative_advantage'),
            "status": status,
            "rationale": drift.get('rationale'),
            "skill_gap": skill_gap,
            "rankings": [
                {
                    "career_id": cid,
                    "title": career_titles.get(cid, cid),
                    "score": round(float(norm_sim[i][career_ids.index(cid)]), 4),
                }
                for cid, _, _ in recs
            ],
            "recommendations": [
                {
                    "career_id": cid,
                    "title": career_titles.get(cid, cid),
                    "score": round(rel, 4),
                }
                for cid, _, rel in recs
            ],
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
    print("\n3. STATISTIK DISTRIBUSI SIMILARITY")
    print_separator()

    all_raw = sim.flatten()
    all_norm = norm_sim.flatten()

    print(f"\n  A. Raw Cosine Similarity (semua mahasiswa x semua karier):")
    print(f"     Count   : {len(all_raw)}")
    print(f"     Mean    : {np.mean(all_raw):.4f}")
    print(f"     Std     : {np.std(all_raw):.4f}")
    print(f"     Min     : {np.min(all_raw):.4f}")
    print(f"     Max     : {np.max(all_raw):.4f}")
    print(f"     Range   : {np.max(all_raw) - np.min(all_raw):.4f}")

    print(f"\n  B. Relative Score / Normalized (semua mahasiswa x semua karier):")
    print(f"     Mean    : {np.mean(all_norm):.4f}")
    print(f"     Std     : {np.std(all_norm):.4f}")
    print(f"     Median  : {np.median(all_norm):.4f}")
    print(f"     Q1 (25%): {np.percentile(all_norm, 25):.4f}")
    print(f"     Q3 (75%): {np.percentile(all_norm, 75):.4f}")

    # Per-career statistics (relative score)
    print(f"\n  C. Per-Career Relative Score Statistics:")
    print(f"     {'Career':<35s} {'Mean':>7s} {'Std':>7s} {'Min':>7s} {'Max':>7s}")
    print(f"     {'-'*35} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")
    for j, cid in enumerate(career_ids):
        col = norm_sim[:, j]
        print(f"     {career_titles.get(cid, cid):<35s} {np.mean(col):>7.4f} {np.std(col):>7.4f} {np.min(col):>7.4f} {np.max(col):>7.4f}")

    # =========================================================
    # 4. Inter-Career Similarity Analysis
    # =========================================================
    print_separator()
    print("\n4. ANALISIS INTER-CAREER SIMILARITY")
    print("   (Membuktikan profil karier terdiskriminasi satu sama lain)")
    print_separator()

    career_sim = cosine_similarity_matrix(career_emb, career_emb)

    # Mask diagonal (self-similarity = 1.0)
    mask = ~np.eye(len(career_ids), dtype=bool)
    off_diag = career_sim[mask]

    print(f"\n  Cosine similarity antar profil karier (tanpa diagonal):")
    print(f"     Mean    : {np.mean(off_diag):.4f}")
    print(f"     Std     : {np.std(off_diag):.4f}")
    print(f"     Min     : {np.min(off_diag):.4f}")
    print(f"     Max     : {np.max(off_diag):.4f}")

    # Find most similar career pairs
    pairs = []
    for i in range(len(career_ids)):
        for j in range(i + 1, len(career_ids)):
            pairs.append((career_ids[i], career_ids[j], career_sim[i, j]))

    print(f"\n  Top-5 pasangan karier PALING MIRIP (potensi overlap):")
    pairs_sorted_desc = sorted(pairs, key=lambda x: -x[2])
    for rank, (a, b, s) in enumerate(pairs_sorted_desc[:5], 1):
        print(f"     {rank}. {career_titles[a]} <-> {career_titles[b]} : {s:.4f}")

    print(f"\n  Top-5 pasangan karier PALING BERBEDA (diskriminasi terbaik):")
    pairs_sorted_asc = sorted(pairs, key=lambda x: x[2])
    for rank, (a, b, s) in enumerate(pairs_sorted_asc[:5], 1):
        print(f"     {rank}. {career_titles[a]} <-> {career_titles[b]} : {s:.4f}")

    # =========================================================
    # 5. Expected vs Actual Drift Validation
    # =========================================================
    print_separator()
    print("\n5. VALIDASI: EXPECTED vs ACTUAL DRIFT")
    print("   (Ground truth manual berdasarkan analisis isi CV)")
    print_separator()

    # Manual ground truth based on CV content analysis
    expected_drift = {
        "stu-001": "Minor Drift",        # Andi: backend dev, declared data-scientist
        "stu-002": "Aligned",            # Bunga: DKV/Figma, declared ui-ux-designer
        "stu-003": "Aligned",            # Chan: CS/algorithms, declared software-engineer
        "stu-004": "Aligned",            # Dewi: marketing/social media, declared digital-marketer
        "stu-005": "Aligned",            # Eko: lab jaringan, declared network-engineer
        "stu-006": "Major Drift",        # Fani: psikologi/HRD, declared data-scientist
        "stu-007": "Minor Drift",        # Gita: full-stack, declared mobile-engineer
        "stu-008": "Minor Drift",        # Hadi: sysadmin, declared software-engineer
        "stu-009": "Minor Drift",        # Ika: marketing+SQL, declared product-manager
        "stu-010": "Minor Drift",        # Joko: QA intern, declared software-engineer
        "stu-011": "Aligned",            # Karin: ML research, declared ml-engineer
        "stu-012": "Major Drift",        # Lina: AWS infra, declared cybersecurity
        "stu-013": "Major Drift",        # Made: graphic design, declared software-engineer
        "stu-014": "Minor Drift",        # Nanda: backend+data pipeline, declared data-scientist
        "stu-015": "Exploration Needed",  # Omar: generalist, declared software-engineer
    }

    match_count = 0
    total_count = 0
    print(f"\n  {'Student':<12s} {'Name':<12s} {'Expected':<22s} {'Actual':<22s} {'Match':>5s}")
    print(f"  {'-'*12} {'-'*12} {'-'*22} {'-'*22} {'-'*5}")

    for r in all_results:
        sid = r['student_id']
        expected = expected_drift.get(sid)
        actual = r['status']
        if expected:
            total_count += 1
            match = expected == actual
            if match:
                match_count += 1
            symbol = 'OK' if match else 'MISS'
            print(f"  {sid:<12s} {r['name']:<12s} {expected:<22s} {actual:<22s} {symbol:>5s}")

    if total_count > 0:
        accuracy = match_count / total_count * 100
        print(f"\n  Agreement rate: {match_count}/{total_count} ({accuracy:.1f}%)")
        print(f"  Catatan: Expected drift ditentukan secara manual berdasarkan analisis isi CV.")
        print(f"  Perbedaan (MISS) bukan berarti salah — bisa jadi threshold perlu penyesuaian")
        print(f"  atau ground truth manual perlu direvisi berdasarkan semantik model.")

    # =========================================================
    # 6. Visualizations
    # =========================================================
    category_order = ['Aligned', 'Minor Drift', 'Major Drift', 'Exploration Needed']
    cat_colors = {
        'Aligned': '#2ecc71',
        'Minor Drift': '#f39c12',
        'Major Drift': '#e74c3c',
        'Exploration Needed': '#95a5a6',
    }

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        student_names = [s.get('name', s.get('id', f'stu-{i}')) for i, s in enumerate(students)]
        career_labels = [career_titles.get(cid, cid) for cid in career_ids]

        # 6a. Relative Score Heatmap (Student x Career)
        fig, ax = plt.subplots(figsize=(max(14, len(career_ids) * 0.9), max(6, len(students) * 0.45)))
        im = ax.imshow(norm_sim, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_xticks(range(len(career_ids)))
        ax.set_xticklabels(career_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(students)))
        ax.set_yticklabels(student_names, fontsize=8)
        ax.set_xlabel('Career Profile')
        ax.set_ylabel('Student')
        ax.set_title('Relative Score Heatmap (Student x Career)')
        for i in range(len(students)):
            for j in range(len(career_ids)):
                ax.text(j, i, f"{norm_sim[i, j]:.2f}", ha='center', va='center', fontsize=5,
                        color='white' if norm_sim[i, j] > 0.7 or norm_sim[i, j] < 0.15 else 'black')
        plt.colorbar(im, ax=ax, label='Relative Score (0-1)')
        plt.tight_layout()
        heatmap_path = os.path.join(args.output_dir, 'relative_score_heatmap.png')
        plt.savefig(heatmap_path, dpi=150)
        plt.close()
        print(f"\n  Heatmap saved to: {heatmap_path}")

        # 6b. Inter-Career Similarity Heatmap
        fig, ax = plt.subplots(figsize=(max(12, len(career_ids) * 0.8), max(10, len(career_ids) * 0.7)))
        im = ax.imshow(career_sim, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
        ax.set_xticks(range(len(career_ids)))
        ax.set_xticklabels(career_labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(career_ids)))
        ax.set_yticklabels(career_labels, fontsize=7)
        ax.set_title('Inter-Career Cosine Similarity')
        for i in range(len(career_ids)):
            for j in range(len(career_ids)):
                ax.text(j, i, f"{career_sim[i, j]:.2f}", ha='center', va='center', fontsize=5,
                        color='white' if career_sim[i, j] > 0.7 else 'black')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        plt.tight_layout()
        inter_career_path = os.path.join(args.output_dir, 'inter_career_similarity.png')
        plt.savefig(inter_career_path, dpi=150)
        plt.close()
        print(f"  Inter-career heatmap saved to: {inter_career_path}")

        # 6c. Relative Score Distribution per Drift Category (Boxplot)
        drift_scores = {}
        for r in all_results:
            s = r['status']
            score = r.get('declared_relative_score')
            if score is not None:
                drift_scores.setdefault(s, []).append(score)

        if drift_scores:
            labels = [c for c in category_order if c in drift_scores]
            data = [drift_scores[c] for c in labels]

            fig, ax = plt.subplots(figsize=(10, 6))
            bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5)
            for patch, label in zip(bp['boxes'], labels):
                patch.set_facecolor(cat_colors.get(label, '#3498db'))
                patch.set_alpha(0.7)

            # Overlay individual points
            for idx, (d, label) in enumerate(zip(data, labels), 1):
                jitter = np.random.default_rng(42).normal(0, 0.04, size=len(d))
                ax.scatter(np.full(len(d), idx) + jitter, d, alpha=0.6, s=30, zorder=3,
                           color=cat_colors.get(label, '#3498db'), edgecolor='black', linewidth=0.5)

            ax.axhline(args.tau_high, color='green', linestyle='--', alpha=0.5, label=f'tau_high={args.tau_high}')
            ax.axhline(args.tau_mid, color='orange', linestyle='--', alpha=0.5, label=f'tau_mid={args.tau_mid}')
            ax.set_ylabel('Declared Interest Relative Score')
            ax.set_title('Score Distribution per Drift Category')
            ax.legend(loc='upper right')
            plt.tight_layout()
            boxplot_path = os.path.join(args.output_dir, 'drift_boxplot.png')
            plt.savefig(boxplot_path, dpi=150)
            plt.close()
            print(f"  Boxplot saved to: {boxplot_path}")

        # 6d. Drift Category Bar Chart
        fig, ax = plt.subplots(figsize=(8, 5))
        statuses = [c for c in category_order if c in drift_counts]
        counts = [drift_counts[s] for s in statuses]
        colors = [cat_colors.get(s, '#3498db') for s in statuses]
        ax.barh(statuses, counts, color=colors, edgecolor='black')
        ax.set_xlabel('Jumlah Mahasiswa')
        ax.set_title('Distribusi Kategori Drift')
        for i, cnt in enumerate(counts):
            ax.text(cnt + 0.1, i, str(cnt), va='center', fontweight='bold')
        plt.tight_layout()
        drift_chart_path = os.path.join(args.output_dir, 'drift_distribution.png')
        plt.savefig(drift_chart_path, dpi=150)
        plt.close()
        print(f"  Drift chart saved to: {drift_chart_path}")

        # 6e. Raw vs Normalized Comparison Histogram
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(all_raw, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].set_xlabel('Cosine Similarity')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Raw Cosine Similarity Distribution')
        axes[0].axvline(np.mean(all_raw), color='red', linestyle='--', label=f'Mean={np.mean(all_raw):.3f}')
        axes[0].legend()

        axes[1].hist(all_norm, bins=30, edgecolor='black', alpha=0.7, color='coral')
        axes[1].axvline(args.tau_high, color='green', linestyle='--', label=f'tau_high={args.tau_high}')
        axes[1].axvline(args.tau_mid, color='orange', linestyle='--', label=f'tau_mid={args.tau_mid}')
        axes[1].set_xlabel('Relative Score')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Relative Score Distribution (Normalized)')
        axes[1].legend()

        plt.suptitle('Perbandingan: Raw Cosine vs Relative Score', fontsize=13)
        plt.tight_layout()
        compare_path = os.path.join(args.output_dir, 'raw_vs_normalized.png')
        plt.savefig(compare_path, dpi=150)
        plt.close()
        print(f"  Comparison histogram saved to: {compare_path}")

    except ImportError:
        print("\n  [WARNING] matplotlib not installed — skipping chart generation.")
        print("  Install with: pip install matplotlib")

    # =========================================================
    # 7. Studi Kasus (one per category)
    # =========================================================
    print_separator()
    print("\n7. STUDI KASUS")
    print_separator()

    case_studies = {}
    for r in all_results:
        s = r['status']
        if s not in case_studies:
            case_studies[s] = r

    for status_label in category_order:
        r = case_studies.get(status_label)
        if not r:
            continue
        print(f"\n  --- {status_label}: {r['name']} ({r['student_id']}) ---")
        print(f"  Declared interest : {career_titles.get(r['declared_interest'], r['declared_interest'])}")
        print(f"  Declared score    : {r.get('declared_relative_score', '-')}")
        print(f"  Best alternative  : {career_titles.get(r.get('best_alt_id', ''), '-')} "
              f"(score={r.get('best_alt_relative_score', '-')})")
        print(f"  Advantage         : {r.get('relative_advantage', '-')}")
        print(f"  Rationale         : {r.get('rationale', '-')}")
        sg = r.get('skill_gap')
        if sg:
            print(f"  Skill Match       : {sg['match_ratio']:.0%}")
            if sg['matched_skills']:
                print(f"  Matched Skills    : {', '.join(s['skill'] for s in sg['matched_skills'])}")
            if sg['missing_skills']:
                print(f"  Missing Skills    : {', '.join(s['skill'] for s in sg['missing_skills'])}")
        print(f"  Top-3 Recommendations:")
        for rank, rec in enumerate(r['recommendations'][:3], 1):
            print(f"    {rank}. {rec['title']} (score={rec['score']})")

    # =========================================================
    # 8. Export Full Results as JSON
    # =========================================================
    export = {
        "model": args.model,
        "thresholds": {
            "tau_high": args.tau_high,
            "tau_mid": args.tau_mid,
            "delta_minor": args.delta_minor,
            "skill_threshold": args.skill_threshold,
            "min_similarity": args.min_sim,
        },
        "count_students": len(students),
        "count_careers": len(careers),
        "drift_distribution": dict(drift_counts),
        "similarity_stats": {
            "raw_cosine": {
                "mean": round(float(np.mean(all_raw)), 4),
                "std": round(float(np.std(all_raw)), 4),
                "min": round(float(np.min(all_raw)), 4),
                "max": round(float(np.max(all_raw)), 4),
                "range": round(float(np.max(all_raw) - np.min(all_raw)), 4),
            },
            "relative_score": {
                "mean": round(float(np.mean(all_norm)), 4),
                "std": round(float(np.std(all_norm)), 4),
                "median": round(float(np.median(all_norm)), 4),
            },
        },
        "inter_career_similarity": {
            "mean": round(float(np.mean(off_diag)), 4),
            "std": round(float(np.std(off_diag)), 4),
            "min": round(float(np.min(off_diag)), 4),
            "max": round(float(np.max(off_diag)), 4),
            "most_similar_pairs": [
                {"a": career_titles[a], "b": career_titles[b], "similarity": round(float(s), 4)}
                for a, b, s in pairs_sorted_desc[:5]
            ],
        },
        "expected_vs_actual": {
            "agreement_rate": round(match_count / total_count * 100, 1) if total_count > 0 else None,
            "matches": match_count,
            "total": total_count,
        },
        "results": all_results,
    }
    json_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
    print(f"\n  Full results exported to: {json_path}")

    print_separator()
    print(f"\nEvaluation complete. All outputs saved in: {args.output_dir}/")
    print_separator()


if __name__ == '__main__':
    main()

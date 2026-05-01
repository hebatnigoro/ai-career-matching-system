"""Offline evaluation harness for the multi-criteria job matcher.

For each labeled case in ``data/eval_gold.json``, run the matcher and
report ranking-quality metrics:

* **Precision@k**  proportion of the top-k items with relevance >= 1.
* **NDCG@k**       Normalized Discounted Cumulative Gain on integer
                   labels (gain = label, no exponentiation needed when
                   labels are small integers).
* **MRR**          Reciprocal rank of the first relevant (label >= 1)
                   item in the ranking.
* **Eligible@k**   Fraction of top-k flagged eligible by the matcher.

Run from the project root:

    python evaluate_jobs.py
    python evaluate_jobs.py --gold data/eval_gold.json --topk 5 10

The harness uses the **same** ``match_jobs_for_cv`` entry point that the
API serves, so improvements to the scoring logic show up here too —
this is an end-to-end integration check, not a unit test.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple

# Make the project's `src` importable when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.embedding import load_model
from src.skill_extract import build_skill_registry
from src.job_matcher import enrich_jobs_with_skills, match_jobs_for_cv


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def precision_at_k(labels_in_rank_order: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = labels_in_rank_order[:k]
    if not top:
        return 0.0
    return sum(1 for x in top if x >= 1) / k


def dcg_at_k(labels: List[int], k: int) -> float:
    return sum(
        (label / math.log2(i + 2))     # i+2 because positions are 1-indexed inside log
        for i, label in enumerate(labels[:k])
    )


def ndcg_at_k(labels_in_rank_order: List[int], k: int) -> float:
    actual = dcg_at_k(labels_in_rank_order, k)
    ideal = dcg_at_k(sorted(labels_in_rank_order, reverse=True), k)
    if ideal == 0:
        return 0.0
    return actual / ideal


def mrr(labels_in_rank_order: List[int]) -> float:
    for i, label in enumerate(labels_in_rank_order, start=1):
        if label >= 1:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------

def evaluate_case(
    case: Dict,
    model,
    registry,
    ks: List[int],
) -> Dict:
    """Run the matcher on one case and return metric dict."""
    cv_text = case["cv_text"]
    jobs = case["jobs"]
    label_map = {j["id"]: int(j["label"]) for j in jobs}

    # Enrich jobs with inferred skills + experience range so the matcher
    # exercises the same hot path the API uses post-refresh.
    job_pool = [
        {k: v for k, v in j.items() if k != "label"}
        for j in jobs
    ]
    enriched = enrich_jobs_with_skills(job_pool, model, registry)

    result = match_jobs_for_cv(
        cv_text=cv_text,
        jobs=enriched,
        model=model,
        registry=registry,
        topk=len(jobs),  # rank everything so metrics are well-defined
    )

    ranked_labels = [label_map.get(r["job"]["id"], 0) for r in result["ranked"]]

    metrics: Dict = {
        "case_id": case["case_id"],
        "n_jobs": len(jobs),
        "n_relevant": sum(1 for l in label_map.values() if l >= 1),
        "n_perfect": sum(1 for l in label_map.values() if l >= 2),
        "mrr": round(mrr(ranked_labels), 4),
        "eligible_count": result["eligible_count"],
    }
    for k in ks:
        metrics[f"P@{k}"] = round(precision_at_k(ranked_labels, k), 4)
        metrics[f"NDCG@{k}"] = round(ndcg_at_k(ranked_labels, k), 4)
        eligible_topk = sum(1 for r in result["ranked"][:k] if r["eligible"])
        metrics[f"eligible@{k}"] = round(eligible_topk / k if k else 0.0, 4)

    metrics["_top_predictions"] = [
        {
            "rank": i + 1,
            "job_id": r["job"]["id"],
            "title": r["job"]["title"],
            "label": label_map.get(r["job"]["id"], 0),
            "final_score": r["scores"]["final"],
            "eligible": r["eligible"],
        }
        for i, r in enumerate(result["ranked"][:max(ks)])
    ]
    return metrics


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def _load_global_skill_registry(careers_path: str):
    """Build the same registry the API uses (covers all known skills)."""
    if not os.path.exists(careers_path):
        return build_skill_registry([])
    with open(careers_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if isinstance(doc, dict) and "fields" in doc:
        flat: List[Dict] = []
        for field in doc["fields"]:
            for c in field.get("careers", []):
                flat.append({**c, "field": field.get("name")})
        return build_skill_registry(flat)
    return build_skill_registry(doc.get("careers", doc) if isinstance(doc, dict) else doc)


def _aggregate(metrics_per_case: List[Dict], ks: List[int]) -> Dict:
    if not metrics_per_case:
        return {}
    n = len(metrics_per_case)
    agg: Dict = {"n_cases": n, "mrr": round(sum(m["mrr"] for m in metrics_per_case) / n, 4)}
    for k in ks:
        agg[f"P@{k}"] = round(sum(m[f"P@{k}"] for m in metrics_per_case) / n, 4)
        agg[f"NDCG@{k}"] = round(sum(m[f"NDCG@{k}"] for m in metrics_per_case) / n, 4)
        agg[f"eligible@{k}"] = round(sum(m[f"eligible@{k}"] for m in metrics_per_case) / n, 4)
    return agg


def main():
    parser = argparse.ArgumentParser(description="Evaluate the job matcher against a labeled gold set.")
    parser.add_argument("--gold", default="data/eval_gold.json", help="Path to the gold-set JSON file.")
    parser.add_argument("--careers", default="data/careers.json", help="Career list used to build the global skill registry.")
    parser.add_argument("--model", default="intfloat/multilingual-e5-base", help="Embedding model name.")
    parser.add_argument("--topk", type=int, nargs="+", default=[3, 5, 10], help="K values for P@k and NDCG@k.")
    parser.add_argument("--out", default=None, help="Optional path to write a JSON report.")
    args = parser.parse_args()

    if not os.path.exists(args.gold):
        print(f"[error] gold set not found at {args.gold}")
        sys.exit(2)

    with open(args.gold, "r", encoding="utf-8") as f:
        gold = json.load(f)
    cases = gold.get("cases", [])
    if not cases:
        print("[error] gold set has no cases")
        sys.exit(2)

    print(f"[evaluate] loading careers + building skill registry from {args.careers}")
    registry = _load_global_skill_registry(args.careers)
    print(f"           {len(registry.canonical_skills)} canonical skills")

    print(f"[evaluate] loading embedding model {args.model}")
    model = load_model(args.model)

    print(f"[evaluate] running {len(cases)} cases")
    per_case: List[Dict] = []
    for case in cases:
        m = evaluate_case(case, model, registry, args.topk)
        per_case.append(m)

    aggregate = _aggregate(per_case, args.topk)

    # ---------- Report ----------
    print()
    print("=" * 72)
    print("PER-CASE METRICS")
    print("=" * 72)
    cols = ["case_id", "n_jobs", "n_relevant"]
    for k in args.topk:
        cols += [f"P@{k}", f"NDCG@{k}"]
    cols += ["MRR"]
    header = " | ".join(f"{c:>9}" if c != "case_id" else f"{c:<22}" for c in cols)
    print(header)
    print("-" * len(header))
    for m in per_case:
        row = [f"{m['case_id']:<22}"]
        row.append(f"{m['n_jobs']:>9}")
        row.append(f"{m['n_relevant']:>9}")
        for k in args.topk:
            row.append(f"{m[f'P@{k}']:>9.3f}")
            row.append(f"{m[f'NDCG@{k}']:>9.3f}")
        row.append(f"{m['mrr']:>9.3f}")
        print(" | ".join(row))

    print()
    print("=" * 72)
    print("AGGREGATE (mean over cases)")
    print("=" * 72)
    print(f"  n_cases = {aggregate['n_cases']}")
    for k in args.topk:
        print(f"  P@{k}      = {aggregate[f'P@{k}']:.4f}")
        print(f"  NDCG@{k}   = {aggregate[f'NDCG@{k}']:.4f}")
        print(f"  Eligible@{k} = {aggregate[f'eligible@{k}']:.4f}")
    print(f"  MRR       = {aggregate['mrr']:.4f}")

    print()
    print("=" * 72)
    print("TOP PREDICTIONS PER CASE (with relevance label)")
    print("=" * 72)
    for m in per_case:
        print(f"\n[{m['case_id']}]")
        for p in m["_top_predictions"]:
            tag = "FIT" if p["label"] >= 2 else ("partial" if p["label"] == 1 else "no-fit")
            print(f"  #{p['rank']:>2}  [{p['final_score']:.3f}]  {p['title'][:48]:<48}  -> {tag}{'  (ineligible)' if not p['eligible'] else ''}")

    if args.out:
        report = {"per_case": per_case, "aggregate": aggregate, "ks": args.topk, "model": args.model}
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[evaluate] report written to {args.out}")


if __name__ == "__main__":
    main()

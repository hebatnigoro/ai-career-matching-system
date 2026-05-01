"""Multi-criteria job matching: rank jobs against a CV.

Pipeline (per job j and CV c):

    S_sem   = cosine(emb(c), emb(j.description))    — min-max normalized
    S_skill = matched / (matched + missing)         — uses skill_extract.py
              against the skills inferred from the JD via the global
              skill registry (built from careers.json)
    S_exp   = exp(- (cv_yrs - req_yrs)^2 / (2σ²) )  — gaussian fit, σ=2
    S_loc   = exact city → 1.0 | same country → 0.6
              | remote-allowed → 0.8 | else → 0.2

    final_score = w_sem·S_sem + w_skill·S_skill + w_exp·S_exp + w_loc·S_loc

A job is *eligible* if ``S_loc >= 0.2`` and ``S_skill >= τ_skill``.
Eligibility is a hard filter; ranking is by ``final_score``. Both stages
expose every sub-score so the UI can show "why this match?".
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.embedding import embed_texts
from src.preprocess import preprocess_batch, preprocess_text
from src.similarity import cosine_similarity_matrix, normalize_scores_minmax
from src.skill_extract import (
    SkillEvidence,
    SkillRegistry,
    extract_cv_skills,
    evidence_to_dict,
)
from src.cv_profile import (
    CVProfile,
    extract_cv_profile,
    required_experience_from_jd,
    profile_to_dict,
)


# ----------------------------------------------------------------------
# Default weights (tunable per request)
# ----------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "semantic": 0.30,
    "skill":    0.40,   # dominant — recruiters filter on skills first
    "experience": 0.15,
    "location": 0.15,
}

DEFAULT_EXP_SIGMA = 2.0   # years; tolerance band for experience fit
DEFAULT_SKILL_THRESHOLD = 0.30   # eligibility floor on skill match
DEFAULT_LOC_THRESHOLD = 0.20     # eligibility floor on location fit

# Coarse-to-fine cap: keep this many candidates after the cheap embedding
# pass before running per-job skill / experience / location scoring. The
# embedding step is vectorized (one batched model.encode call), but the
# layered skill extractor on every JD is O(jobs * skills * tokens) and
# becomes the bottleneck when matching against 500+ unenriched jobs.
# 150 is wide enough that the final top-K (typically 10-20) is rarely
# clipped — the highest skill/exp/loc scores almost always sit among the
# highest-semantic candidates.
DEFAULT_COARSE_CAP = 150


# ----------------------------------------------------------------------
# Skill inference from JD text
# ----------------------------------------------------------------------

def infer_job_required_skills(
    description: str,
    registry: SkillRegistry,
    model: SentenceTransformer,
    semantic_threshold: float = 0.78,
    enable_semantic: bool = True,
    enable_fuzzy: bool = True,
) -> List[str]:
    """Use the layered extractor on a JD to infer its required skills.

    The same machinery that tells us *what skills a CV has* also tells us
    *what skills a JD asks for* — there is no asymmetry. This avoids
    relying on the JD author to provide a clean ``required_skills`` array
    (most ATS exports don't have that field).

    ``enable_semantic`` is exposed because the semantic layer dominates
    runtime (one model.encode pass per JD). When matching against many
    unenriched jobs at request time, the caller can drop semantic and
    rely on lexical+fuzzy alone — proper-noun tech skills (the bulk of
    JD requirements) are caught by lexical matching anyway. Refresh-time
    enrichment keeps semantic on for full quality.
    """
    if not description:
        return []
    extracted = extract_cv_skills(
        cv_text=description,
        registry=registry,
        model=model,
        semantic_threshold=semantic_threshold,
        enable_semantic=enable_semantic,
        enable_fuzzy=enable_fuzzy,
    )
    return list(extracted.keys())


def enrich_jobs_with_skills(
    jobs: List[Dict],
    model: SentenceTransformer,
    registry: SkillRegistry,
    semantic_threshold: float = 0.78,
) -> List[Dict]:
    """Run skill inference + JD embedding once per job, store on the dict.

    This moves the two expensive per-job operations out of the match
    request path: ``/jobs/refresh`` runs them once, ``/match_jobs`` reads
    the cached values instead of re-running them on every match.

    Each job gains:
        * ``inferred_skills``         list[str] — required skills from JD
        * ``required_experience``     [min, max] | None — years
        * ``embedding``               list[float] — title+description vector
                                      (L2-normalized, same model as match)

    Embeddings are stored as a Python list of floats so the cache stays
    JSON-serializable. For ~1000 jobs × 768 dims that's ~3 MB on disk —
    a worthwhile trade for sub-second matches.
    """
    enriched: List[Dict] = []
    # Batch-embed all JDs in one model.encode call — much faster than
    # calling embed_texts per job.
    job_texts = [
        f"{j.get('title', '')}\n{j.get('description', '')}"
        for j in jobs
    ]
    job_embs = embed_texts(model, preprocess_batch(job_texts), is_passage=True)

    for idx, j in enumerate(jobs):
        desc = j.get("description") or ""
        inferred = infer_job_required_skills(desc, registry, model, semantic_threshold)
        req_range = required_experience_from_jd(desc)
        out = dict(j)
        out["inferred_skills"] = inferred
        out["required_experience"] = list(req_range) if req_range else None
        out["embedding"] = job_embs[idx].tolist()
        enriched.append(out)
    return enriched


# ----------------------------------------------------------------------
# Sub-scorers
# ----------------------------------------------------------------------

def score_skill_match(
    cv_skills: Dict[str, "SkillEvidence"],
    required_skills: List[str],
) -> Dict:
    """Skill overlap score against pre-extracted CV skills.

    The CV's skill-extraction is identical for every job in the same match
    request, so the caller now extracts once and passes the result here
    instead of re-running the layered extractor per job (the dominant cost
    when matching against a large unenriched cache).
    """
    if not required_skills:
        return {
            "score": 0.0,
            "matched": [],
            "missing": [],
            "match_ratio": 0.0,
        }
    matched: List[Dict] = []
    missing: List[str] = []
    for skill in required_skills:
        if skill in cv_skills:
            ev = cv_skills[skill]
            matched.append({
                "skill": skill,
                "source": ev.source,
                "confidence": ev.confidence,
                "evidence": ev.matched_text,
            })
        else:
            missing.append(skill)
    n_required = len(required_skills)
    match_ratio = len(matched) / n_required if n_required else 0.0
    return {
        "score": round(match_ratio, 4),
        "matched": matched,
        "missing": missing,
        "match_ratio": round(match_ratio, 4),
    }


def score_experience(
    cv_years: Optional[float],
    required_range: Optional[Tuple[float, float]],
    sigma: float = DEFAULT_EXP_SIGMA,
) -> Dict:
    """Gaussian penalty on |cv - required|, in [0, 1].

    Edge cases:
        * No CV signal       → 0.5 (neutral, don't penalize)
        * No required signal → 1.0 (job didn't specify; don't filter)
        * CV inside the band → 1.0
        * CV above the band  → small penalty (overqualified, but acceptable)
        * CV below the band  → larger penalty (underqualified)
    """
    if required_range is None:
        return {"score": 1.0, "reason": "jd_unspecified"}
    if cv_years is None:
        return {"score": 0.5, "reason": "cv_unspecified"}
    lo, hi = required_range
    if lo <= cv_years <= hi:
        return {"score": 1.0, "reason": "in_range", "cv": cv_years, "required": [lo, hi]}
    if cv_years > hi:
        gap = cv_years - hi
        # Overqualified: gentler decay
        score = math.exp(-(gap ** 2) / (2 * (sigma * 1.5) ** 2))
        return {
            "score": round(score, 4), "reason": "overqualified",
            "cv": cv_years, "required": [lo, hi], "gap": round(gap, 1),
        }
    gap = lo - cv_years
    score = math.exp(-(gap ** 2) / (2 * sigma ** 2))
    return {
        "score": round(score, 4), "reason": "underqualified",
        "cv": cv_years, "required": [lo, hi], "gap": round(gap, 1),
    }


def _normalize_loc_token(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def score_location(
    cv_city: Optional[str],
    cv_country: Optional[str],
    cv_remote: Optional[bool],
    job: Dict,
) -> Dict:
    """Score how well the job's location fits the CV's location.

    Hierarchy: exact city > same country > remote-allowed > else.
    """
    job_loc = _normalize_loc_token(job.get("location"))
    job_country = _normalize_loc_token(job.get("country"))
    job_workplace = (job.get("workplace_type") or "unspecified").lower()
    is_remote_job = bool(job.get("remote")) or job_workplace == "remote"

    cv_city_n = _normalize_loc_token(cv_city)
    cv_country_n = _normalize_loc_token(cv_country)

    # Both unknown → neutral
    if not cv_city_n and not cv_country_n:
        return {"score": 0.5, "reason": "cv_location_unknown"}

    # Job is remote → high score regardless of where the CV is
    if is_remote_job:
        return {"score": 1.0, "reason": "job_is_remote"}

    # Exact city match
    if cv_city_n and job_loc and cv_city_n in job_loc:
        return {"score": 1.0, "reason": "city_match", "cv_city": cv_city, "job_location": job.get("location")}

    # Same country (compare ISO codes or full names)
    if cv_country_n and (job_country and cv_country_n == job_country):
        return {"score": 0.6, "reason": "country_match"}
    if cv_country_n and job_loc and cv_country_n in job_loc:
        return {"score": 0.6, "reason": "country_match_via_loc"}

    # Hybrid jobs are partially location-flexible
    if job_workplace == "hybrid":
        return {"score": 0.4, "reason": "hybrid_partial_fit"}

    # CV is open to remote but the job isn't — still a small bonus over
    # a hard mismatch because some "onsite" listings actually allow remote
    if cv_remote:
        return {"score": 0.3, "reason": "cv_remote_pref_only"}

    return {"score": 0.2, "reason": "location_mismatch"}


# ----------------------------------------------------------------------
# Top-level matcher
# ----------------------------------------------------------------------

def match_jobs_for_cv(
    cv_text: str,
    jobs: List[Dict],
    model: SentenceTransformer,
    registry: SkillRegistry,
    weights: Optional[Dict[str, float]] = None,
    skill_threshold: float = DEFAULT_SKILL_THRESHOLD,
    loc_threshold: float = DEFAULT_LOC_THRESHOLD,
    topk: int = 20,
    filters: Optional[Dict] = None,
    coarse_cap: int = DEFAULT_COARSE_CAP,
) -> Dict:
    """Rank jobs for one CV. Returns ranked list with full sub-score audit.

    ``filters`` is a coarse pre-filter applied before scoring (cheap):
        {"location": "Jakarta", "remote": True, "employment_type": "FullTime"}
    """
    if not jobs:
        return {"ranked": [], "filtered_out": 0, "cv_profile": None}

    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    filters = filters or {}

    # 1) Coarse filter
    f_loc = (filters.get("location") or "").strip().lower() or None
    f_remote = filters.get("remote")
    f_emptype = (filters.get("employment_type") or "").strip().lower() or None
    f_company = (filters.get("company") or "").strip().lower() or None

    candidates: List[Dict] = []
    filtered_out = 0
    for j in jobs:
        if f_loc:
            hay = " ".join(filter(None, [j.get("location"), j.get("country")])).lower()
            if f_loc not in hay and not (f_remote and j.get("remote")):
                filtered_out += 1
                continue
        if f_remote is True and not j.get("remote"):
            filtered_out += 1
            continue
        if f_emptype:
            et = (j.get("employment_type") or "").lower().replace(" ", "")
            if f_emptype.replace(" ", "") not in et:
                filtered_out += 1
                continue
        if f_company:
            if f_company not in (j.get("company") or "").lower():
                filtered_out += 1
                continue
        candidates.append(j)

    if not candidates:
        return {
            "ranked": [],
            "filtered_out": filtered_out,
            "candidates_considered": 0,
            "cv_profile": profile_to_dict(extract_cv_profile(cv_text)),
        }

    # 2) Profile + embeddings + CV skill extraction (ONCE per request)
    profile = extract_cv_profile(cv_text)

    # CV skill extraction is identical across all candidate jobs in this
    # request, so we run the layered extractor exactly once. Previously
    # this was called inside ``score_skill_match`` for every job, which
    # made matching with an unenriched cache pathologically slow
    # (O(n_jobs * cv_size) instead of O(cv_size)).
    cv_skills = extract_cv_skills(cv_text=cv_text, registry=registry, model=model)

    cv_emb = embed_texts(model, preprocess_batch([cv_text]), is_passage=False)[0]

    # Prefer cached JD embeddings produced by ``enrich_jobs_with_skills``.
    # Encoding 600+ JDs at request time on CPU costs ~100s; reading them
    # from cache is instant. Mixed cache (some embedded, some not) is
    # supported — only the missing JDs get encoded.
    cached_idx = [i for i, j in enumerate(candidates) if j.get("embedding")]
    missing_idx = [i for i, j in enumerate(candidates) if not j.get("embedding")]
    job_emb = np.zeros((len(candidates), cv_emb.shape[0]), dtype=np.float32)
    for i in cached_idx:
        job_emb[i] = np.asarray(candidates[i]["embedding"], dtype=np.float32)
    if missing_idx:
        miss_texts = [
            f"{candidates[i].get('title','')}\n{candidates[i].get('description','')}"
            for i in missing_idx
        ]
        live_emb = embed_texts(model, preprocess_batch(miss_texts), is_passage=True)
        for k, i in enumerate(missing_idx):
            job_emb[i] = live_emb[k]
    sim_row = cosine_similarity_matrix(cv_emb.reshape(1, -1), job_emb)[0]

    # Coarse-to-fine cap. The embedding pass above is fast (one batched
    # forward pass), but the layered skill extractor in the per-job loop
    # below is the dominant cost when the cache lacks ``inferred_skills``.
    # Keeping only the top-N most semantically similar candidates here
    # drops total latency from O(N_jobs) to O(coarse_cap) without hurting
    # final ranking quality (skill/exp/loc top-K is dense in the high-
    # semantic tail).
    n_total_candidates = len(candidates)
    if coarse_cap and n_total_candidates > coarse_cap:
        keep_idx = np.argsort(-sim_row)[:coarse_cap]
        candidates = [candidates[i] for i in keep_idx]
        sim_row = sim_row[keep_idx]
    coarse_dropped = n_total_candidates - len(candidates)

    norm_row = normalize_scores_minmax(sim_row)

    # 3) Per-job scoring
    scored: List[Dict] = []
    for idx, j in enumerate(candidates):
        s_sem = float(norm_row[idx])

        # Skill score — prefer the cached `inferred_skills` produced by
        # ``enrich_jobs_with_skills`` at refresh time. Fall back to a
        # pre-supplied ``skills`` array (rare for ATS feeds) or to live
        # inference (slow, only hit when the cache is stale).
        cached = j.get("inferred_skills")
        if isinstance(cached, list):
            required = cached
        elif j.get("skills"):
            required = j["skills"]
        else:
            # Live fallback — keep only the lexical layer. Fuzzy and
            # semantic each add ~0.5-1s per JD on CPU, which is
            # untenable when matching against 100+ unenriched jobs.
            # Tech skills in JDs are proper nouns ("React", "PostgreSQL",
            # "Docker") that lexical word-boundary matching catches with
            # high precision. Refresh-time enrichment uses the full
            # 3-layer pipeline for max recall.
            required = infer_job_required_skills(
                j.get("description", ""), registry, model,
                enable_semantic=False,
                enable_fuzzy=False,
            )
        sk = score_skill_match(cv_skills, required)

        # Experience — prefer cached parse from refresh time
        cached_range = j.get("required_experience")
        if isinstance(cached_range, list) and len(cached_range) == 2:
            req_range = (float(cached_range[0]), float(cached_range[1]))
        else:
            req_range = required_experience_from_jd(j.get("description", ""))
        exp = score_experience(profile.experience_years, req_range)

        # Location
        loc = score_location(
            profile.location_city,
            profile.location_country,
            profile.remote_preference,
            j,
        )

        composite = (
            weights["semantic"] * s_sem
            + weights["skill"] * sk["score"]
            + weights["experience"] * exp["score"]
            + weights["location"] * loc["score"]
        )

        eligible = (sk["score"] >= skill_threshold) and (loc["score"] >= loc_threshold)

        scored.append({
            "job": _slim_job(j),
            "scores": {
                "semantic":   round(s_sem, 4),
                "skill":      round(sk["score"], 4),
                "experience": round(exp["score"], 4),
                "location":   round(loc["score"], 4),
                "final":      round(composite, 4),
            },
            "eligible": eligible,
            "skill_match": {
                "matched": sk["matched"],
                "missing": sk["missing"],
                "match_ratio": sk["match_ratio"],
                "required_skills": required,
            },
            "experience_match": exp,
            "location_match": loc,
        })

    # 4) Eligibility split + ranking
    eligible = [s for s in scored if s["eligible"]]
    ineligible = [s for s in scored if not s["eligible"]]
    eligible.sort(key=lambda x: -x["scores"]["final"])
    ineligible.sort(key=lambda x: -x["scores"]["final"])

    ranked = (eligible + ineligible)[:topk]

    return {
        "weights": weights,
        "thresholds": {
            "skill_threshold": skill_threshold,
            "loc_threshold": loc_threshold,
        },
        "cv_profile": profile_to_dict(profile),
        "candidates_considered": len(candidates),
        "filtered_out_by_pre_filter": filtered_out,
        "coarse_dropped": coarse_dropped,
        "coarse_cap": coarse_cap,
        "eligible_count": len(eligible),
        "ranked": ranked,
    }


def _slim_job(j: Dict) -> Dict:
    """Project only the fields the UI needs to render a job card."""
    keys = (
        "id", "source", "company", "title", "department", "team",
        "location", "country", "remote", "workplace_type", "employment_type",
        "url", "apply_url", "posted_at", "compensation",
    )
    return {k: j.get(k) for k in keys}

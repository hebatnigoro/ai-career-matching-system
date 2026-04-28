import re
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import os
import numpy as np

from src.preprocess import preprocess_batch
from src.embedding import load_model, embed_texts
from src.similarity import cosine_similarity_matrix, normalize_scores_minmax
from src.drift import analyze_drift
from src.recommender import recommend_alternatives
from src.skill_gap import analyze_skill_gap
from src.file_extract import extract_text_auto
from src.ai_planner import generate_career_plan


# ---------- Data Models ----------
class Thresholds(BaseModel):
    tau_high: float = 0.70
    tau_mid: float = 0.40
    delta_minor: float = 0.10
    skill_threshold: float = 0.6


class AnalyzeSingleRequest(BaseModel):
    cv_text: str
    target_career_id: str
    topk: int = 5
    min_sim: float = 0.55
    thresholds: Thresholds = Thresholds()
    model: Optional[str] = 'intfloat/multilingual-e5-base'
    include_ai_plan: bool = True


class Career(BaseModel):
    id: str
    title: str
    description: str
    skills: Optional[List[str]] = []
    field: Optional[str] = None


class Student(BaseModel):
    id: str
    name: Optional[str] = None
    cv_text: str
    declared_interest: Optional[str] = None


class AnalyzeBatchRequest(BaseModel):
    students: List[Student]
    careers: List[Career]
    topk: int = 5
    min_sim: float = 0.55
    thresholds: Thresholds = Thresholds()
    model: Optional[str] = 'intfloat/multilingual-e5-base'
    include_ai_plan: bool = False


# ---------- FastAPI App ----------
app = FastAPI(
    title="Career Path Drift API",
    version="2.0.0",
    description=(
        "API untuk analisis career path drift. "
        "Mendukung input CV teks maupun unggah berkas (PDF/DOCX). "
        "Profil karier dimuat dari server (data/careers.json). "
        "Skor utama adalah relative_score (0-1) yang dinormalisasi per-mahasiswa."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Career Repository & Embedding Cache ----------
_CAREERS_PATH = os.path.join("data", "careers.json")
_career_repo: List[Dict] = []
_career_index: Dict[str, Dict] = {}
_career_ids: List[str] = []
_career_titles: Dict[str, str] = {}
_career_emb: Optional[np.ndarray] = None
_title_emb: Optional[np.ndarray] = None
_default_model_name = "intfloat/multilingual-e5-base"
_RESOLVE_THRESHOLD = 0.78


def _load_careers_from_file() -> List[Dict]:
    if not os.path.exists(_CAREERS_PATH):
        return []
    with open(_CAREERS_PATH, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    if isinstance(doc, dict) and 'fields' in doc:
        flat: List[Dict] = []
        for field in doc['fields']:
            for c in field.get('careers', []):
                flat.append({**c, 'field': field.get('name')})
        return flat
    return doc.get('careers', doc)


def _career_text(c: Dict) -> str:
    skills = ", ".join(c.get('skills', []))
    return f"{c.get('title','')}\n{c.get('description','')}\nSkills: {skills}"


def _precompute_career_embeddings(model_name: str):
    """Precompute and cache career + title embeddings at startup."""
    global _career_emb, _title_emb
    model = load_model(model_name)
    career_texts = [_career_text(c) for c in _career_repo]
    preprocessed = preprocess_batch(career_texts)
    _career_emb = embed_texts(model, preprocessed, is_passage=True)
    title_texts = preprocess_batch([c.get("title", c.get("id", "")) for c in _career_repo])
    _title_emb = embed_texts(model, title_texts, is_passage=True)


def _normalize_to_slug(s: str) -> str:
    """Lowercase, replace whitespace/underscores with hyphens, strip non-slug chars."""
    s = (s or "").strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


def _resolve_career_id(query: str) -> str:
    """Loose match: exact id → normalized id → exact title → semantic title match.

    Raises HTTPException(400) with top suggestions if no layer matches.
    """
    if not query or not query.strip():
        raise HTTPException(status_code=400, detail="target_career_id tidak boleh kosong.")

    if query in _career_index:
        return query

    normalized = _normalize_to_slug(query)
    if normalized and normalized in _career_index:
        return normalized

    q_lower = query.strip().lower()
    for c in _career_repo:
        if c.get("title", "").strip().lower() == q_lower:
            return c["id"]

    if _title_emb is not None and _career_ids:
        try:
            model = load_model(_default_model_name)
            q_emb = embed_texts(model, preprocess_batch([query]), is_passage=False)[0]
            sims = cosine_similarity_matrix(q_emb.reshape(1, -1), _title_emb)[0]
            top_idx = int(np.argmax(sims))
            best_score = float(sims[top_idx])
            if best_score >= _RESOLVE_THRESHOLD:
                return _career_ids[top_idx]
            top3 = np.argsort(-sims)[:3]
            suggestions = "; ".join(
                f"'{_career_ids[i]}' ({_career_titles.get(_career_ids[i], _career_ids[i])}, skor {float(sims[i]):.2f})"
                for i in top3
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"target_career_id '{query}' tidak cocok dengan karier mana pun "
                    f"(skor terbaik {best_score:.2f} < {_RESOLVE_THRESHOLD}). "
                    f"Mungkin yang Anda maksud: {suggestions}."
                ),
            )
        except HTTPException:
            raise
        except Exception:
            pass

    raise HTTPException(
        status_code=400,
        detail=f"target_career_id '{query}' tidak ditemukan. Gunakan GET /careers untuk daftar lengkap.",
    )


@app.on_event("startup")
def _startup():
    global _career_repo, _career_index, _career_ids, _career_titles
    _career_repo = _load_careers_from_file()
    _career_index = {c['id']: c for c in _career_repo if 'id' in c}
    _career_ids = [c['id'] for c in _career_repo]
    _career_titles = {c['id']: c.get('title', c['id']) for c in _career_repo}
    if _career_repo:
        _precompute_career_embeddings(_default_model_name)
    _gemini_key = os.environ.get("GEMINI_API_KEY") or ""
    if _gemini_key:
        print(f"[startup] GEMINI_API_KEY visible to uvicorn process (length={len(_gemini_key)}, prefix={_gemini_key[:8]}...)")
    else:
        print("[startup] GEMINI_API_KEY NOT visible to uvicorn process — ai_plan will return error")


# ---------- Core Logic ----------
def analyze_single_core(
    cv_text: str,
    target_career_id: str,
    topk: int = 5,
    min_sim: float = 0.55,
    thresholds: Thresholds = None,
    model_name: Optional[str] = None,
    include_ai_plan: bool = True,
) -> Dict:
    """Reusable analysis function for a single CV against all careers."""
    thresholds = thresholds or Thresholds()
    model_name = model_name or _default_model_name

    if not _career_repo:
        raise HTTPException(status_code=500, detail="Career repository kosong atau tidak ditemukan.")

    original_query = target_career_id
    target_career_id = _resolve_career_id(target_career_id)

    # Use cached embeddings if model matches, otherwise compute on-the-fly
    if model_name == _default_model_name and _career_emb is not None:
        career_emb = _career_emb
    else:
        model = load_model(model_name)
        career_texts = [_career_text(c) for c in _career_repo]
        preprocessed = preprocess_batch(career_texts)
        career_emb = embed_texts(model, preprocessed, is_passage=True)

    model = load_model(model_name)

    # Embed CV as query
    preprocessed_cv = preprocess_batch([cv_text])
    cv_emb = embed_texts(model, preprocessed_cv, is_passage=False)[0]

    # Similarity (raw cosine) + normalization (relative score 0-1)
    sim_row = cosine_similarity_matrix(cv_emb.reshape(1, -1), career_emb)[0]
    norm_row = normalize_scores_minmax(sim_row)

    target_idx = _career_ids.index(target_career_id)
    target_relative = round(float(norm_row[target_idx]), 4)

    # Ranking sorted by relative_score (descending)
    ranked_indices = np.argsort(-norm_row)[:topk]
    ranked_fmt = [
        {
            "rank": rank,
            "career_id": _career_ids[idx],
            "title": _career_titles.get(_career_ids[idx], _career_ids[idx]),
            "field": _career_index.get(_career_ids[idx], {}).get("field"),
            "score": round(float(norm_row[idx]), 4),
        }
        for rank, idx in enumerate(ranked_indices, 1)
    ]

    # Drift analysis (uses normalized scores internally)
    drift = analyze_drift(
        student_vector=cv_emb,
        career_vectors=career_emb,
        career_ids=_career_ids,
        declared_interest=target_career_id,
        thresholds={
            'tau_high': thresholds.tau_high,
            'tau_mid': thresholds.tau_mid,
            'delta_minor': thresholds.delta_minor,
        }
    )

    best_alt_id = drift.get('best_alt_id')
    best_alt_title = _career_titles.get(best_alt_id, best_alt_id) if best_alt_id else None

    # Recommendations via shared recommender module
    recs = recommend_alternatives(
        sim_row=sim_row,
        career_ids=_career_ids,
        topk=topk,
        min_similarity=min_sim,
    )
    recommendations = [
        {
            "career_id": cid,
            "title": _career_titles.get(cid, cid),
            "field": _career_index.get(cid, {}).get("field"),
            "score": round(rel, 4),
        }
        for cid, _, rel in recs
    ]

    # Skill gap analysis
    target_skills = _career_index.get(target_career_id, {}).get('skills', [])
    skill_gap = analyze_skill_gap(
        cv_text=cv_text,
        skills=target_skills,
        model=model,
        threshold=thresholds.skill_threshold,
    )

    # Career transition context: jika karier terkuat CV berbeda dari target
    top_career_idx = int(np.argmax(norm_row))
    top_career_id = _career_ids[top_career_idx]
    transition_context = None
    if top_career_id != target_career_id:
        top_title = _career_titles.get(top_career_id, top_career_id)
        target_title = _career_titles.get(target_career_id, target_career_id)
        top_score = round(float(norm_row[top_career_idx]), 4)
        target_score = round(float(norm_row[target_idx]), 4)
        score_gap = round(top_score - target_score, 4)

        n_matched = len(skill_gap.get("matched_skills", []))
        n_missing = len(skill_gap.get("missing_skills", []))
        n_total = n_matched + n_missing
        n_upgrade = sum(1 for s in skill_gap.get("missing_skills", []) if s.get("type") == "upgrade")
        n_new = n_missing - n_upgrade

        if n_missing == 0 and n_total > 0:
            summary = (
                f"CV kamu lebih kuat di '{top_title}' (skor {top_score:.0%}), namun sudah match "
                f"{n_matched}/{n_total} skill untuk '{target_title}' (skor {target_score:.0%}). "
                f"Skill teknis sudah mencukupi — perkuat portofolio proyek nyata di bidang "
                f"{target_title} untuk membuktikan kompetensi ke rekruter."
            )
        elif n_missing > 0:
            parts = []
            if n_upgrade:
                parts.append(f"{n_upgrade} skill bisa di-upgrade dari skill yang sudah dimiliki")
            if n_new:
                parts.append(f"{n_new} skill perlu dipelajari dari awal")
            detail = " dan ".join(parts)
            summary = (
                f"CV kamu lebih kuat di '{top_title}' (skor {top_score:.0%}). "
                f"Untuk transisi ke '{target_title}' (skor saat ini {target_score:.0%}), "
                f"terdapat {n_missing} skill yang perlu diisi dari total {n_total} skill: {detail}."
            )
        else:
            summary = (
                f"CV kamu lebih kuat di '{top_title}' (skor {top_score:.0%}) "
                f"dibanding '{target_title}' (skor {target_score:.0%})."
            )

        transition_context = {
            "from_career_id": top_career_id,
            "from_career_title": top_title,
            "from_career_field": _career_index.get(top_career_id, {}).get("field"),
            "from_career_score": top_score,
            "to_career_id": target_career_id,
            "to_career_title": target_title,
            "to_career_field": _career_index.get(target_career_id, {}).get("field"),
            "to_career_score": target_score,
            "score_gap": score_gap,
            "skill_match": {
                "matched": n_matched,
                "missing": n_missing,
                "total": n_total,
                "upgrade": n_upgrade,
                "new": n_new,
                "match_ratio": round(n_matched / n_total, 4) if n_total else 0.0,
            },
            "summary": summary,
        }

    target_block = {
        "career_id": target_career_id,
        "title": _career_titles.get(target_career_id, target_career_id),
        "field": _career_index.get(target_career_id, {}).get("field"),
        "score": target_relative,
    }
    if original_query != target_career_id:
        target_block["resolved_from"] = original_query

    payload = {
        "target": target_block,
        "best_alternative": {
            "career_id": best_alt_id,
            "title": best_alt_title,
            "field": _career_index.get(best_alt_id, {}).get("field") if best_alt_id else None,
            "score": drift.get('best_alt_relative_score'),
            "advantage": drift.get('relative_advantage'),
        },
        "status": drift.get('status'),
        "rationale": drift.get('rationale'),
        "transition_context": transition_context,
        "skill_gap": skill_gap,
        "rankings": ranked_fmt,
        "recommendations": recommendations,
        "thresholds": {
            "tau_high": thresholds.tau_high,
            "tau_mid": thresholds.tau_mid,
            "delta_minor": thresholds.delta_minor,
            "skill_threshold": thresholds.skill_threshold,
        },
        "model": model_name,
    }

    if include_ai_plan:
        payload["ai_plan"] = generate_career_plan(payload)

    return payload


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "careers_loaded": len(_career_repo),
        "embeddings_cached": _career_emb is not None,
    }


@app.get("/careers")
def list_careers():
    """Daftar semua profil karier yang tersedia di server."""
    return [
        {"id": c.get("id"), "title": c.get("title"), "field": c.get("field")}
        for c in _career_repo
    ]


@app.post("/analyze_single")
def analyze_single(req: AnalyzeSingleRequest):
    """Analisis satu CV (teks) terhadap target karier."""
    return analyze_single_core(
        cv_text=req.cv_text,
        target_career_id=req.target_career_id,
        topk=req.topk,
        min_sim=req.min_sim,
        thresholds=req.thresholds,
        model_name=req.model,
        include_ai_plan=req.include_ai_plan,
    )


@app.post("/analyze_cv_file")
async def analyze_cv_file(
    file: UploadFile = File(...),
    target_career_id: str = Form(...),
    topk: int = Form(5),
    min_sim: float = Form(0.55),
    model: Optional[str] = Form('intfloat/multilingual-e5-base'),
    tau_high: float = Form(0.70),
    tau_mid: float = Form(0.40),
    delta_minor: float = Form(0.10),
    skill_threshold: float = Form(0.6),
    include_ai_plan: bool = Form(True),
):
    """Analisis CV dari file upload (PDF/DOCX) terhadap target karier."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File kosong atau tidak dapat dibaca.")

    extracted = extract_text_auto(file.filename or "", content, getattr(file, "content_type", None))
    if isinstance(extracted, dict):
        text = extracted.get("text", "")
    elif isinstance(extracted, (tuple, list)):
        text = extracted[0]
    else:
        text = extracted
    if not isinstance(text, str):
        text = str(text or "")

    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "Teks CV tidak terdeteksi dari file. Pastikan unggahan bertipe PDF/DOCX dan bukan hasil scan gambar. "
                "Jika dokumen berupa scan, gunakan OCR terlebih dahulu."
            ),
        )

    th = Thresholds(tau_high=tau_high, tau_mid=tau_mid, delta_minor=delta_minor, skill_threshold=skill_threshold)

    return analyze_single_core(
        cv_text=text,
        target_career_id=target_career_id,
        topk=topk,
        min_sim=min_sim,
        thresholds=th,
        model_name=model,
        include_ai_plan=include_ai_plan,
    )


@app.post("/analyze_batch")
def analyze_batch(req: AnalyzeBatchRequest):
    """Analisis batch: kirim daftar mahasiswa dan karier sekaligus."""
    careers = req.careers
    students = req.students

    career_ids = [c.id for c in careers]
    career_titles = {c.id: c.title for c in careers}
    career_fields = {c.id: c.field for c in careers}
    career_skills = {c.id: c.skills or [] for c in careers}
    career_texts = [
        f"{c.title}\n{c.description}\nSkills: {', '.join(c.skills or [])}"
        for c in careers
    ]

    model = load_model(req.model or _default_model_name)

    preprocessed_careers = preprocess_batch(career_texts)
    preprocessed_cvs = preprocess_batch([s.cv_text for s in students])

    career_emb = embed_texts(model, preprocessed_careers, is_passage=True)
    student_emb = embed_texts(model, preprocessed_cvs, is_passage=False)

    sim = cosine_similarity_matrix(student_emb, career_emb)

    results = []
    for i, s in enumerate(students):
        norm_row = normalize_scores_minmax(sim[i])

        ranked_indices = np.argsort(-norm_row)[:req.topk]
        ranked = [
            {
                "rank": rank,
                "career_id": career_ids[idx],
                "title": career_titles.get(career_ids[idx], career_ids[idx]),
                "field": career_fields.get(career_ids[idx]),
                "score": round(float(norm_row[idx]), 4),
            }
            for rank, idx in enumerate(ranked_indices, 1)
        ]

        drift = None
        if s.declared_interest and s.declared_interest in career_ids:
            drift = analyze_drift(
                student_vector=student_emb[i],
                career_vectors=career_emb,
                career_ids=career_ids,
                declared_interest=s.declared_interest,
                thresholds={
                    'tau_high': req.thresholds.tau_high,
                    'tau_mid': req.thresholds.tau_mid,
                    'delta_minor': req.thresholds.delta_minor,
                },
            )

        recs = recommend_alternatives(
            sim_row=sim[i],
            career_ids=career_ids,
            topk=req.topk,
            min_similarity=req.min_sim,
        )
        recs_fmt = [
            {
                "career_id": cid,
                "title": career_titles.get(cid, cid),
                "field": career_fields.get(cid),
                "score": round(rel, 4),
            }
            for cid, _, rel in recs
        ]

        # Skill gap analysis
        skill_gap = None
        if s.declared_interest and s.declared_interest in career_skills:
            skill_gap = analyze_skill_gap(
                cv_text=s.cv_text,
                skills=career_skills[s.declared_interest],
                model=model,
                threshold=req.thresholds.skill_threshold,
            )

        # Career transition context
        norm_row_i = normalize_scores_minmax(sim[i])
        top_career_idx = int(np.argmax(norm_row_i))
        top_career_id = career_ids[top_career_idx]
        transition_context = None
        if s.declared_interest and top_career_id != s.declared_interest and skill_gap:
            top_title = career_titles.get(top_career_id, top_career_id)
            target_title = career_titles.get(s.declared_interest, s.declared_interest)
            top_score = round(float(norm_row_i[top_career_idx]), 4)
            target_idx_b = career_ids.index(s.declared_interest)
            target_score = round(float(norm_row_i[target_idx_b]), 4)
            score_gap = round(top_score - target_score, 4)

            n_matched = len(skill_gap.get("matched_skills", []))
            n_missing = len(skill_gap.get("missing_skills", []))
            n_total = n_matched + n_missing
            n_upgrade = sum(1 for sk in skill_gap.get("missing_skills", []) if sk.get("type") == "upgrade")
            n_new = n_missing - n_upgrade

            if n_missing == 0 and n_total > 0:
                summary = (
                    f"CV lebih kuat di '{top_title}' (skor {top_score:.0%}), namun sudah match "
                    f"{n_matched}/{n_total} skill untuk '{target_title}' (skor {target_score:.0%}). "
                    f"Perkuat portofolio proyek nyata di bidang {target_title}."
                )
            elif n_missing > 0:
                parts = []
                if n_upgrade:
                    parts.append(f"{n_upgrade} bisa di-upgrade")
                if n_new:
                    parts.append(f"{n_new} perlu dipelajari dari awal")
                detail = " dan ".join(parts)
                summary = (
                    f"CV lebih kuat di '{top_title}' (skor {top_score:.0%}). "
                    f"Untuk transisi ke '{target_title}' (skor {target_score:.0%}), "
                    f"terdapat {n_missing} skill gap dari {n_total} skill: {detail}."
                )
            else:
                summary = (
                    f"CV lebih kuat di '{top_title}' (skor {top_score:.0%}) "
                    f"dibanding '{target_title}' (skor {target_score:.0%})."
                )

            transition_context = {
                "from_career_id": top_career_id,
                "from_career_title": top_title,
                "from_career_score": top_score,
                "to_career_id": s.declared_interest,
                "to_career_title": target_title,
                "to_career_score": target_score,
                "score_gap": score_gap,
                "skill_match": {
                    "matched": n_matched,
                    "missing": n_missing,
                    "total": n_total,
                    "upgrade": n_upgrade,
                    "new": n_new,
                    "match_ratio": round(n_matched / n_total, 4) if n_total else 0.0,
                },
                "summary": summary,
            }

        results.append({
            "student_id": s.id,
            "name": s.name,
            "declared_interest": s.declared_interest,
            "rankings": ranked,
            "drift": drift,
            "transition_context": transition_context,
            "skill_gap": skill_gap,
            "recommendations": recs_fmt,
        })

    return {
        "model": req.model,
        "thresholds": {
            "tau_high": req.thresholds.tau_high,
            "tau_mid": req.thresholds.tau_mid,
            "delta_minor": req.thresholds.delta_minor,
            "skill_threshold": req.thresholds.skill_threshold,
            "min_similarity": req.min_sim,
        },
        "count_students": len(students),
        "count_careers": len(careers),
        "results": results,
    }

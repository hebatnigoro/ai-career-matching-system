from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import json
import os
import numpy as np

from src.preprocess import preprocess_batch
from src.embedding import load_model, embed_texts
from src.similarity import cosine_similarity_matrix, normalize_scores_minmax
from src.drift import analyze_drift
from src.recommender import recommend_alternatives
from src.file_extract import extract_text_auto


# ---------- Data Models ----------
class Thresholds(BaseModel):
    tau_high: float = 0.70
    tau_mid: float = 0.40
    delta_minor: float = 0.10


class AnalyzeSingleRequest(BaseModel):
    cv_text: str
    target_career_id: str
    topk: int = 5
    min_sim: float = 0.55
    thresholds: Thresholds = Thresholds()
    model: Optional[str] = 'intfloat/multilingual-e5-base'


class Career(BaseModel):
    id: str
    title: str
    description: str
    skills: Optional[List[str]] = []


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


# ---------- Career Repository & Embedding Cache ----------
_CAREERS_PATH = os.path.join("data", "careers.json")
_career_repo: List[Dict] = []
_career_index: Dict[str, Dict] = {}
_career_ids: List[str] = []
_career_titles: Dict[str, str] = {}
_career_emb: Optional[np.ndarray] = None
_default_model_name = "intfloat/multilingual-e5-base"


def _load_careers_from_file() -> List[Dict]:
    if not os.path.exists(_CAREERS_PATH):
        return []
    with open(_CAREERS_PATH, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    return doc.get('careers', doc)


def _career_text(c: Dict) -> str:
    skills = ", ".join(c.get('skills', []))
    return f"{c.get('title','')}\n{c.get('description','')}\nSkills: {skills}"


def _precompute_career_embeddings(model_name: str):
    """Precompute and cache career embeddings at startup."""
    global _career_emb
    model = load_model(model_name)
    career_texts = [_career_text(c) for c in _career_repo]
    preprocessed = preprocess_batch(career_texts)
    _career_emb = embed_texts(model, preprocessed, is_passage=True)


@app.on_event("startup")
def _startup():
    global _career_repo, _career_index, _career_ids, _career_titles
    _career_repo = _load_careers_from_file()
    _career_index = {c['id']: c for c in _career_repo if 'id' in c}
    _career_ids = [c['id'] for c in _career_repo]
    _career_titles = {c['id']: c.get('title', c['id']) for c in _career_repo}
    if _career_repo:
        _precompute_career_embeddings(_default_model_name)


# ---------- Core Logic ----------
def analyze_single_core(
    cv_text: str,
    target_career_id: str,
    topk: int = 5,
    min_sim: float = 0.55,
    thresholds: Thresholds = None,
    model_name: Optional[str] = None,
) -> Dict:
    """Reusable analysis function for a single CV against all careers."""
    thresholds = thresholds or Thresholds()
    model_name = model_name or _default_model_name

    if not _career_repo:
        raise HTTPException(status_code=500, detail="Career repository kosong atau tidak ditemukan.")
    if target_career_id not in _career_index:
        raise HTTPException(
            status_code=400,
            detail=f"target_career_id '{target_career_id}' tidak ada di repository. "
                   f"Gunakan GET /careers untuk melihat daftar karier yang tersedia."
        )

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

    # Recommendations: top-k by relative_score, filtered by min_sim on raw cosine
    rec_indices = np.argsort(-norm_row)
    recommendations = []
    for idx in rec_indices:
        if float(sim_row[idx]) < min_sim:
            continue
        recommendations.append({
            "career_id": _career_ids[idx],
            "title": _career_titles.get(_career_ids[idx], _career_ids[idx]),
            "score": round(float(norm_row[idx]), 4),
        })
        if len(recommendations) >= topk:
            break

    return {
        "target": {
            "career_id": target_career_id,
            "title": _career_titles.get(target_career_id, target_career_id),
            "score": target_relative,
        },
        "best_alternative": {
            "career_id": best_alt_id,
            "title": best_alt_title,
            "score": drift.get('best_alt_relative_score'),
            "advantage": drift.get('relative_advantage'),
        },
        "status": drift.get('status'),
        "rationale": drift.get('rationale'),
        "rankings": ranked_fmt,
        "recommendations": recommendations,
        "thresholds": {
            "tau_high": thresholds.tau_high,
            "tau_mid": thresholds.tau_mid,
            "delta_minor": thresholds.delta_minor,
        },
        "model": model_name,
    }


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
    return [{"id": c.get('id'), "title": c.get('title')} for c in _career_repo]


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

    th = Thresholds(tau_high=tau_high, tau_mid=tau_mid, delta_minor=delta_minor)

    return analyze_single_core(
        cv_text=text,
        target_career_id=target_career_id,
        topk=topk,
        min_sim=min_sim,
        thresholds=th,
        model_name=model,
    )


@app.post("/analyze_batch")
def analyze_batch(req: AnalyzeBatchRequest):
    """Analisis batch: kirim daftar mahasiswa dan karier sekaligus."""
    careers = req.careers
    students = req.students

    career_ids = [c.id for c in careers]
    career_titles = {c.id: c.title for c in careers}
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
            student_vector=student_emb[i],
            career_vectors=career_emb,
            career_ids=career_ids,
            topk=req.topk,
            min_similarity=req.min_sim,
        )
        recs_fmt = [
            {
                "career_id": cid,
                "title": career_titles.get(cid, cid),
                "score": round(float(norm_row[career_ids.index(cid)]), 4),
            }
            for cid, score in recs
        ]

        results.append({
            "student_id": s.id,
            "name": s.name,
            "declared_interest": s.declared_interest,
            "rankings": ranked,
            "drift": drift,
            "recommendations": recs_fmt,
        })

    return {
        "model": req.model,
        "thresholds": {
            "tau_high": req.thresholds.tau_high,
            "tau_mid": req.thresholds.tau_mid,
            "delta_minor": req.thresholds.delta_minor,
            "min_similarity": req.min_sim,
        },
        "count_students": len(students),
        "count_careers": len(careers),
        "results": results,
    }

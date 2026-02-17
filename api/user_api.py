from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import json
import os

from src.preprocess import preprocess_batch
from src.embedding import load_model, embed_texts
from src.similarity import cosine_similarity_matrix, rank_topk
from src.drift import analyze_drift
from src.recommender import recommend_alternatives
from src.file_extract import extract_text_auto


# ---------- Data Models (User-Facing) ----------
class Thresholds(BaseModel):
    tau_high: float = 0.70
    tau_mid: float = 0.60
    delta_minor: float = 0.08


class AnalyzeSingleRequest(BaseModel):
    cv_text: str
    target_career_id: str
    topk: int = 5
    min_sim: float = 0.55
    thresholds: Thresholds = Thresholds()
    model: Optional[str] = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


# ---------- FastAPI App ----------
app = FastAPI(
    title="Career Path Drift API (User-Facing)",
    version="1.0.0",
    description=(
        "API sederhana untuk pengguna akhir: input CV (cv_text) + target karier (target_career_id). "
        "Sistem membandingkan CV dengan seluruh profil karier (repo server), menghitung similarity ke target, "
        "mencari alternatif terbaik, menghitung advantage, mendeteksi drift, dan memberikan rekomendasi."),
)


# ---------- Career Repository (Server-Side) ----------
_CAREERS_PATH = os.path.join("data", "careers.json")
_career_repo: List[Dict] = []
_career_index: Dict[str, Dict] = {}


def _load_careers_from_file() -> List[Dict]:
    if not os.path.exists(_CAREERS_PATH):
        return []
    with open(_CAREERS_PATH, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    careers = doc.get('careers', doc)
    return careers


@app.on_event("startup")
def _startup():
    global _career_repo, _career_index
    _career_repo = _load_careers_from_file()
    _career_index = {c['id']: c for c in _career_repo if 'id' in c}


# ---------- Utility ----------
def _career_text(c: Dict) -> str:
    skills = ", ".join(c.get('skills', []))
    return f"{c.get('title','')}\n{c.get('description','')}\nSkills: {skills}"


# ---------- Core Logic Reusable Function ----------
def analyze_single_core(
    cv_text: str,
    target_career_id: str,
    topk: int = 5,
    min_sim: float = 0.55,
    thresholds: "Thresholds" = None,
    model_name: Optional[str] = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
) -> Dict:
    """Pure function that reuses existing BERT pipeline for a single CV.
    IO-free; suitable for both JSON and file-upload endpoints.
    """
    # Defaults
    thresholds = thresholds or Thresholds()

    # Validate repository and target career id
    if not _career_repo:
        raise HTTPException(status_code=500, detail="Career repository kosong atau tidak ditemukan.")
    if target_career_id not in _career_index:
        raise HTTPException(status_code=400, detail=f"target_career_id '{target_career_id}' tidak ada di repository.")

    # Prepare texts and mappings
    career_ids = [c['id'] for c in _career_repo]
    career_titles = {c['id']: c.get('title', c['id']) for c in _career_repo}
    career_texts = [_career_text(c) for c in _career_repo]

    # Load model (pretrained, tanpa fine-tuning)
    model_name_resolved = model_name or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    model = load_model(model_name_resolved)

    # Preprocess
    preprocessed_careers = preprocess_batch(career_texts)
    preprocessed_cv = preprocess_batch([cv_text])

    # Embeddings (L2-normalized)
    career_emb = embed_texts(model, preprocessed_careers)
    cv_emb = embed_texts(model, preprocessed_cv)[0]

    # Similarity matrix untuk satu CV vs semua karier
    sim_row = cosine_similarity_matrix(cv_emb.reshape(1, -1), career_emb)[0]

    # Similarity ke target karier
    target_idx = career_ids.index(target_career_id)
    similarity_target = float(sim_row[target_idx])

    # Ranking keseluruhan
    rankings = rank_topk(sim_row, career_ids, topk=topk)
    ranked_fmt = [
        {"career_id": cid, "title": career_titles.get(cid, cid), "similarity": round(score, 4)}
        for cid, score in rankings
    ]

    # Drift analysis dengan declared_interest = target_career_id
    drift = analyze_drift(
        student_vector=cv_emb,
        career_vectors=career_emb,
        career_ids=career_ids,
        declared_interest=target_career_id,
        thresholds={
            'tau_high': thresholds.tau_high,
            'tau_mid': thresholds.tau_mid,
            'delta_minor': thresholds.delta_minor,
        }
    )

    # Best alternative (dari drift)
    best_alt_id = drift.get('best_alt_id')
    best_alt_similarity = drift.get('best_alt_similarity')
    best_alt_title = career_titles.get(best_alt_id, best_alt_id) if best_alt_id else None

    # Rekomendasi di atas min_sim
    recs = recommend_alternatives(
        student_vector=cv_emb,
        career_vectors=career_emb,
        career_ids=career_ids,
        topk=topk,
        min_similarity=min_sim,
    )
    recommendations = [
        {"career_id": cid, "title": career_titles.get(cid, cid), "similarity": round(score, 4)}
        for cid, score in recs
    ]

    # Response user-facing yang sederhana (konsisten dengan /analyze_single)
    return {
        "target": {
            "career_id": target_career_id,
            "title": career_titles.get(target_career_id, target_career_id),
            "similarity": round(similarity_target, 4),
        },
        "best_alternative": {
            "career_id": best_alt_id,
            "title": best_alt_title,
            "similarity": best_alt_similarity,
            "advantage": drift.get('advantage'),
        },
        "status": drift.get('status'),
        "rationale": drift.get('rationale'),
        "rankings": ranked_fmt,
        "recommendations": recommendations,
        "thresholds": {
            "tau_high": thresholds.tau_high,
            "tau_mid": thresholds.tau_mid,
            "delta_minor": thresholds.delta_minor,
            "min_similarity": min_sim,
        },
        "model": model_name_resolved,
    }
# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "careers_loaded": len(_career_repo)}


@app.get("/careers")
def list_careers():
    return [{"id": c.get('id'), "title": c.get('title')} for c in _career_repo]


@app.post("/analyze_single")
def analyze_single(req: AnalyzeSingleRequest):
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
    model: Optional[str] = Form('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'),
    tau_high: float = Form(0.70),
    tau_mid: float = Form(0.60),
    delta_minor: float = Form(0.08),
):
    """User-facing endpoint to upload a CV file (PDF/DOCX), extract text, and reuse the
    /analyze_single pipeline. Ensures IO (file handling) is separated from NLP logic.
    """
    # Read file content
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="File kosong atau tidak dapat dibaca.")

    # Extract text automatically based on content type / filename
    extracted = extract_text_auto(file.filename or "", content, getattr(file, "content_type", None))
    # Support str, (text, source), and {"text": ...} return types
    if isinstance(extracted, dict):
        text = extracted.get("text", "")
    elif isinstance(extracted, (tuple, list)):
        text = extracted[0]
    else:
        text = extracted
    if not isinstance(text, str):
        text = str(text or "")

    # Validate extraction result
    if not text or not text.strip():
        raise HTTPException(
            status_code=400,
            detail=(
                "Teks CV tidak terdeteksi dari file. Pastikan unggahan bertipe PDF/DOCX dan bukan hasil scan gambar. "
                "Jika dokumen berupa scan, gunakan OCR terlebih dahulu."
            ),
        )

    # Thresholds via form values (defaults align with AnalyzeSingleRequest)
    th = Thresholds(tau_high=tau_high, tau_mid=tau_mid, delta_minor=delta_minor)

    # Reuse core logic; return structure identical to /analyze_single
    return analyze_single_core(
        cv_text=text,
        target_career_id=target_career_id,
        topk=topk,
        min_sim=min_sim,
        thresholds=th,
        model_name=model,
    )

from typing import List, Dict, Optional
from fastapi import FastAPI
from pydantic import BaseModel

from src.preprocess import preprocess_batch
from src.embedding import load_model, embed_texts
from src.similarity import cosine_similarity_matrix, rank_topk
from src.drift import analyze_drift
from src.recommender import recommend_alternatives


class Career(BaseModel):
    id: str
    title: str
    description: str
    skills: Optional[List[str]] = []
    
class Hebat(BaseModel):
    id: str
    title: str
    description: str
    skills: Optional[List[str]] = []

class Student(BaseModel):
    id: str
    name: Optional[str] = None
    cv_text: str
    declared_interest: Optional[str] = None


class Thresholds(BaseModel):
    tau_high: float = 0.70
    tau_mid: float = 0.60
    delta_minor: float = 0.08


class AnalyzeRequest(BaseModel):
    students: List[Student]
    careers: List[Career]
    topk: int = 5
    min_sim: float = 0.55
    thresholds: Thresholds = Thresholds()
    model: Optional[str] = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


app = FastAPI(title="Career Path Drift API", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    careers = req.careers
    students = req.students

    career_ids = [c.id for c in careers]
    career_titles = {c.id: c.title for c in careers}
    career_texts = [f"{c.title}\n{c.description}\nSkills: {', '.join(c.skills or [])}" for c in careers]

    model = load_model(req.model or 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

    # Preprocess and embed
    preprocessed_careers = preprocess_batch(career_texts)
    preprocessed_cvs = preprocess_batch([s.cv_text for s in students])

    career_emb = embed_texts(model, preprocessed_careers)
    student_emb = embed_texts(model, preprocessed_cvs)

    sim = cosine_similarity_matrix(student_emb, career_emb)

    results = []
    for i, s in enumerate(students):
        rankings = rank_topk(sim[i], career_ids, topk=req.topk)
        ranked = [
            {
                "career_id": cid,
                "title": career_titles.get(cid, cid),
                "similarity": round(score, 4),
            }
            for cid, score in rankings
        ]

        drift = analyze_drift(
            student_vector=student_emb[i],
            career_vectors=career_emb,
            career_ids=career_ids,
            declared_interest=s.declared_interest,
            thresholds={
                'tau_high': req.thresholds.tau_high,
                'tau_mid': req.thresholds.tau_mid,
                'delta_minor': req.thresholds.delta_minor,
            }
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
                "similarity": round(score, 4),
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
        "count_students": len(students),
        "count_careers": len(careers),
        "results": results,
    }

from typing import List, Tuple
import numpy as np


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


def _skill_overlap(cv_text: str, skills: List[str]) -> float:
    """
    Compute lightweight skill overlap score.
    Returns value in [0, 1].
    """
    if not skills:
        return 0.0

    cv_lower = cv_text.lower()
    hits = 0

    for skill in skills:
        if skill.lower() in cv_lower:
            hits += 1

    return hits / len(skills)

def _calibrate_scores(sim_row: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    """
    Convert similarity scores into a sharper distribution.

    Lower temperature → stronger separation
    Higher temperature → softer ranking
    """

    # Numerical stability
    shifted = sim_row - np.max(sim_row)

    exp_scores = np.exp(shifted / temperature)
    probs = exp_scores / np.sum(exp_scores)

    return probs


def cosine_similarity_matrix(
    student_embeddings: np.ndarray,
    career_embeddings: np.ndarray,
    student_texts: List[str],
    career_skills: List[List[str]],
    alpha: float = 0.75,   # embedding weight
    beta: float = 0.25,    # skill weight
) -> np.ndarray:
    """
    Hybrid similarity:
        alpha * cosine_similarity + beta * skill_overlap
    """

    A_n = l2_normalize(student_embeddings)
    B_n = l2_normalize(career_embeddings)

    cosine_sim = A_n @ B_n.T

    adjusted = np.zeros_like(cosine_sim)

    for i in range(len(student_texts)):
        for j in range(len(career_skills)):
            overlap = _skill_overlap(student_texts[i], career_skills[j])
            adjusted[i, j] = alpha * cosine_sim[i, j] + beta * overlap

    # Apply calibration per student
    for i in range(adjusted.shape[0]):
        adjusted[i] = _calibrate_scores(adjusted[i])

    return adjusted


def rank_topk(sim_row: np.ndarray, ids: List[str], topk: int = 5) -> List[Tuple[str, float]]:
    idx = np.argsort(-sim_row)[:topk]
    return [(ids[i], float(sim_row[i])) for i in idx]

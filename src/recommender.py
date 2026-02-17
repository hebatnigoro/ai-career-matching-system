from typing import List, Tuple
import numpy as np


def recommend_alternatives(
    student_vector: np.ndarray,
    career_vectors: np.ndarray,
    career_ids: List[str],
    topk: int = 5,
    min_similarity: float = 0.55,
) -> List[Tuple[str, float]]:
    """Return top-k alternatives above a minimum similarity threshold."""
    s_norm = student_vector / max(np.linalg.norm(student_vector), 1e-12)
    sims = s_norm @ (career_vectors / np.clip(np.linalg.norm(career_vectors, axis=1, keepdims=True), 1e-12, None)).T
    idx = np.argsort(-sims)
    out = []
    for i in idx:
        sim = float(sims[i])
        if sim < min_similarity:
            continue
        out.append((career_ids[i], sim))
        if len(out) >= topk:
            break
    return out

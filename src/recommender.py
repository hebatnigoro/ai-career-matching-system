from typing import List, Tuple
import numpy as np


def recommend_alternatives(
    sim_row: np.ndarray,          # ← PRECOMPUTED SCORES
    career_ids: List[str],
    topk: int = 5,
    min_similarity: float = 0.55,
) -> List[Tuple[str, float]]:

    idx = np.argsort(-sim_row)

    out = []
    for i in idx:
        sim = float(sim_row[i])
        if sim < min_similarity:
            continue

        out.append((career_ids[i], sim))

        if len(out) >= topk:
            break

    return out

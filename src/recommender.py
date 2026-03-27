from typing import List, Tuple
import numpy as np

from src.similarity import normalize_scores_minmax


def recommend_alternatives(
    sim_row: np.ndarray,
    career_ids: List[str],
    topk: int = 5,
    min_similarity: float = 0.55,
) -> List[Tuple[str, float, float]]:
    """Return top-k career recommendations sorted by relative score.

    Parameters
    ----------
    sim_row : np.ndarray
        Pre-computed cosine similarity row (1-D) for one student vs all careers.
    career_ids : list of str
        Career IDs corresponding to each column in sim_row.
    topk : int
        Maximum number of recommendations to return.
    min_similarity : float
        Minimum raw cosine similarity threshold to filter out noise.

    Returns
    -------
    list of (career_id, raw_similarity, relative_score)
    """
    norm_row = normalize_scores_minmax(sim_row)
    idx = np.argsort(-norm_row)

    out = []
    for i in idx:
        raw_sim = float(sim_row[i])
        if raw_sim < min_similarity:
            continue
        out.append((career_ids[i], raw_sim, float(norm_row[i])))
        if len(out) >= topk:
            break
    return out

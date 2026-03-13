from typing import List, Tuple
import numpy as np


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of A and rows of B.
    Returns matrix M where M[i,j] = cos(A[i], B[j]).
    """
    A_n = l2_normalize(A)
    B_n = l2_normalize(B)
    return A_n @ B_n.T


def normalize_scores_minmax(sim_row: np.ndarray) -> np.ndarray:
    """Per-student min-max normalization to [0, 1].

    Maps the best career to 1.0 and worst to 0.0 for each student,
    making drift detection robust against models with narrow score ranges.
    """
    sim_min = np.min(sim_row)
    sim_max = np.max(sim_row)
    sim_range = sim_max - sim_min
    if sim_range < 1e-9:
        return np.full_like(sim_row, 0.5)
    return (sim_row - sim_min) / sim_range


def rank_topk(sim_row: np.ndarray, ids: List[str], topk: int = 5) -> List[Tuple[str, float]]:
    """Rank career ids by similarity for a single student row."""
    idx = np.argsort(-sim_row)[:topk]
    return [(ids[i], float(sim_row[i])) for i in idx]

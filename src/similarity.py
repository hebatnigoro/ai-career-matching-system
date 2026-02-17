from typing import List, Tuple
import numpy as np


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between rows of A and rows of B.
    Assumes inputs are already L2-normalized. If not, normalizes them.
    Returns matrix M where M[i,j] = cos(A[i], B[j]).
    """
    # Ensure normalization to safeguard if upstream changed
    A_n = l2_normalize(A)
    B_n = l2_normalize(B)
    return A_n @ B_n.T


def rank_topk(sim_row: np.ndarray, ids: List[str], topk: int = 5) -> List[Tuple[str, float]]:
    """Rank career ids by similarity for a single student row."""
    idx = np.argsort(-sim_row)[:topk]
    return [(ids[i], float(sim_row[i])) for i in idx]

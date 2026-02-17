from functools import lru_cache
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


@lru_cache(maxsize=2)
def load_model(model_name: str) -> SentenceTransformer:
    """Load and cache a SentenceTransformer model.
    For improved retrieval/ranking quality, use E5 base by default.
    """
    # Force E5 base for this system; ignore incoming model_name to keep API flow unchanged
    model = SentenceTransformer("intfloat/multilingual-e5-base")
    # Track model name for downstream formatting decisions (E5 query/passage prefixes)
    try:
        setattr(model, "kilocode_model_name", "intfloat/multilingual-e5-base")
    except Exception:
        pass
    return model


def _prefix_for_e5(texts: List[str], is_passage_hint: bool) -> List[str]:
    """
    Apply E5 retrieval prefixes.
    If is_passage_hint True, use 'passage: ' else 'query: '.
    """
    prefix = "passage: " if is_passage_hint else "query: "
    return [prefix + t for t in texts]


def _needs_e5_prefix(model: SentenceTransformer) -> bool:
    name = getattr(model, "kilocode_model_name", "")
    return isinstance(name, str) and name.startswith("intfloat/multilingual-e5")


def embed_texts(model: SentenceTransformer, texts: List[str]) -> np.ndarray:
    """Encode a list of texts into L2-normalized embeddings (numpy array).
    For E5 models, apply query/passage prefix formatting for optimal retrieval performance.
    Heuristic: inputs that include 'Skills:' are treated as passages (career definitions);
    others as queries (CVs). This keeps the API analyze() flow unchanged.
    """
    to_encode = texts
    if _needs_e5_prefix(model):
        # Heuristic: careers are constructed with 'Skills:' in api/main.py
        is_passage_hint = any("Skills:" in t for t in texts)
        to_encode = _prefix_for_e5(texts, is_passage_hint)

    embeddings = model.encode(
        to_encode,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)

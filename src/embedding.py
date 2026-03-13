from functools import lru_cache
from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np


@lru_cache(maxsize=4)
def load_model(model_name: str = "intfloat/multilingual-e5-base") -> SentenceTransformer:
    """Load and cache a SentenceTransformer model."""
    model = SentenceTransformer(model_name)
    model._loaded_model_name = model_name
    return model


def _is_e5_model(model: SentenceTransformer) -> bool:
    name = getattr(model, "_loaded_model_name", "")
    return isinstance(name, str) and "e5" in name.lower()


def embed_texts(
    model: SentenceTransformer,
    texts: List[str],
    is_passage: bool = False,
) -> np.ndarray:
    """Encode texts into L2-normalized embeddings.

    Args:
        model: SentenceTransformer model instance.
        texts: List of raw text strings.
        is_passage: If True, texts are treated as passages/documents (career profiles).
                    If False, texts are treated as queries (CVs).
                    Only relevant for E5 models that require query/passage prefixes.
    """
    to_encode = texts
    if _is_e5_model(model):
        prefix = "passage: " if is_passage else "query: "
        to_encode = [prefix + t for t in texts]

    embeddings = model.encode(
        to_encode,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return embeddings.astype(np.float32)

import re
from typing import List

_WS = re.compile(r"\s+")


def preprocess_text(text: str) -> str:
    """Minimal preprocessing preserving semantics.
    - Normalize whitespace
    - Strip leading/trailing spaces
    - Optionally lowercase (kept to avoid over-normalization)
    """
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = _WS.sub(" ", t)
    # t = t.lower()  # Optional: uncomment if consistent casing preferred
    return t


def preprocess_batch(texts: List[str]) -> List[str]:
    return [preprocess_text(t) for t in texts]

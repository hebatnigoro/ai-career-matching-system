import re
import unicodedata
from typing import List


# --- Compiled regex patterns ---
_WS = re.compile(r"\s+")
_EMAIL = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(
    r"(?:\+62|62|0)[\s\-]?\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4}"  # ID phone numbers
    r"|"
    r"\(?\d{3,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{3,4}"  # generic phone patterns
)
_REPEATED_PUNCT = re.compile(r"([^\w\s])\1{2,}")  # e.g. "!!!" → "!"
_BULLET = re.compile(r"^[\s]*[•●○▪►\-\*]\s*", re.MULTILINE)


def preprocess_text(text: str) -> str:
    """Preprocessing ringan yang menjaga makna semantik.

    Langkah-langkah:
    1. Normalisasi unicode (NFKC) — menyeragamkan karakter seperti em-dash, smart quotes
    2. Hapus email — noise, bukan informasi semantik karier
    3. Hapus URL — noise, bukan informasi semantik karier
    4. Hapus nomor telepon — PII, bukan informasi semantik karier
    5. Normalisasi bullet points — menyeragamkan format list
    6. Hapus punctuation berulang — e.g. "!!!" → "!"
    7. Normalisasi whitespace — spasi ganda, tab, newline → spasi tunggal
    8. Strip leading/trailing spaces

    Tidak dilakukan (by design):
    - Stemming/lemmatization — agar embedding menangkap bentuk kata asli
    - Stopword removal — agar konteks kalimat tetap utuh untuk BERT
    - Lowercase — model multilingual E5 sudah case-aware
    """
    if not isinstance(text, str):
        return ""

    t = text.strip()
    if not t:
        return ""

    # 1. Normalisasi unicode (NFKC)
    t = unicodedata.normalize("NFKC", t)

    # 2. Hapus email
    t = _EMAIL.sub("", t)

    # 3. Hapus URL
    t = _URL.sub("", t)

    # 4. Hapus nomor telepon
    t = _PHONE.sub("", t)

    # 5. Normalisasi bullet points → hapus simbol bullet
    t = _BULLET.sub("", t)

    # 6. Hapus punctuation berulang
    t = _REPEATED_PUNCT.sub(r"\1", t)

    # 7. Normalisasi whitespace
    t = _WS.sub(" ", t)

    # 8. Strip
    t = t.strip()

    return t


def preprocess_batch(texts: List[str]) -> List[str]:
    """Batch preprocessing untuk daftar teks."""
    return [preprocess_text(t) for t in texts]

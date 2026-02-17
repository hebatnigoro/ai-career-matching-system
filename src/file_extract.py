from typing import Optional
from tempfile import NamedTemporaryFile
import io
import re

from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document

_WS = re.compile(r"\s+")


def _normalize_text(t: str) -> str:
    t = t.strip()
    t = _WS.sub(" ", t)
    return t


def _extract_pdf_pdfminer(content: bytes) -> str:
    try:
        with NamedTemporaryFile(delete=True, suffix='.pdf') as tmp:
            tmp.write(content)
            tmp.flush()
            text = pdf_extract_text(tmp.name) or ''
        return text
    except Exception:
        return ""


def _extract_pdf_pymupdf(content: bytes) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=content, filetype="pdf")
        text = "\n".join(page.get_text("text") or "" for page in doc)
        doc.close()
        return text
    except Exception:
        return ""


def _extract_pdf_pdfplumber(content: bytes) -> str:
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        return text
    except Exception:
        return ""


def _extract_pdf_ocr(content: bytes) -> str:
    """OCR fallback for scanned PDFs. Requires Tesseract installed in system PATH."""
    try:
        import fitz  # PyMuPDF
        from PIL import Image
        import pytesseract
        doc = fitz.open(stream=content, filetype="pdf")
        texts = []
        for page in doc:
            # Render at 200% scale to improve OCR
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            mode = "RGBA" if pix.alpha else "RGB"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            texts.append(pytesseract.image_to_string(img))
        doc.close()
        return "\n".join(texts)
    except Exception:
        return ""


def extract_text_auto(filename: str, content: bytes, content_type: Optional[str]):
    """Extract text from PDF or DOCX bytes with robust fallbacks.
    Returns (text, source) where source indicates extractor used.
    """
    ct = (content_type or '').lower()
    fn = (filename or '').lower()

    # PDF branch with fallback chain: pdfminer -> PyMuPDF -> pdfplumber -> OCR
    if 'pdf' in ct or fn.endswith('.pdf'):
        # Try pdfminer first
        text = _extract_pdf_pdfminer(content)
        source = 'pdfminer'
        norm = _normalize_text(text)
        if len(norm) < 50:  # heuristic: too short likely image or protected
            text2 = _extract_pdf_pymupdf(content)
            if len(_normalize_text(text2)) > len(norm):
                text, source, norm = text2, 'pymupdf', _normalize_text(text2)
        if len(norm) < 50:
            text3 = _extract_pdf_pdfplumber(content)
            if len(_normalize_text(text3)) > len(norm):
                text, source, norm = text3, 'pdfplumber', _normalize_text(text3)
        if len(norm) < 50:
            text4 = _extract_pdf_ocr(content)
            if len(_normalize_text(text4)) > len(norm):
                text, source, norm = text4, 'ocr', _normalize_text(text4)
        return norm, source

    # DOCX branch
    if 'wordprocessingml' in ct or 'docx' in ct or fn.endswith('.docx'):
        try:
            doc = Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
            return _normalize_text(text), 'docx'
        except Exception:
            return "", 'docx_error'

    # Unknown type
    return "", 'unknown'

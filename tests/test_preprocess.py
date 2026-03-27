"""Unit tests for src/preprocess.py"""
import pytest
from src.preprocess import preprocess_text, preprocess_batch


class TestPreprocessText:
    def test_empty_string(self):
        assert preprocess_text("") == ""

    def test_none_input(self):
        assert preprocess_text(None) == ""

    def test_whitespace_normalization(self):
        assert preprocess_text("hello   world\t\nfoo") == "hello world foo"

    def test_strip(self):
        assert preprocess_text("  hello  ") == "hello"

    def test_email_removal(self):
        result = preprocess_text("Contact me at john@example.com for details")
        assert "john@example.com" not in result
        assert "Contact me at" in result
        assert "for details" in result

    def test_url_removal(self):
        result = preprocess_text("Visit https://github.com/user for code")
        assert "https://github.com/user" not in result
        assert "Visit" in result

    def test_phone_removal_id_format(self):
        result = preprocess_text("Hubungi 0822-3061-2003 atau +62 812 3456 7890")
        assert "0822-3061-2003" not in result
        assert "+62 812 3456 7890" not in result

    def test_unicode_normalization(self):
        # NFKC normalizes full-width chars, e.g. full-width A -> A
        text = "Backend\uff21PI"  # full-width A
        result = preprocess_text(text)
        assert "\uff21" not in result
        assert "API" in result  # normalized to regular A

    def test_repeated_punctuation(self):
        assert "!!!" not in preprocess_text("Great!!!")
        assert "!" in preprocess_text("Great!!!")

    def test_bullet_removal(self):
        text = "Skills:\n• Python\n• JavaScript\n• Go"
        result = preprocess_text(text)
        assert "•" not in result
        assert "Python" in result

    def test_preserves_meaningful_content(self):
        text = "Developed RESTful APIs using Node.js and PostgreSQL"
        result = preprocess_text(text)
        assert result == text  # should be unchanged

    def test_preserves_case(self):
        text = "Backend Developer at Google"
        assert preprocess_text(text) == text


class TestPreprocessBatch:
    def test_batch(self):
        texts = ["hello  world", "foo@bar.com test", "  strip  "]
        results = preprocess_batch(texts)
        assert len(results) == 3
        assert results[0] == "hello world"
        assert "foo@bar.com" not in results[1]
        assert results[2] == "strip"

    def test_empty_batch(self):
        assert preprocess_batch([]) == []

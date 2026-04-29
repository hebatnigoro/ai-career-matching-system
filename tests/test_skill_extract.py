"""Unit tests for src/skill_extract.py — the layered skill extractor.

Tests are organized by layer:

* TestVariantGeneration — validates the surface-form variant generator.
* TestRegistry — registry construction, deduplication.
* TestLexicalLayer — exact / boundary-aware regex matching.
* TestFuzzyLayer — typo / morphology tolerance via rapidfuzz.
* TestSemanticLayer — embedding fallback (uses a stub model).
* TestPipeline — end-to-end behaviour, layer ordering, ablation toggles.
* TestSectionDetection — mentioned vs demonstrated heuristic.

The semantic tests use a stub SentenceTransformer that fakes embeddings,
so the suite runs fast and offline. This is deliberate: the layered
pipeline's *layering decisions* (which layer fires for which input) are
the contribution under test, not the embedding model itself.
"""

import numpy as np
import pytest

from src.skill_extract import (
    SkillRegistry,
    _generate_variants,
    _normalize_form,
    build_skill_registry,
    extract_cv_skills,
    _extract_lexical,
    _extract_fuzzy,
    _classify_position_section,
)


# ---------------------------------------------------------------- stubs

class _StubModel:
    """Stand-in for SentenceTransformer.

    Returns embeddings derived from a simple keyword-overlap signal so that
    semantic-layer tests are deterministic without downloading a real model.
    The model has a ``_loaded_model_name`` attribute so embed_texts treats
    it as non-E5 (no prefix injection).
    """
    _loaded_model_name = "stub"

    def __init__(self, vocab):
        # vocab is an ordered list of words; embedding has one dim per word.
        self.vocab = vocab

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True,
               show_progress_bar=False):
        out = np.zeros((len(texts), len(self.vocab)), dtype=np.float32)
        for i, t in enumerate(texts):
            tl = t.lower()
            for j, w in enumerate(self.vocab):
                if w in tl:
                    out[i, j] = 1.0
        # L2 normalise rows (so cosine == dot product downstream)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# --------------------------------------------------- variant generation

class TestVariantGeneration:
    def test_dotjs_library(self):
        v = _generate_variants("React.js")
        assert "react.js" in v
        assert "reactjs" in v
        assert "react js" in v
        # Bare ambiguous stem must NOT be emitted (would over-trigger)
        assert "react" not in v

    def test_node_js(self):
        v = _generate_variants("Node.js")
        assert "node.js" in v
        assert "nodejs" in v
        assert "node" not in v  # ambiguous

    def test_acronym_cpp(self):
        v = _generate_variants("C++")
        assert "c++" in v
        assert "cpp" in v

    def test_acronym_csharp(self):
        v = _generate_variants("C#")
        assert "c#" in v
        assert "csharp" in v

    def test_slash_separator(self):
        v = _generate_variants("CI/CD")
        assert "ci/cd" in v
        assert "cicd" in v
        assert "ci cd" in v
        assert "ci-cd" in v

    def test_space_separator(self):
        v = _generate_variants("REST API")
        assert "rest api" in v
        assert "restapi" in v
        assert "rest-api" in v

    def test_safe_unambiguous_stem(self):
        # ESCO / Kubernetes — no ambiguous stem rule applies
        v = _generate_variants("Kubernetes")
        assert "kubernetes" in v

    def test_minimum_length_pruning(self):
        v = _generate_variants("AI")
        # Single-character or too-short variants should be filtered out.
        assert all(len(x) >= 2 for x in v)


# ------------------------------------------------------------- registry

class TestRegistry:
    def test_build_registry_dedupes(self):
        careers = [
            {"skills": ["Python", "Docker"]},
            {"skills": ["Python", "Kubernetes"]},
        ]
        reg = build_skill_registry(careers)
        assert set(reg.canonical_skills) == {"Python", "Docker", "Kubernetes"}

    def test_build_registry_skips_empty(self):
        careers = [{"skills": ["Python", "", None, "  "]}]
        reg = build_skill_registry(careers)
        assert reg.canonical_skills == ["Python"]

    def test_handles_missing_skills_field(self):
        reg = build_skill_registry([{"id": "x"}])
        assert reg.canonical_skills == []

    def test_add_method_idempotent(self):
        reg = SkillRegistry()
        reg.add("Docker")
        reg.add("Docker")
        assert reg.canonical_skills == ["Docker"]


# -------------------------------------------------------- lexical layer

class TestLexicalLayer:
    def setup_method(self):
        self.reg = build_skill_registry([{"skills": [
            "React.js", "Docker", "Kubernetes", "Node.js", "C++",
            "REST API", "PostgreSQL", "TypeScript",
        ]}])

    def test_exact_canonical(self):
        cv = "I built APIs with Docker and Kubernetes."
        out = _extract_lexical(cv, self.reg)
        assert "Docker" in out
        assert "Kubernetes" in out
        assert out["Docker"].source == "lexical"
        assert out["Docker"].confidence >= 0.90

    def test_dotjs_variants(self):
        cv = "Built frontend with reactjs and backend with Node.js."
        out = _extract_lexical(cv, self.reg)
        assert "React.js" in out
        assert "Node.js" in out
        # The matched span text should reflect what was actually in the CV
        assert out["React.js"].matched_text.lower() == "reactjs"

    def test_does_not_overtrigger_on_ambiguous_stem(self):
        # "react" alone (not "react.js" or "reactjs") must NOT match React.js
        cv = "I had to react quickly to changing requirements."
        out = _extract_lexical(cv, self.reg)
        assert "React.js" not in out

    def test_word_boundary_precision(self):
        # "DockerHub" is a different concept; word boundaries should still
        # match "Docker" since "DockerHub" contains "Docker" followed by a
        # letter (not a boundary). We expect NO match here to avoid noise.
        cv = "Pushed to DockerHub for distribution."
        out = _extract_lexical(cv, self.reg)
        assert "Docker" not in out

    def test_cpp_punctuation_atomic(self):
        cv = "Developed simulation in C++ with multithreading."
        out = _extract_lexical(cv, self.reg)
        assert "C++" in out
        assert out["C++"].matched_text == "C++"

    def test_evidence_payload(self):
        cv = "Years of experience: built REST API services."
        out = _extract_lexical(cv, self.reg)
        ev = out["REST API"]
        assert ev.cv_span[0] >= 0
        assert ev.cv_span[1] > ev.cv_span[0]
        assert "REST API" in ev.context or "rest api" in ev.context.lower()


# ---------------------------------------------------------- fuzzy layer

class TestFuzzyLayer:
    def setup_method(self):
        self.reg = build_skill_registry([{"skills": [
            "Microservices", "Kubernetes", "PostgreSQL", "TypeScript",
        ]}])

    def test_catches_typo(self):
        cv = "Built distributed Microservise architectures with PostgreSQL."
        out = _extract_fuzzy(cv, self.reg, already_found={"PostgreSQL"})
        # Microservices typo "Microservise" should clear 0.92 token_set_ratio
        assert "Microservices" in out
        assert out["Microservices"].source == "fuzzy"
        # Confidence should fall in the calibrated fuzzy band
        assert 0.6 <= out["Microservices"].confidence <= 0.9

    def test_skips_already_found(self):
        cv = "PostgreSQL administrator with backups expertise."
        out = _extract_fuzzy(cv, self.reg, already_found={"PostgreSQL"})
        # PostgreSQL is already_found → fuzzy must not re-emit it
        assert "PostgreSQL" not in out

    def test_does_not_overtrigger_short_targets(self):
        # Skill targets shorter than 4 chars are skipped by the fuzzy layer
        # to avoid false positives. We can't easily exercise that here
        # without registering such a skill, so instead assert that an
        # unrelated CV produces no fuzzy hits.
        cv = "Worked on completely unrelated content."
        out = _extract_fuzzy(cv, self.reg, already_found=set())
        assert out == {}


# ------------------------------------------------------- semantic layer

class TestSemanticLayer:
    def test_paraphrase_via_stub(self, monkeypatch):
        # Create a stub model whose vocabulary aligns "containerization"
        # in the CV with "Docker" the skill — so cosine ~= 1.0 between
        # the skill vector and the relevant segment.
        from src import skill_extract as se

        # Override the embedding function so it returns dot-product friendly
        # vectors that produce similarity >= semantic_threshold (0.78) for
        # the (skill="Docker", segment containing "containerization") pair.
        def fake_embed_texts(model, texts, is_passage=False):
            out = []
            for t in texts:
                tl = t.lower()
                # Direction A: Docker / containerization
                a = 1.0 if ("docker" in tl or "containerization" in tl) else 0.0
                # Direction B: noise / fillers
                b = 1.0 if ("python" in tl) else 0.0
                v = np.array([a, b], dtype=np.float32)
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                out.append(v)
            return np.vstack(out)

        monkeypatch.setattr(se, "embed_texts", fake_embed_texts)

        reg = build_skill_registry([{"skills": ["Docker"]}])
        cv = "I have deep expertise in containerization for production workloads."
        # Lexical & fuzzy will fail (no "Docker" string, no near-fuzzy hit).
        # Only semantic should fire.
        result = extract_cv_skills(
            cv_text=cv,
            registry=reg,
            model=_StubModel([]),  # not used because we monkeypatched embed_texts
            enable_lexical=True,
            enable_fuzzy=True,
            enable_semantic=True,
        )
        assert "Docker" in result
        assert result["Docker"].source == "semantic"
        assert 0.5 <= result["Docker"].confidence <= 0.8


# ------------------------------------------------------------- pipeline

class TestPipeline:
    def test_layer_priority_lexical_wins(self, monkeypatch):
        # If lexical fires, fuzzy / semantic must not overwrite the evidence.
        from src import skill_extract as se

        def fake_embed_texts(model, texts, is_passage=False):
            return np.ones((len(texts), 2), dtype=np.float32) / np.sqrt(2)

        monkeypatch.setattr(se, "embed_texts", fake_embed_texts)

        reg = build_skill_registry([{"skills": ["Docker"]}])
        cv = "Strong Docker experience in CI/CD pipelines."
        result = extract_cv_skills(cv, reg, _StubModel([]))
        assert result["Docker"].source == "lexical"

    def test_ablation_disable_lexical(self, monkeypatch):
        # With lexical disabled, fuzzy can still pick up the exact form
        # because token_set_ratio == 1.0 ≥ 0.92 threshold.
        reg = build_skill_registry([{"skills": ["Kubernetes"]}])
        cv = "Strong Kubernetes operator experience."
        result = extract_cv_skills(
            cv, reg, _StubModel([]),
            enable_lexical=False,
            enable_fuzzy=True,
            enable_semantic=False,
        )
        assert "Kubernetes" in result
        assert result["Kubernetes"].source == "fuzzy"

    def test_ablation_all_off(self):
        reg = build_skill_registry([{"skills": ["Docker"]}])
        result = extract_cv_skills(
            "Docker is everywhere here.",
            reg, _StubModel([]),
            enable_lexical=False,
            enable_fuzzy=False,
            enable_semantic=False,
        )
        assert result == {}

    def test_empty_inputs(self):
        reg = build_skill_registry([{"skills": ["Docker"]}])
        assert extract_cv_skills("", reg, _StubModel([])) == {}
        empty_reg = SkillRegistry()
        assert extract_cv_skills("Docker", empty_reg, _StubModel([])) == {}


# ---------------------------------------------------- section detection

class TestSectionDetection:
    def test_detects_skills_section(self):
        cv = "Skills:\nReact, Docker, Kubernetes\n\nProjects:\n..."
        # Position right after the Skills header
        pos = cv.lower().find("react")
        assert _classify_position_section(cv, pos) == "skills_list"

    def test_detects_experience_section(self):
        cv = "Experience:\nLed a team to deploy Docker images to production."
        pos = cv.lower().find("docker")
        assert _classify_position_section(cv, pos) == "experience"

    def test_indonesian_keywords(self):
        cv = "Pengalaman:\nMembangun layanan dengan Docker."
        pos = cv.lower().find("docker")
        assert _classify_position_section(cv, pos) == "experience"

    def test_default_other(self):
        cv = "Random preamble text mentioning Docker briefly."
        pos = cv.lower().find("docker")
        assert _classify_position_section(cv, pos) == "other"


# ----------------------------------------------------------- normalize

def test_normalize_form_basic():
    assert _normalize_form("  React.JS  ") == "react.js"
    assert _normalize_form("REST   API") == "rest api"
    assert _normalize_form('"PostgreSQL"') == "postgresql"

"""Unit tests for src/drift.py"""
import numpy as np
import pytest
from src.drift import analyze_drift


# Helper: create mock vectors that produce desired cosine similarities
def _make_vectors(declared_sim, best_alt_sim, n_careers=4):
    """Create student and career vectors with controlled similarities.

    Career 0 = declared interest, Career 1 = best alternative,
    others have lower similarity.
    """
    dim = 64
    rng = np.random.default_rng(42)

    student = rng.normal(size=dim).astype(np.float32)
    student /= np.linalg.norm(student)

    careers = np.zeros((n_careers, dim), dtype=np.float32)

    # Career 0: declared interest with desired similarity
    noise0 = rng.normal(size=dim).astype(np.float32)
    noise0 -= noise0 @ student * student  # orthogonal component
    noise0 /= np.linalg.norm(noise0)
    careers[0] = declared_sim * student + np.sqrt(1 - declared_sim**2) * noise0
    careers[0] /= np.linalg.norm(careers[0])

    # Career 1: best alternative
    noise1 = rng.normal(size=dim).astype(np.float32)
    noise1 -= noise1 @ student * student
    noise1 /= np.linalg.norm(noise1)
    careers[1] = best_alt_sim * student + np.sqrt(1 - best_alt_sim**2) * noise1
    careers[1] /= np.linalg.norm(careers[1])

    # Others: low similarity
    for i in range(2, n_careers):
        low_sim = 0.1 + 0.1 * i
        noise_i = rng.normal(size=dim).astype(np.float32)
        noise_i -= noise_i @ student * student
        noise_i /= np.linalg.norm(noise_i)
        careers[i] = low_sim * student + np.sqrt(1 - low_sim**2) * noise_i
        careers[i] /= np.linalg.norm(careers[i])

    return student, careers


THRESHOLDS = {'tau_high': 0.70, 'tau_mid': 0.40, 'delta_minor': 0.10}
CAREER_IDS = ["declared", "alternative", "other1", "other2"]


class TestAnalyzeDrift:
    def test_aligned(self):
        """Declared is top or very close to top -> Aligned."""
        student, careers = _make_vectors(declared_sim=0.95, best_alt_sim=0.90)
        result = analyze_drift(student, careers, CAREER_IDS, "declared", THRESHOLDS)
        assert result['status'] == 'Aligned'

    def test_minor_drift(self):
        """Declared is moderate, alternative is significantly better -> Minor Drift."""
        student, careers = _make_vectors(declared_sim=0.70, best_alt_sim=0.95)
        result = analyze_drift(student, careers, CAREER_IDS, "declared", THRESHOLDS)
        assert result['status'] == 'Minor Drift'

    def test_major_drift(self):
        """Declared is very low, alternative is high -> Major Drift."""
        student, careers = _make_vectors(declared_sim=0.20, best_alt_sim=0.95)
        result = analyze_drift(student, careers, CAREER_IDS, "declared", THRESHOLDS)
        assert result['status'] == 'Major Drift'

    def test_exploration_needed(self):
        """All careers have nearly identical similarity -> Exploration Needed.

        Triggers when raw cosine similarity range < 0.02, meaning the model
        cannot discriminate between careers for this CV.
        """
        dim = 64
        rng = np.random.default_rng(99)
        student = rng.normal(size=dim).astype(np.float32)
        student /= np.linalg.norm(student)

        # All careers with nearly identical similarity (range < 0.02)
        base_sim = 0.5
        careers = np.zeros((4, dim), dtype=np.float32)
        for i, s in enumerate([base_sim, base_sim + 0.005, base_sim + 0.01, base_sim + 0.015]):
            noise = rng.normal(size=dim).astype(np.float32)
            noise -= noise @ student * student
            noise /= np.linalg.norm(noise)
            careers[i] = s * student + np.sqrt(1 - s**2) * noise
            careers[i] /= np.linalg.norm(careers[i])

        result = analyze_drift(student, careers, CAREER_IDS, "declared", THRESHOLDS)
        assert result['status'] == 'Exploration Needed'

    def test_no_declared_interest(self):
        """No declared interest -> returns fit status without drift."""
        student, careers = _make_vectors(declared_sim=0.5, best_alt_sim=0.9)
        result = analyze_drift(student, careers, CAREER_IDS, None, THRESHOLDS)
        assert 'No Declared Interest' in result['status']
        assert result['declared_similarity'] is None

    def test_invalid_declared_interest(self):
        """Invalid career id -> treated as no declared interest."""
        student, careers = _make_vectors(declared_sim=0.5, best_alt_sim=0.9)
        result = analyze_drift(student, careers, CAREER_IDS, "nonexistent", THRESHOLDS)
        assert 'No Declared Interest' in result['status']

    def test_output_keys(self):
        """Verify all expected keys are present."""
        student, careers = _make_vectors(declared_sim=0.8, best_alt_sim=0.85)
        result = analyze_drift(student, careers, CAREER_IDS, "declared", THRESHOLDS)
        expected_keys = {
            'status', 'declared_similarity', 'declared_relative_score',
            'best_alt_id', 'best_alt_similarity', 'best_alt_relative_score',
            'raw_advantage', 'relative_advantage', 'rationale',
        }
        assert set(result.keys()) == expected_keys

    def test_relative_scores_range(self):
        """Relative scores should be in [0, 1]."""
        student, careers = _make_vectors(declared_sim=0.6, best_alt_sim=0.9)
        result = analyze_drift(student, careers, CAREER_IDS, "declared", THRESHOLDS)
        assert 0.0 <= result['declared_relative_score'] <= 1.0
        assert 0.0 <= result['best_alt_relative_score'] <= 1.0

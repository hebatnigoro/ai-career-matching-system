"""Unit tests for src/similarity.py"""
import numpy as np
import pytest
from src.similarity import l2_normalize, cosine_similarity_matrix, normalize_scores_minmax, rank_topk


class TestL2Normalize:
    def test_unit_vectors(self):
        mat = np.array([[3.0, 4.0]])
        result = l2_normalize(mat)
        assert np.allclose(np.linalg.norm(result, axis=1), [1.0])

    def test_multiple_rows(self):
        mat = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])
        result = l2_normalize(mat)
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, [1.0, 1.0, 1.0])

    def test_zero_vector(self):
        mat = np.array([[0.0, 0.0]])
        result = l2_normalize(mat)
        # Should not produce NaN/Inf
        assert np.all(np.isfinite(result))


class TestCosineSimilarityMatrix:
    def test_identical_vectors(self):
        A = np.array([[1.0, 0.0, 0.0]])
        sim = cosine_similarity_matrix(A, A)
        assert np.allclose(sim, [[1.0]])

    def test_orthogonal_vectors(self):
        A = np.array([[1.0, 0.0]])
        B = np.array([[0.0, 1.0]])
        sim = cosine_similarity_matrix(A, B)
        assert np.allclose(sim, [[0.0]], atol=1e-7)

    def test_matrix_shape(self):
        A = np.random.randn(3, 10)
        B = np.random.randn(5, 10)
        sim = cosine_similarity_matrix(A, B)
        assert sim.shape == (3, 5)

    def test_similarity_range(self):
        A = np.random.randn(5, 20)
        B = np.random.randn(8, 20)
        sim = cosine_similarity_matrix(A, B)
        assert np.all(sim >= -1.0 - 1e-7)
        assert np.all(sim <= 1.0 + 1e-7)

    def test_symmetry(self):
        A = np.random.randn(4, 10)
        sim = cosine_similarity_matrix(A, A)
        assert np.allclose(sim, sim.T, atol=1e-6)


class TestNormalizeScoresMinmax:
    def test_basic(self):
        row = np.array([0.5, 0.7, 0.9])
        result = normalize_scores_minmax(row)
        assert np.isclose(result[0], 0.0)  # min -> 0
        assert np.isclose(result[2], 1.0)  # max -> 1
        assert np.isclose(result[1], 0.5)  # mid -> 0.5

    def test_all_equal(self):
        row = np.array([0.8, 0.8, 0.8])
        result = normalize_scores_minmax(row)
        assert np.allclose(result, [0.5, 0.5, 0.5])

    def test_two_values(self):
        row = np.array([0.3, 0.7])
        result = normalize_scores_minmax(row)
        assert np.isclose(result[0], 0.0)
        assert np.isclose(result[1], 1.0)

    def test_preserves_order(self):
        row = np.array([0.76, 0.82, 0.79, 0.80])
        result = normalize_scores_minmax(row)
        original_order = np.argsort(-row)
        normalized_order = np.argsort(-result)
        assert np.array_equal(original_order, normalized_order)


class TestRankTopk:
    def test_basic_ranking(self):
        row = np.array([0.5, 0.9, 0.7, 0.3])
        ids = ["a", "b", "c", "d"]
        result = rank_topk(row, ids, topk=2)
        assert result[0] == ("b", pytest.approx(0.9))
        assert result[1] == ("c", pytest.approx(0.7))

    def test_topk_limit(self):
        row = np.array([0.5, 0.9, 0.7, 0.3])
        ids = ["a", "b", "c", "d"]
        result = rank_topk(row, ids, topk=3)
        assert len(result) == 3

    def test_topk_exceeds_length(self):
        row = np.array([0.5, 0.9])
        ids = ["a", "b"]
        result = rank_topk(row, ids, topk=10)
        assert len(result) == 2

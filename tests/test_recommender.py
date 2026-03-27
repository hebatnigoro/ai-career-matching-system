"""Unit tests for src/recommender.py"""
import numpy as np
import pytest
from src.recommender import recommend_alternatives


class TestRecommendAlternatives:
    def setup_method(self):
        self.sim_row = np.array([0.9, 0.7, 0.85, 0.5, 0.3])
        self.career_ids = ["a", "b", "c", "d", "e"]

    def test_returns_topk(self):
        recs = recommend_alternatives(self.sim_row, self.career_ids, topk=3)
        assert len(recs) == 3

    def test_sorted_by_relative_score(self):
        recs = recommend_alternatives(self.sim_row, self.career_ids, topk=5, min_similarity=0.0)
        scores = [rel for _, _, rel in recs]
        assert scores == sorted(scores, reverse=True)

    def test_min_similarity_filter(self):
        recs = recommend_alternatives(self.sim_row, self.career_ids, topk=10, min_similarity=0.6)
        for cid, raw, rel in recs:
            assert raw >= 0.6

    def test_output_format(self):
        recs = recommend_alternatives(self.sim_row, self.career_ids, topk=1)
        assert len(recs) == 1
        cid, raw, rel = recs[0]
        assert isinstance(cid, str)
        assert isinstance(raw, float)
        assert isinstance(rel, float)

    def test_relative_scores_range(self):
        recs = recommend_alternatives(self.sim_row, self.career_ids, topk=5, min_similarity=0.0)
        for _, _, rel in recs:
            assert 0.0 <= rel <= 1.0

    def test_best_has_relative_score_1(self):
        recs = recommend_alternatives(self.sim_row, self.career_ids, topk=5, min_similarity=0.0)
        _, _, top_rel = recs[0]
        assert np.isclose(top_rel, 1.0)

    def test_empty_after_filter(self):
        recs = recommend_alternatives(self.sim_row, self.career_ids, topk=5, min_similarity=0.99)
        assert len(recs) == 0

"""
Unit tests for score.py

Tests factual consistency scoring with hand-crafted scenarios:
- Fully faithful summary (score near 1.0)
- Summary with injected wrong name/number (flagged as unsupported)
- Paraphrased summary (should still match)
"""

import pytest
import numpy as np
from kg_checker.extract import Entity, Fact
from kg_checker.align import EntityAlignment, FactAlignment
from kg_checker.score import (
    compute_entity_score,
    compute_fact_score,
    compute_consistency_score,
    identify_hallucinations,
    compute_scores,
    ScoreResult,
)


class TestEntityScore:
    """Test entity-level scoring."""

    def test_all_entities_aligned(self):
        """Test score when all entities align."""
        alignments = [
            EntityAlignment("apple", "apple", 1.0, "exact"),
            EntityAlignment("banana", "banana", 1.0, "exact"),
        ]

        score, aligned, total = compute_entity_score(alignments)

        assert score == 1.0
        assert aligned == 2
        assert total == 2

    def test_partial_entities_aligned(self):
        """Test score when some entities don't align."""
        alignments = [
            EntityAlignment("apple", "apple", 1.0, "exact"),
            EntityAlignment("unknown", None, 0.0, "none"),
        ]

        score, aligned, total = compute_entity_score(alignments)

        assert score == 0.5
        assert aligned == 1
        assert total == 2

    def test_no_entities_aligned(self):
        """Test score when no entities align."""
        alignments = [
            EntityAlignment("xyz", None, 0.0, "none"),
            EntityAlignment("abc", None, 0.0, "none"),
        ]

        score, aligned, total = compute_entity_score(alignments)

        assert score == 0.0
        assert aligned == 0
        assert total == 2

    def test_empty_entities_returns_nan(self):
        """Test that empty entity list returns NaN."""
        score, aligned, total = compute_entity_score([])

        assert np.isnan(score)
        assert aligned == 0
        assert total == 0


class TestFactScore:
    """Test fact-level scoring."""

    def test_all_facts_aligned(self):
        """Test score when all facts align."""
        alignments = [
            FactAlignment(("s1", "r", "o1"), ("s1", "r", "o1"), 1.0, "exact"),
            FactAlignment(("s2", "r", "o2"), ("s2", "r", "o2"), 1.0, "exact"),
        ]

        score, aligned, total = compute_fact_score(alignments)

        assert score == 1.0
        assert aligned == 2
        assert total == 2

    def test_partial_facts_aligned(self):
        """Test score when some facts don't align."""
        alignments = [
            FactAlignment(("s1", "r", "o1"), ("s1", "r", "o1"), 1.0, "exact"),
            FactAlignment(("s3", "r", "o3"), None, 0.0, "none"),
        ]

        score, aligned, total = compute_fact_score(alignments)

        assert score == 0.5
        assert aligned == 1
        assert total == 2

    def test_no_facts_aligned(self):
        """Test score when no facts align."""
        alignments = [
            FactAlignment(("x", "r", "y"), None, 0.0, "none"),
            FactAlignment(("a", "r", "b"), None, 0.0, "none"),
        ]

        score, aligned, total = compute_fact_score(alignments)

        assert score == 0.0
        assert aligned == 0
        assert total == 2

    def test_empty_facts_returns_nan(self):
        """Test that empty fact list returns NaN."""
        score, aligned, total = compute_fact_score([])

        assert np.isnan(score)
        assert aligned == 0
        assert total == 0


class TestConsistencyScore:
    """Test weighted consistency scoring."""

    def test_balanced_weights(self):
        """Test with equal entity and fact weights."""
        weights = {"entity_weight": 0.5, "fact_weight": 0.5}

        score = compute_consistency_score(0.8, 0.6, weights)

        # (0.5 * 0.8) + (0.5 * 0.6) = 0.4 + 0.3 = 0.7
        assert score == 0.7

    def test_uneven_weights(self):
        """Test with different entity and fact weights."""
        weights = {"entity_weight": 0.4, "fact_weight": 0.6}

        score = compute_consistency_score(1.0, 0.5, weights)

        # (0.4 * 1.0) + (0.6 * 0.5) / (0.4 + 0.6) = 0.4 + 0.3 = 0.7
        assert score == 0.7

    def test_fact_only_score(self):
        """Test when entity score is NaN, uses fact score."""
        weights = {"entity_weight": 0.5, "fact_weight": 0.5}

        score = compute_consistency_score(np.nan, 0.8, weights)

        assert score == 0.8

    def test_entity_only_score(self):
        """Test when fact score is NaN, uses entity score."""
        weights = {"entity_weight": 0.5, "fact_weight": 0.5}

        score = compute_consistency_score(0.8, np.nan, weights)

        assert score == 0.8

    def test_both_nan_returns_nan(self):
        """Test that both NaN returns NaN."""
        weights = {"entity_weight": 0.5, "fact_weight": 0.5}

        score = compute_consistency_score(np.nan, np.nan, weights)

        assert np.isnan(score)


class TestHallucinationIdentification:
    """Test hallucination detection."""

    def test_identify_unsupported_entities(self):
        """Test identifying unsupported entities."""
        entity_alignments = [
            EntityAlignment("apple", "apple", 1.0, "exact"),
            EntityAlignment("unknown", None, 0.0, "none"),
        ]

        unsupported, _ = identify_hallucinations(entity_alignments, [])

        assert "unknown" in [e.lower() for e in unsupported]

    def test_identify_unsupported_facts(self):
        """Test identifying unsupported facts."""
        fact_alignments = [
            FactAlignment(("s1", "r", "o1"), ("s1", "r", "o1"), 1.0, "exact"),
            FactAlignment(("s2", "r", "o2"), None, 0.0, "none"),
        ]

        _, unsupported = identify_hallucinations([], fact_alignments)

        assert len(unsupported) == 1
        assert unsupported[0] == ("s2", "r", "o2")

    def test_identify_multiple_hallucinations(self):
        """Test multiple hallucinations detected."""
        entity_alignments = [
            EntityAlignment("a", None, 0.0, "none"),
            EntityAlignment("b", None, 0.0, "none"),
        ]
        fact_alignments = [
            FactAlignment(("s1", "r", "o1"), ("s1", "r", "o1"), 1.0, "exact"),
            FactAlignment(("s2", "r", "o2"), None, 0.0, "none"),
            FactAlignment(("s3", "r", "o3"), None, 0.0, "none"),
        ]

        unsupported_entities, unsupported_facts = identify_hallucinations(
            entity_alignments, fact_alignments
        )

        assert len(unsupported_entities) == 2
        assert len(unsupported_facts) == 2

    def test_no_hallucinations(self):
        """Test when all items are supported."""
        entity_alignments = [
            EntityAlignment("a", "a", 1.0, "exact"),
        ]
        fact_alignments = [
            FactAlignment(("s", "r", "o"), ("s", "r", "o"), 1.0, "exact"),
        ]

        unsupported_entities, unsupported_facts = identify_hallucinations(
            entity_alignments, fact_alignments
        )

        assert len(unsupported_entities) == 0
        assert len(unsupported_facts) == 0


class TestComprehensiveScenarios:
    """Test with comprehensive scenarios from the spec."""

    @pytest.fixture
    def weights(self):
        """Standard scoring weights from config."""
        return {"entity_weight": 0.4, "fact_weight": 0.6}

    def test_fully_faithful_summary(self, weights):
        """
        Test a fully faithful summary - score should be near 1.0.

        Source: "Barack Obama was born in Hawaii. He worked as president."
        Summary: "Barack Obama was born in Hawaii. He served as president."
        """
        # All entities and facts align
        entity_alignments = [
            EntityAlignment("Barack Obama", "Barack Obama", 1.0, "exact"),
            EntityAlignment("Hawaii", "Hawaii", 1.0, "exact"),
        ]

        fact_alignments = [
            FactAlignment(("Barack Obama", "born", "Hawaii"),
                         ("Barack Obama", "born", "Hawaii"), 1.0, "exact"),
            FactAlignment(("Barack Obama", "president", "Hawaii"),  # paraphrased
                         ("Barack Obama", "president", "Hawaii"), 1.0, "exact"),
        ]

        result = compute_scores(entity_alignments, fact_alignments, weights)

        assert result.consistency_score >= 0.9, \
            f"Expected high score for faithful summary, got {result.consistency_score}"
        assert len(result.unsupported_entities) == 0
        assert len(result.unsupported_facts) == 0

    def test_summary_with_wrong_name(self, weights):
        """
        Test a summary with wrong entity name - should be flagged.

        Source: "Barack Obama was born in Hawaii."
        Summary: "George Bush was born in Hawaii."
        """
        entity_alignments = [
            EntityAlignment("George Bush", None, 0.0, "none"),  # Wrong name
        ]

        fact_alignments = [
            FactAlignment(("George Bush", "born", "Hawaii"), None, 0.0, "none"),
        ]

        result = compute_scores(entity_alignments, fact_alignments, weights)

        assert result.consistency_score < 0.5, \
            f"Expected low score for wrong entity, got {result.consistency_score}"
        assert "George Bush" in result.unsupported_entities

    def test_summary_with_wrong_number(self, weights):
        """
        Test a summary with wrong number - should be flagged.

        Source: "The company has 100 employees."
        Summary: "The company has 500 employees."
        """
        entity_alignments = [
            EntityAlignment("company", "company", 1.0, "exact"),
            EntityAlignment("500", None, 0.0, "none"),  # Wrong number
        ]

        fact_alignments = [
            FactAlignment(("company", "has", "500"), None, 0.0, "none"),
        ]

        result = compute_scores(entity_alignments, fact_alignments, weights)

        assert result.consistency_score < 0.5, \
            f"Expected low score for wrong number, got {result.consistency_score}"
        assert len(result.unsupported_entities) > 0

    def test_paraphrased_summary(self, weights):
        """
        Test a paraphrased summary - should still match via fuzzy.

        Source: "Apple released the iPhone in 2007."
        Summary: "The company Apple came out with iPhone back in 2007."
        """
        entity_alignments = [
            EntityAlignment("Apple", "Apple", 1.0, "exact"),
            EntityAlignment("iPhone", "iPhone", 1.0, "exact"),
            EntityAlignment("2007", "2007", 1.0, "exact"),
        ]

        # Facts might not match exactly due to paraphrasing
        # but the entities should align
        fact_alignments = [
            FactAlignment(("Apple", "released", "iPhone"),
                         ("Apple", "released", "iPhone"), 1.0, "fuzzy"),
        ]

        result = compute_scores(entity_alignments, fact_alignments, weights)

        assert result.consistency_score >= 0.7, \
            f"Expected decent score for paraphrased summary, got {result.consistency_score}"

    def test_empty_summary(self, weights):
        """Test that empty summary returns NaN scores appropriately."""
        result = compute_scores([], [], weights)

        # Empty summary should have low/nan scores
        assert result.total_entities == 0
        assert result.total_facts == 0
        assert np.isnan(result.consistency_score) or result.consistency_score == 0.0


class TestScoreResult:
    """Test ScoreResult dataclass."""

    @pytest.fixture
    def weights(self):
        """Standard scoring weights from config."""
        return {"entity_weight": 0.4, "fact_weight": 0.6}

    def test_score_result_fields(self, weights):
        """Test that ScoreResult contains all expected fields."""
        entity_alignments = [
            EntityAlignment("a", "a", 1.0, "exact"),
        ]
        fact_alignments = [
            FactAlignment(("s", "r", "o"), ("s", "r", "o"), 1.0, "exact"),
        ]

        result = compute_scores(entity_alignments, fact_alignments, weights)

        assert hasattr(result, 'entity_precision')
        assert hasattr(result, 'fact_precision')
        assert hasattr(result, 'consistency_score')
        assert hasattr(result, 'unsupported_entities')
        assert hasattr(result, 'unsupported_facts')
        assert hasattr(result, 'aligned_entities')
        assert hasattr(result, 'aligned_facts')
        assert hasattr(result, 'total_entities')
        assert hasattr(result, 'total_facts')
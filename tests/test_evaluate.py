"""
Unit tests for evaluate.py

Tests evaluation pipeline including ROUGE scores, factual consistency,
ablation study, and human evaluation metrics.
"""

import pytest
import numpy as np
from kg_checker.evaluate import (
    compute_rouge_scores,
    compute_ablation_scores,
    generate_evaluation_report
)


class TestRougeScores:
    """Test ROUGE score computation."""

    def test_empty_inputs(self):
        """Test ROUGE with empty inputs."""
        scores = compute_rouge_scores([], [])

        assert scores['rouge1'] == 0.0
        assert scores['rouge2'] == 0.0
        assert scores['rougeL'] == 0.0

    def test_identical_summaries(self):
        """Test ROUGE with identical summaries (perfect match)."""
        summaries = ["The quick brown fox jumps over the lazy dog."]
        references = ["The quick brown fox jumps over the lazy dog."]

        scores = compute_rouge_scores(summaries, references)

        # Identical summaries should have near-perfect scores
        assert scores['rouge1'] >= 0.95
        assert scores['rouge2'] >= 0.95
        assert scores['rougeL'] >= 0.95

    def test_completely_different_summaries(self):
        """Test ROUGE with completely different summaries."""
        summaries = ["The cat sat on the mat."]
        references = ["A completely different text about nothing related."]

        scores = compute_rouge_scores(summaries, references)

        # Very different texts should have low scores
        assert scores['rouge1'] < 0.5

    def test_partial_match(self):
        """Test ROUGE with partial overlap."""
        summaries = ["Apple released the new iPhone in 2024."]
        references = ["Apple released iPhone in 2024."]

        scores = compute_rouge_scores(summaries, references)

        # Partial match should give reasonable scores
        assert 0.5 <= scores['rouge1'] <= 1.0

    def test_multiple_summaries(self):
        """Test ROUGE with multiple summaries."""
        summaries = [
            "The quick brown fox jumps.",
            "A quick dog runs."
        ]
        references = [
            "The quick fox jumps over.",
            "A fast dog runs."
        ]

        scores = compute_rouge_scores(summaries, references)

        assert scores['rouge1'] > 0
        assert scores['rouge2'] >= 0
        assert scores['rougeL'] > 0


class TestAblationStudy:
    """Test ablation study computation."""

    @pytest.fixture
    def mock_summaries(self):
        """Mock summaries for testing."""
        return [
            "Barack Obama was born in Hawaii.",
            "Apple released the iPhone.",
            "The Eiffel Tower is in Paris."
        ]

    @pytest.fixture
    def mock_sources(self):
        """Mock sources for testing."""
        return [
            "Barack Hussein Obama II is an American politician who served as the 44th president.",
            "Apple Inc. is an American multinational technology company.",
            "The Eiffel Tower is a wrought-iron lattice tower in Paris."
        ]

    @pytest.fixture
    def mock_config(self):
        """Mock configuration."""
        return {
            'alignment': {
                'fuzzy_threshold': 85,
                'embedding_threshold': 0.75
            },
            'scoring': {
                'entity_weight': 0.4,
                'fact_weight': 0.6
            }
        }

    @pytest.fixture
    def mock_nlp(self):
        """Mock spaCy model."""
        import spacy
        return spacy.load("en_core_web_sm")

    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pytest.skip("Embedding model not available")

    def test_ablation_returns_valid_scores(self, mock_summaries, mock_sources,
                                           mock_config, mock_nlp, mock_embedding_model):
        """Test that ablation returns valid score ranges."""
        results = compute_ablation_scores(
            mock_summaries, mock_sources, mock_nlp, mock_config, mock_embedding_model
        )

        # All scores should be in valid range [0, 1]
        assert 0 <= results['entity_only']['mean_score'] <= 1
        assert 0 <= results['fact_only']['mean_score'] <= 1
        assert 0 <= results['combined']['mean_score'] <= 1

    def test_ablation_entity_only_vs_combined(self, mock_summaries, mock_sources,
                                              mock_config, mock_nlp, mock_embedding_model):
        """Test that entity-only vs combined comparison makes sense."""
        results = compute_ablation_scores(
            mock_summaries, mock_sources, mock_nlp, mock_config, mock_embedding_model
        )

        # Combined score should use configured weights
        entity_only = results['entity_only']['mean_score']
        fact_only = results['fact_only']['mean_score']
        combined = results['combined']['mean_score']

        # Combined should be weighted average (within reason)
        entity_weight = mock_config['scoring']['entity_weight']
        fact_weight = mock_config['scoring']['fact_weight']

        # Combined should be between min and max of entity/fact
        min_score = min(entity_only, fact_only)
        max_score = max(entity_only, fact_only)

        if not np.isnan(combined):
            assert min_score <= combined <= max_score + 0.1  # Allow small tolerance

    def test_ablation_empty_inputs(self, mock_config, mock_nlp, mock_embedding_model):
        """Test ablation with empty inputs."""
        results = compute_ablation_scores(
            [], [], mock_nlp, mock_config, mock_embedding_model
        )

        assert results['entity_only']['mean_score'] == 0.0
        assert results['fact_only']['mean_score'] == 0.0
        assert results['combined']['mean_score'] == 0.0


class TestEvaluationReport:
    """Test evaluation report generation."""

    def test_report_generation(self):
        """Test that report generation works."""
        results = {
            'abstractive': {
                'factual_consistency': {
                    'mean_consistency_score': 0.85,
                    'mean_entity_precision': 0.90,
                    'mean_fact_precision': 0.80,
                    'pct_summaries_with_hallucinations': 15.5
                },
                'rouge': {
                    'rouge1': 0.88,
                    'rouge2': 0.75,
                    'rougeL': 0.82
                }
            },
            'extractive': {
                'factual_consistency': {
                    'mean_consistency_score': 0.78,
                    'mean_entity_precision': 0.85,
                    'mean_fact_precision': 0.72,
                    'pct_summaries_with_hallucinations': 22.3
                },
                'rouge': {
                    'rouge1': 0.82,
                    'rouge2': 0.68,
                    'rougeL': 0.76
                }
            }
        }

        report = generate_evaluation_report(results)

        # Check report contains expected sections
        assert "EVALUATION REPORT" in report
        assert "ABSTRACTIVE SUMMARIZATION" in report
        assert "EXTRACTIVE SUMMARIZATION" in report
        assert "0.85" in report  # consistency score
        assert "0.88" in report  # rouge1

    def test_report_generation_single_system(self):
        """Test report generation for single system."""
        results = {
            'mean_consistency_score': 0.82,
            'mean_entity_precision': 0.88,
            'mean_fact_precision': 0.76,
            'pct_summaries_with_hallucinations': 18.5
        }

        report = generate_evaluation_report(results)

        assert "EVALUATION REPORT" in report
        assert "ABSTRACTIVE" not in report  # Should not mention comparison
        assert "0.82" in report

    def test_report_empty_input(self):
        """Test report with minimal input."""
        results = {
            'abstractive': {
                'factual_consistency': {
                    'mean_consistency_score': 0.0,
                    'mean_entity_precision': 0.0,
                    'mean_fact_precision': 0.0,
                    'pct_summaries_with_hallucinations': 100.0
                },
                'rouge': {
                    'rouge1': 0.0,
                    'rouge2': 0.0,
                    'rougeL': 0.0
                }
            },
            'extractive': {
                'factual_consistency': {
                    'mean_consistency_score': 0.0,
                    'mean_entity_precision': 0.0,
                    'mean_fact_precision': 0.0,
                    'pct_summaries_with_hallucinations': 100.0
                },
                'rouge': {
                    'rouge1': 0.0,
                    'rouge2': 0.0,
                    'rougeL': 0.0
                }
            }
        }

        report = generate_evaluation_report(results)

        assert "EVALUATION REPORT" in report
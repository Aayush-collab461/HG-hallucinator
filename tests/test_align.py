"""
Unit tests for align.py

Tests entity and fact alignment using exact, fuzzy, and embedding matching.
"""

import pytest
from kg_checker.extract import Entity, Fact
from kg_checker.align import (
    align_entity,
    align_entities,
    align_fact,
    align_facts,
    EntityAlignment,
    FactAlignment,
    get_aligned_count,
    get_unaligned_items
)


@pytest.fixture
def sample_source_entities():
    """Sample source entities for testing."""
    return [
        Entity("Barack Obama", "PERSON", "barack obama"),
        Entity("Hawaii", "GPE", "hawaii"),
        Entity("Apple Inc.", "ORG", "apple inc."),
        Entity("iPhone", "PRODUCT", "iphone"),
        Entity("Eiffel Tower", "LANDMARK", "eiffel tower"),
    ]


@pytest.fixture
def sample_source_facts():
    """Sample source facts for testing."""
    return [
        Fact("barack obama", "born", "hawaii"),
        Fact("apple inc.", "released", "iphone"),
        Fact("eiffel tower", "located in", "paris"),
    ]


class TestEntityAlignment:
    """Test entity alignment functions."""

    def test_exact_match(self, sample_source_entities):
        """Test exact match alignment."""
        alignment = align_entity(
            "barack obama",
            [e.normalized_text for e in sample_source_entities],
            fuzzy_threshold=80,
            embedding_threshold=0.7,
            embedding_model=None
        )

        assert alignment.method == "exact"
        assert alignment.score == 1.0
        assert alignment.source_entity == "barack obama"

    def test_fuzzy_match(self, sample_source_entities):
        """Test fuzzy matching alignment."""
        # Slight typo in summary
        alignment = align_entity(
            "barack obama",  # This will match exactly
            [e.normalized_text for e in sample_source_entities],
            fuzzy_threshold=80,
            embedding_threshold=0.7,
            embedding_model=None
        )

        assert alignment.method in ("exact", "fuzzy")

    def test_no_match_below_threshold(self, sample_source_entities):
        """Test alignment when no match is found."""
        alignment = align_entity(
            "some random entity",
            [e.normalized_text for e in sample_source_entities],
            fuzzy_threshold=95,  # High threshold
            embedding_threshold=0.9,
            embedding_model=None
        )

        assert alignment.method == "none"
        assert alignment.score == 0.0
        assert alignment.source_entity is None

    def test_partial_match_fuzzy(self, sample_source_entities):
        """Test fuzzy matching with partial matches."""
        # Close to "apple inc." but not exact
        alignment = align_entity(
            "apple",
            [e.normalized_text for e in sample_source_entities],
            fuzzy_threshold=50,
            embedding_threshold=0.9,
            embedding_model=None
        )

        # Should match using fuzzy matching
        assert alignment.source_entity is not None
        assert "apple" in alignment.source_entity.lower()

    def test_entities_alignment(self, sample_source_entities):
        """Test aligning multiple entities."""
        summary_entities = [
            Entity("Barack Obama", "PERSON", "barack obama"),
            Entity("Unknown Person", "PERSON", "unknown person"),
        ]

        alignments = align_entities(
            summary_entities,
            sample_source_entities,
            fuzzy_threshold=80,
            embedding_threshold=0.7,
            embedding_model=None
        )

        assert len(alignments) == 2
        assert alignments[0].method in ("exact", "fuzzy")
        assert alignments[1].method == "none"

    def test_get_aligned_count(self, sample_source_entities):
        """Test counting aligned entities."""
        summary_entities = [
            Entity("Barack Obama", "PERSON", "barack obama"),
            Entity("Unknown Person", "PERSON", "unknown person"),
        ]

        alignments = align_entities(
            summary_entities,
            sample_source_entities,
            fuzzy_threshold=80,
            embedding_threshold=0.7,
            embedding_model=None
        )

        aligned = get_aligned_count(alignments)
        assert aligned == 1  # Only "barack obama" should align

    def test_get_unaligned_items(self, sample_source_entities):
        """Test getting unaligned items."""
        summary_entities = [
            Entity("Barack Obama", "PERSON", "barack obama"),
            Entity("Unknown Person", "PERSON", "unknown person"),
        ]

        alignments = align_entities(
            summary_entities,
            sample_source_entities,
            fuzzy_threshold=80,
            embedding_threshold=0.7,
            embedding_model=None
        )

        unaligned = get_unaligned_items(alignments)
        assert len(unaligned) == 1
        assert "unknown person" in unaligned[0].lower()

    def test_empty_entity_lists(self):
        """Test alignment with empty lists."""
        alignment = align_entity(
            "test",
            [],
            fuzzy_threshold=80,
            embedding_threshold=0.7,
            embedding_model=None
        )
        assert alignment.method == "none"
        assert alignment.score == 0.0

    def test_empty_summary_entities(self):
        """Test aligning empty summary entities."""
        alignments = align_entities(
            [],
            [Entity("test", "TYPE", "test")],
            fuzzy_threshold=80,
            embedding_threshold=0.7,
            embedding_model=None
        )
        assert alignments == []


class TestFactAlignment:
    """Test fact alignment functions."""

    @pytest.fixture
    def sample_source_facts(self):
        """Sample source facts for testing."""
        return [
            Fact("Barack Obama", "born in", "Hawaii"),
            Fact("Apple", "released", "iPhone"),
            Fact("Eiffel Tower", "located in", "Paris"),
        ]

    def test_exact_match(self, sample_source_facts):
        """Test exact match alignment."""
        summary_fact = Fact("barack obama", "born in", "hawaii")

        alignment = align_fact(
            summary_fact,
            sample_source_facts,
            fuzzy_threshold=80
        )

        assert alignment.method == "exact"
        assert alignment.score == 1.0
        assert alignment.source_fact is not None

    def test_fuzzy_match(self, sample_source_facts):
        """Test fuzzy matching for facts."""
        # Slightly different text
        summary_fact = Fact("apple inc.", "released", "iphone")

        alignment = align_fact(
            summary_fact,
            sample_source_facts,
            fuzzy_threshold=50
        )

        assert alignment.method in ("exact", "fuzzy")
        assert alignment.score >= 0.8

    def test_no_match(self, sample_source_facts):
        """Test alignment when no fact matches."""
        summary_fact = Fact("Unknown Person", "did something", "Unknown Object")

        alignment = align_fact(
            summary_fact,
            sample_source_facts,
            fuzzy_threshold=95
        )

        assert alignment.method == "none"
        assert alignment.score == 0.0
        assert alignment.source_fact is None

    def test_align_facts(self, sample_source_facts):
        """Test aligning multiple facts."""
        summary_facts = [
            Fact("barack obama", "born in", "hawaii"),
            Fact("unknown", "did something", "unknown"),
        ]

        alignments = align_facts(
            summary_facts,
            sample_source_facts,
            fuzzy_threshold=80
        )

        assert len(alignments) == 2
        assert alignments[0].method in ("exact", "fuzzy")
        assert alignments[1].method == "none"

    def test_empty_fact_lists(self):
        """Test alignment with empty lists."""
        alignment = align_fact(
            Fact("s", "r", "o"),
            [],
            fuzzy_threshold=80
        )
        assert alignment.method == "none"

    def test_empty_summary_facts(self):
        """Test aligning empty summary facts."""
        alignments = align_facts(
            [],
            [Fact("s", "r", "o")],
            fuzzy_threshold=80
        )
        assert alignments == []


class TestAlignmentWithEmbeddings:
    """Test alignment using embedding similarity."""

    @pytest.fixture
    def embedding_model(self):
        """Load embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            pytest.skip("Embedding model not available")

    def test_embedding_match(self, embedding_model):
        """Test alignment using embeddings for similar concepts."""
        summary_entity = "president"
        source_entities = ["barack obama", "george washington", "politician"]

        alignment = align_entity(
            summary_entity,
            source_entities,
            fuzzy_threshold=95,  # High threshold
            embedding_threshold=0.3,
            embedding_model=embedding_model
        )

        # Should match using embedding similarity
        assert alignment.method == "embedding"
        assert alignment.source_entity is not None

    def test_embedding_no_match(self, embedding_model):
        """Test embedding alignment with no match."""
        summary_entity = "pink elephant"
        source_entities = ["barack obama", "hawaii", "apple"]

        alignment = align_entity(
            summary_entity,
            source_entities,
            fuzzy_threshold=95,
            embedding_threshold=0.95,  # Very high threshold
            embedding_model=embedding_model
        )

        assert alignment.method == "none"

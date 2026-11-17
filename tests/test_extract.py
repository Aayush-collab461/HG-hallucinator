"""
Unit tests for extract.py

Tests entity extraction, fact extraction, and normalization with hand-crafted sentences.
"""

import pytest
import spacy
from kg_checker.extract import (
    extract_entities,
    extract_facts,
    normalize_text,
    is_pronoun,
    Entity,
    Fact
)
from kg_checker.kg import (
    KnowledgeGraph,
    build_kg_from_text,
    build_kg_from_entities_and_facts
)


@pytest.fixture(scope="module")
def nlp():
    """Load spaCy model once for all tests."""
    return spacy.load("en_core_web_sm")


class TestNormalization:
    """Test text normalization functions."""

    def test_normalize_numbers(self):
        """Test number normalization removes commas."""
        assert normalize_text("1,000") == "1000"
        assert normalize_text("1,000,000") == "1000000"
        assert normalize_text("The cost is 5,500 dollars") == "The cost is 5500 dollars"

    def test_normalize_dates(self):
        """Test date abbreviation normalization."""
        assert "January" in normalize_text("Jan 5")
        assert "February" in normalize_text("Feb 2024")
        assert "December" in normalize_text("Dec 31")

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        assert normalize_text("  Hello   world  ") == "Hello world"
        assert normalize_text("Test\n\nwith\nnewlines") == "Test with newlines"

    def test_normalize_empty(self):
        """Test normalization of empty strings."""
        assert normalize_text("") == ""
        assert normalize_text("   ") == ""


class TestEntityExtraction:
    """Test entity extraction with hand-crafted sentences."""

    def test_person_entity(self, nlp):
        """Test extraction of person entities."""
        text = "Barack Obama was born in Hawaii."
        entities = extract_entities(text, nlp)

        # Should extract Barack Obama as PERSON and Hawaii as GPE
        assert len(entities) > 0
        entity_texts = [e.normalized_text.lower() for e in entities]
        assert any("obama" in text for text in entity_texts)

    def test_organization_entity(self, nlp):
        """Test extraction of organization entities."""
        text = "Apple Inc. released a new product."
        entities = extract_entities(text, nlp)

        assert len(entities) > 0
        entity_texts = [e.normalized_text.lower() for e in entities]
        assert any("apple" in text for text in entity_texts)

    def test_location_entity(self, nlp):
        """Test extraction of location entities."""
        text = "The Eiffel Tower is located in Paris."
        entities = extract_entities(text, nlp)

        assert len(entities) > 0
        entity_texts = [e.normalized_text.lower() for e in entities]
        assert any("paris" in text for text in entity_texts)

    def test_number_normalization_in_entities(self, nlp):
        """Test that numbers in entities are normalized."""
        text = "The company earned 1,000 dollars."
        entities = extract_entities(text, nlp)

        # Check if any entity has normalized numbers
        for entity in entities:
            if "," in entity.text and entity.type in ["MONEY", "CARDINAL"]:
                assert "," not in entity.normalized_text

    def test_empty_text(self, nlp):
        """Test extraction from empty text returns empty list."""
        assert extract_entities("", nlp) == []
        assert extract_entities("   ", nlp) == []


class TestFactExtraction:
    """Test fact (SVO triple) extraction."""

    def test_simple_fact(self, nlp):
        """Test extraction of simple subject-verb-object fact."""
        text = "Alice founded the company."
        facts = extract_facts(text, nlp)

        # Should extract (Alice, founded/found, company)
        assert len(facts) > 0
        fact = facts[0]
        assert "alice" in fact.subject.lower()
        assert "found" in fact.relation or "founded" in fact.relation
        assert "company" in fact.object.lower()

    def test_location_fact(self, nlp):
        """Test extraction of location-related fact."""
        text = "The museum is in Boston."
        facts = extract_facts(text, nlp)

        # Should extract fact about museum and Boston
        assert len(facts) > 0
        fact_strings = [f"{f.subject} {f.relation} {f.object}".lower() for f in facts]
        assert any("museum" in s and "boston" in s for s in fact_strings)

    def test_relation_lemmatization(self, nlp):
        """Test that relations are lemmatized."""
        text = "Scientists discovered a new species."
        facts = extract_facts(text, nlp)

        # Relation should be lemmatized: "discovered" -> "discover"
        assert len(facts) > 0
        assert any("discover" in f.relation for f in facts)

    def test_pronoun_filtering(self, nlp):
        """Test that pronouns are filtered out from subjects/objects."""
        text = "She founded it."
        facts = extract_facts(text, nlp)

        # Should return empty or very few facts since subject/object are pronouns
        # spaCy might still extract some based on dependency parse
        # Main goal: no facts with literal "she" or "it"
        for fact in facts:
            assert fact.subject.lower() not in ["she", "he", "it", "they", "i", "you", "we"]
            assert fact.object.lower() not in ["she", "he", "it", "they", "i", "you", "we"]

    def test_empty_text_facts(self, nlp):
        """Test fact extraction from empty text returns empty list."""
        assert extract_facts("", nlp) == []
        assert extract_facts("   ", nlp) == []


class TestKnowledgeGraph:
    """Test KnowledgeGraph construction and operations."""

    def test_add_entity(self):
        """Test adding entities to graph."""
        kg = KnowledgeGraph()
        kg.add_entity("Alice", "PERSON")

        assert "Alice" in kg
        assert kg.node_count() == 1
        assert kg.get_entity_type("Alice") == "PERSON"

    def test_add_fact(self):
        """Test adding facts (edges) to graph."""
        kg = KnowledgeGraph()
        kg.add_fact("Alice", "founded", "Company")

        assert "Alice" in kg
        assert "Company" in kg
        assert kg.edge_count() == 1

        facts = kg.get_facts()
        assert len(facts) == 1
        assert facts[0] == ("Alice", "founded", "Company")

    def test_get_neighbors(self):
        """Test getting neighboring entities."""
        kg = KnowledgeGraph()
        kg.add_fact("Alice", "works_at", "Company")
        kg.add_fact("Company", "located_in", "Boston")

        alice_neighbors = kg.get_neighbors("Alice")
        assert "Company" in alice_neighbors

        company_neighbors = kg.get_neighbors("Company")
        assert "Alice" in company_neighbors
        assert "Boston" in company_neighbors

    def test_get_subgraph(self):
        """Test extracting subgraph."""
        kg = KnowledgeGraph()
        kg.add_fact("Alice", "works_at", "Company")
        kg.add_fact("Bob", "works_at", "Company")
        kg.add_fact("Company", "located_in", "Boston")

        # Get subgraph with only Alice and Company
        subgraph = kg.get_subgraph(["Alice", "Company"])

        assert "Alice" in subgraph
        assert "Company" in subgraph
        assert "Bob" not in subgraph
        assert "Boston" not in subgraph
        assert subgraph.edge_count() == 1

    def test_build_kg_from_text(self, nlp):
        """Test building KG from raw text."""
        text = "Google was founded in California."
        kg = build_kg_from_text(text, nlp)

        # Should have entities and possibly facts
        assert kg.node_count() > 0
        entities = kg.get_entities()
        entity_str = " ".join(entities).lower()
        assert "google" in entity_str or "california" in entity_str

    def test_empty_kg(self):
        """Test empty knowledge graph."""
        kg = KnowledgeGraph()
        assert kg.node_count() == 0
        assert kg.edge_count() == 0
        assert len(kg) == 0
        assert kg.get_entities() == []
        assert kg.get_facts() == []


class TestIntegration:
    """Integration tests with multiple hand-crafted sentences."""

    @pytest.fixture
    def sample_sentences(self):
        """Five hand-crafted test sentences."""
        return [
            "Barack Obama was born in Hawaii in 1961.",
            "Apple Inc. released the iPhone on January 9, 2007.",
            "The Eiffel Tower stands 1,083 feet tall.",
            "Marie Curie won the Nobel Prize in Physics.",
            "Amazon operates in Seattle and employs 1,000,000 workers."
        ]

    def test_extract_from_all_sentences(self, nlp, sample_sentences):
        """Test extraction from all sample sentences."""
        for sentence in sample_sentences:
            entities = extract_entities(sentence, nlp)
            facts = extract_facts(sentence, nlp)

            # Each sentence should extract at least some entities
            assert len(entities) > 0, f"No entities extracted from: {sentence}"

            # Verify entities have types and normalized text
            for entity in entities:
                assert entity.text
                assert entity.type
                assert entity.normalized_text

    def test_build_kgs_from_all_sentences(self, nlp, sample_sentences):
        """Test building KGs from all sample sentences."""
        for sentence in sample_sentences:
            kg = build_kg_from_text(sentence, nlp)

            # Each sentence should produce a non-empty KG
            assert kg.node_count() > 0, f"Empty KG from: {sentence}"

    def test_number_normalization_integration(self, nlp):
        """Test that numbers are normalized in full pipeline."""
        text = "The building is 1,000 feet tall and cost 10,000 dollars."
        kg = build_kg_from_text(text, nlp)

        # Check entities - numbers should be normalized
        entities = kg.get_entities()
        entity_str = " ".join(entities)
        # If numbers appear as entities, they shouldn't have commas
        assert "1,000" not in entity_str
        assert "10,000" not in entity_str

    def test_date_normalization_integration(self, nlp):
        """Test that dates are normalized in full pipeline."""
        text = "The event happened on Jan 15, 2020."
        entities = extract_entities(text, nlp)

        # Check if any date entity has normalized month
        for entity in entities:
            if entity.type == "DATE" and "Jan" in entity.text:
                assert "January" in entity.normalized_text


if __name__ == "__main__":
    # Run pytest
    pytest.main([__file__, "-v"])

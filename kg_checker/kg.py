"""
Knowledge graph construction module.

Builds and manages knowledge graphs from extracted entities and facts.
"""

import networkx as nx
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Knowledge graph representation."""

    def __init__(self):
        """Initialize knowledge graph."""
        self.graph = nx.MultiDiGraph()

    def add_entity(self, entity: str, entity_type: str):
        """Add entity node to graph.

        Args:
            entity: Entity name/text
            entity_type: Entity type (PERSON, GPE, ORG, etc.)
        """
        self.graph.add_node(entity, type=entity_type)

    def add_fact(self, subject: str, predicate: str, obj: str):
        """Add fact as edge to graph.

        Args:
            subject: Subject entity
            predicate: Relation/predicate
            obj: Object entity
        """
        # Ensure subject and object exist as nodes
        if not self.graph.has_node(subject):
            self.graph.add_node(subject)
        if not self.graph.has_node(obj):
            self.graph.add_node(obj)

        # Add edge with relation as attribute
        self.graph.add_edge(subject, obj, relation=predicate)

    def get_subgraph(self, entities: list):
        """Extract subgraph containing specified entities.

        Args:
            entities: List of entity names to include

        Returns:
            New KnowledgeGraph containing only specified entities and their edges
        """
        # Create new KG
        subgraph = KnowledgeGraph()

        # Filter nodes to only those in entities list
        entity_set = set(entities)
        nodes_to_include = [n for n in self.graph.nodes() if n in entity_set]

        # Add nodes with their attributes
        for node in nodes_to_include:
            node_data = self.graph.nodes[node]
            entity_type = node_data.get('type', 'UNKNOWN')
            subgraph.add_entity(node, entity_type)

        # Add edges between included nodes
        for u, v, data in self.graph.edges(data=True):
            if u in entity_set and v in entity_set:
                relation = data.get('relation', 'related_to')
                subgraph.add_fact(u, relation, v)

        return subgraph

    def __contains__(self, entity: str):
        """Check if entity is in graph."""
        return self.graph.has_node(entity)

    def __len__(self):
        """Return number of nodes."""
        return self.graph.number_of_nodes()

    def node_count(self) -> int:
        """Return number of nodes."""
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        """Return number of edges."""
        return self.graph.number_of_edges()

    def get_entity_type(self, entity: str) -> str:
        """Get type of entity.

        Args:
            entity: Entity name

        Returns:
            Entity type or None if not found
        """
        if self.graph.has_node(entity):
            return self.graph.nodes[entity].get('type', 'UNKNOWN')
        return None

    def get_entities(self) -> List[str]:
        """Get all entities in graph.

        Returns:
            List of entity names
        """
        return list(self.graph.nodes())

    def get_facts(self) -> List[Tuple[str, str, str]]:
        """Get all facts (edges) in graph.

        Returns:
            List of (subject, relation, object) tuples
        """
        facts = []
        for u, v, data in self.graph.edges(data=True):
            relation = data.get('relation', 'related_to')
            facts.append((u, relation, v))
        return facts

    def get_neighbors(self, entity: str) -> List[str]:
        """Get neighboring entities (connected via any edge).

        Args:
            entity: Entity name

        Returns:
            List of neighboring entity names
        """
        if not self.graph.has_node(entity):
            return []

        neighbors = set()
        # Get predecessors (entities pointing to this entity)
        neighbors.update(self.graph.predecessors(entity))
        # Get successors (entities this entity points to)
        neighbors.update(self.graph.successors(entity))

        return list(neighbors)


def build_kg_from_text(text: str, nlp_model):
    """Build knowledge graph from text.

    Args:
        text: Input text
        nlp_model: Loaded spaCy model

    Returns:
        KnowledgeGraph with entities and facts extracted from text
    """
    from .extract import extract_entities, extract_facts

    kg = KnowledgeGraph()

    if not text or not text.strip():
        return kg

    try:
        # Extract entities and facts
        entities = extract_entities(text, nlp_model)
        facts = extract_facts(text, nlp_model)

        # Add entities to graph
        for entity in entities:
            kg.add_entity(entity.normalized_text, entity.type)

        # Add facts to graph
        for fact in facts:
            kg.add_fact(fact.subject, fact.relation, fact.object)

        return kg

    except Exception as e:
        logger.warning(f"Failed to build KG from text: {e}")
        return kg


def build_kg_from_entities_and_facts(entities: list, facts: list):
    """Build knowledge graph from pre-extracted entities and facts.

    Args:
        entities: List of Entity objects
        facts: List of Fact objects

    Returns:
        KnowledgeGraph with entities and facts
    """
    kg = KnowledgeGraph()

    # Add entities
    for entity in entities:
        kg.add_entity(entity.normalized_text, entity.type)

    # Add facts
    for fact in facts:
        kg.add_fact(fact.subject, fact.relation, fact.object)

    return kg

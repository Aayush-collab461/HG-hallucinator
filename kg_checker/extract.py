"""
Entity and fact extraction module.

Extracts entities, relations, and facts from text using NLP models.
"""

import re
import logging
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Extracted entity with text and type."""
    text: str
    type: str
    normalized_text: str


@dataclass
class Fact:
    """Extracted fact as (subject, relation, object) triple."""
    subject: str
    relation: str
    object: str


def normalize_text(text: str) -> str:
    """
    Normalize text: lowercase, strip, normalize numbers and dates.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = " ".join(text.split())

    # Normalize numbers: remove all commas
    text = re.sub(r'(\d),(?=\d)', r'\1', text)

    # Normalize common date abbreviations
    date_map = {
        r'\bJan\b': 'January',
        r'\bFeb\b': 'February',
        r'\bMar\b': 'March',
        r'\bApr\b': 'April',
        r'\bMay\b': 'May',
        r'\bJun\b': 'June',
        r'\bJul\b': 'July',
        r'\bAug\b': 'August',
        r'\bSep\b': 'September',
        r'\bOct\b': 'October',
        r'\bNov\b': 'November',
        r'\bDec\b': 'December',
    }

    for abbrev, full in date_map.items():
        text = re.sub(abbrev, full, text, flags=re.IGNORECASE)

    return text


def is_pronoun(token) -> bool:
    """
    Check if token is a pronoun.

    Args:
        token: spaCy token

    Returns:
        True if pronoun
    """
    return token.pos_ == "PRON"


def extract_entities(text: str, nlp_model) -> List[Entity]:
    """
    Extract named entities from text using spaCy NER.

    Args:
        text: Input text
        nlp_model: Loaded spaCy model

    Returns:
        List of Entity objects with text, type, and normalized text
    """
    if not text or not text.strip():
        return []

    try:
        doc = nlp_model(text)
        entities = []

        for ent in doc.ents:
            # Skip empty entities
            if not ent.text.strip():
                continue

            # Skip pronouns
            if all(is_pronoun(token) for token in ent):
                continue

            entity = Entity(
                text=ent.text.strip(),
                type=ent.label_,
                normalized_text=normalize_text(ent.text)
            )
            entities.append(entity)

        return entities

    except Exception as e:
        logger.warning(f"Failed to extract entities: {e}")
        return []


def extract_relations(doc) -> List[Tuple[str, str, str]]:
    """
    Extract semantic relations from spaCy doc using dependency parse.

    Args:
        doc: spaCy Doc object

    Returns:
        List of (subject, relation, object) tuples
    """
    relations = []

    for token in doc:
        # Look for verb as relation
        if token.pos_ == "VERB":
            subject = None
            obj = None

            # Find subject
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass"):
                    if not is_pronoun(child):
                        # Get the full noun phrase if available
                        subject = child.text
                        # Try to get compound subject
                        for subchild in child.children:
                            if subchild.dep_ == "compound":
                                subject = f"{subchild.text} {subject}"

            # Find object
            for child in token.children:
                if child.dep_ in ("dobj", "attr", "pobj"):
                    if not is_pronoun(child):
                        obj = child.text
                        # Try to get compound object
                        for subchild in child.children:
                            if subchild.dep_ == "compound":
                                obj = f"{subchild.text} {obj}"
                # Handle prepositional objects
                elif child.dep_ == "prep":
                    for prep_child in child.children:
                        if prep_child.dep_ == "pobj" and not is_pronoun(prep_child):
                            obj = prep_child.text

            # Create relation if we have both subject and object
            if subject and obj:
                # Lemmatize the relation (verb)
                relation = token.lemma_.lower()
                relations.append((subject, relation, obj))

    # Also handle copula constructions (e.g., "The museum is in Boston")
    # Look for prepositional phrases that relate a subject to an object
    for token in doc:
        if token.dep_ == "prep" and token.pos_ == "ADP":
            # Find the subject of this preposition (the noun it's modifying)
            subject = None
            for child in token.children:
                if child.dep_ == "pobj":
                    # This is the object of the preposition
                    # Now find what the preposition relates
                    obj = child.text
                    # Get compound if present
                    for subchild in child.children:
                        if subchild.dep_ == "compound":
                            obj = f"{subchild.text} {obj}"

                    # Find the subject - look for nmod or pobj parent
                    for ancestor in doc:
                        if ancestor.i < token.i:
                            for anc_child in ancestor.children:
                                if anc_child.dep_ in ("nsubj", "nsubjpass") and anc_child.i < token.i:
                                    if not is_pronoun(anc_child):
                                        subject = anc_child.text
                                        # Get compound subject
                                        for subchild in anc_child.children:
                                            if subchild.dep_ == "compound":
                                                subject = f"{subchild.text} {subject}"
                                        break

                    if subject and obj:
                        # Use preposition as relation
                        relation = token.text.lower()
                        relations.append((subject, relation, obj))
                        break

    return relations


def extract_facts(text: str, nlp_model) -> List[Fact]:
    """
    Extract subject-predicate-object triples from text using dependency parsing.

    Args:
        text: Input text
        nlp_model: Loaded spaCy model

    Returns:
        List of Fact objects with normalized subject, relation, object
    """
    if not text or not text.strip():
        return []

    try:
        doc = nlp_model(text)
        raw_relations = extract_relations(doc)

        facts = []
        for subj, rel, obj in raw_relations:
            # Normalize components
            norm_subj = normalize_text(subj)
            norm_rel = rel.lower()  # Already lemmatized
            norm_obj = normalize_text(obj)

            # Skip if any component is empty after normalization
            if not (norm_subj and norm_rel and norm_obj):
                continue

            fact = Fact(
                subject=norm_subj,
                relation=norm_rel,
                object=norm_obj
            )
            facts.append(fact)

        return facts

    except Exception as e:
        logger.warning(f"Failed to extract facts: {e}")
        return []


def extract_all(text: str, nlp_model) -> Tuple[List[Entity], List[Fact]]:
    """
    Extract both entities and facts from text.

    Args:
        text: Input text
        nlp_model: Loaded spaCy model

    Returns:
        Tuple of (entities, facts)
    """
    entities = extract_entities(text, nlp_model)
    facts = extract_facts(text, nlp_model)
    return entities, facts

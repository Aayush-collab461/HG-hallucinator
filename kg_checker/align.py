"""
Alignment module.

Aligns summary entities and facts against source knowledge graph using
fuzzy matching and semantic similarity.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np

from .extract import Entity, Fact

logger = logging.getLogger(__name__)


@dataclass
class EntityAlignment:
    """Alignment result for an entity."""
    summary_entity: str
    source_entity: Optional[str]
    score: float
    method: str  # 'exact', 'fuzzy', 'embedding', or 'none'


@dataclass
class FactAlignment:
    """Alignment result for a fact."""
    summary_fact: Tuple[str, str, str]
    source_fact: Optional[Tuple[str, str, str]]
    score: float
    method: str


def align_entity(
    summary_entity: str,
    source_entities: List[str],
    fuzzy_threshold: float,
    embedding_threshold: float,
    embedding_model: Optional[SentenceTransformer] = None
) -> EntityAlignment:
    """
    Align a single summary entity to source entities.

    Tries exact match, then fuzzy, then embedding similarity.

    Args:
        summary_entity: Entity text from summary
        source_entities: List of entity texts from source
        fuzzy_threshold: Threshold for fuzzy matching (0-100)
        embedding_threshold: Threshold for embedding similarity (0-1)
        embedding_model: Loaded sentence transformer model (optional)

    Returns:
        EntityAlignment with best match
    """
    if not source_entities:
        return EntityAlignment(summary_entity, None, 0.0, 'none')

    # Normalize for comparison
    summary_norm = summary_entity.lower().strip()

    # 1. Try exact match
    for src_entity in source_entities:
        if summary_norm == src_entity.lower().strip():
            return EntityAlignment(summary_entity, src_entity, 1.0, 'exact')

    # 2. Try fuzzy matching
    best_fuzzy_score = 0.0
    best_fuzzy_match = None

    for src_entity in source_entities:
        score = fuzz.ratio(summary_norm, src_entity.lower().strip())
        if score > best_fuzzy_score:
            best_fuzzy_score = score
            best_fuzzy_match = src_entity

    if best_fuzzy_score >= fuzzy_threshold:
        return EntityAlignment(
            summary_entity,
            best_fuzzy_match,
            best_fuzzy_score / 100.0,  # Normalize to 0-1
            'fuzzy'
        )

    # 3. Try embedding similarity (if model provided)
    if embedding_model is not None:
        try:
            summary_emb = embedding_model.encode([summary_entity])[0]
            source_embs = embedding_model.encode(source_entities)

            # Compute cosine similarities
            similarities = np.dot(source_embs, summary_emb) / (
                np.linalg.norm(source_embs, axis=1) * np.linalg.norm(summary_emb)
            )

            best_idx = np.argmax(similarities)
            best_sim = similarities[best_idx]

            if best_sim >= embedding_threshold:
                return EntityAlignment(
                    summary_entity,
                    source_entities[best_idx],
                    float(best_sim),
                    'embedding'
                )
        except Exception as e:
            logger.warning(f"Embedding alignment failed: {e}")

    # No match found
    return EntityAlignment(summary_entity, None, 0.0, 'none')


def align_entities(
    summary_entities: List[Entity],
    source_entities: List[Entity],
    fuzzy_threshold: float,
    embedding_threshold: float,
    embedding_model: Optional[SentenceTransformer] = None
) -> List[EntityAlignment]:
    """
    Align summary entities to source entities.

    Args:
        summary_entities: Entities from summary
        source_entities: Entities from source
        fuzzy_threshold: Fuzzy matching threshold (0-100)
        embedding_threshold: Embedding similarity threshold (0-1)
        embedding_model: Optional sentence transformer model

    Returns:
        List of EntityAlignment objects
    """
    if not summary_entities:
        return []

    # Extract normalized entity texts
    source_texts = [e.normalized_text for e in source_entities]

    alignments = []
    for summary_entity in summary_entities:
        alignment = align_entity(
            summary_entity.normalized_text,
            source_texts,
            fuzzy_threshold,
            embedding_threshold,
            embedding_model
        )
        alignments.append(alignment)

    return alignments


def align_fact(
    summary_fact: Fact,
    source_facts: List[Fact],
    fuzzy_threshold: float,
    embedding_threshold: float = 0.75,
    embedding_model: Optional[SentenceTransformer] = None
) -> FactAlignment:
    """
    Align a single summary fact to source facts.

    Tries exact match, then fuzzy component matching, then embedding similarity.

    Args:
        summary_fact: Fact from summary
        source_facts: Facts from source
        fuzzy_threshold: Fuzzy matching threshold (0-100)
        embedding_threshold: Embedding similarity threshold (0-1)
        embedding_model: Loaded sentence transformer model (optional)

    Returns:
        FactAlignment with best match
    """
    if not source_facts:
        return FactAlignment(
            (summary_fact.subject, summary_fact.relation, summary_fact.object),
            None,
            0.0,
            'none'
        )

    summary_triple = (
        summary_fact.subject.lower().strip(),
        summary_fact.relation.lower().strip(),
        summary_fact.object.lower().strip()
    )

    best_score = 0.0
    best_match = None
    best_method = 'none'

    for src_fact in source_facts:
        src_triple = (
            src_fact.subject.lower().strip(),
            src_fact.relation.lower().strip(),
            src_fact.object.lower().strip()
        )

        # Check for exact match
        if summary_triple == src_triple:
            return FactAlignment(
                (summary_fact.subject, summary_fact.relation, summary_fact.object),
                (src_fact.subject, src_fact.relation, src_fact.object),
                1.0,
                'exact'
            )

        # Fuzzy matching on each component
        subj_score = fuzz.ratio(summary_triple[0], src_triple[0])
        rel_score = fuzz.ratio(summary_triple[1], src_triple[1])
        obj_score = fuzz.ratio(summary_triple[2], src_triple[2])

        # Average score across components
        avg_score = (subj_score + rel_score + obj_score) / 3.0

        if avg_score > best_score:
            best_score = avg_score
            best_match = (src_fact.subject, src_fact.relation, src_fact.object)
            best_method = 'fuzzy' if avg_score >= fuzzy_threshold else 'none'

    if best_score >= fuzzy_threshold:
        return FactAlignment(
            (summary_fact.subject, summary_fact.relation, summary_fact.object),
            best_match,
            best_score / 100.0,  # Normalize to 0-1
            best_method
        )

    # Try embedding similarity if model provided
    if embedding_model is not None:
        try:
            sum_str = f"{summary_fact.subject} {summary_fact.relation} {summary_fact.object}"
            src_strs = [f"{f.subject} {f.relation} {f.object}" for f in source_facts]

            sum_emb = embedding_model.encode([sum_str])[0]
            src_embs = embedding_model.encode(src_strs)

            similarities = np.dot(src_embs, sum_emb) / (
                np.linalg.norm(src_embs, axis=1) * np.linalg.norm(sum_emb)
            )

            best_idx = np.argmax(similarities)
            best_sim = similarities[best_idx]

            if best_sim >= embedding_threshold:
                src_fact = source_facts[best_idx]
                return FactAlignment(
                    (summary_fact.subject, summary_fact.relation, summary_fact.object),
                    (src_fact.subject, src_fact.relation, src_fact.object),
                    float(best_sim),
                    'embedding'
                )
        except Exception as e:
            logger.warning(f"Fact embedding alignment failed: {e}")

    return FactAlignment(
        (summary_fact.subject, summary_fact.relation, summary_fact.object),
        None,
        0.0,
        'none'
    )


def align_facts(
    summary_facts: List[Fact],
    source_facts: List[Fact],
    fuzzy_threshold: float,
    embedding_threshold: float = 0.75,
    embedding_model: Optional[SentenceTransformer] = None
) -> List[FactAlignment]:
    """
    Align summary facts to source facts.

    Args:
        summary_facts: Facts from summary
        source_facts: Facts from source
        fuzzy_threshold: Fuzzy matching threshold (0-100)
        embedding_threshold: Embedding similarity threshold (0-1)
        embedding_model: Optional sentence transformer model

    Returns:
        List of FactAlignment objects
    """
    if not summary_facts:
        return []

    alignments = []
    for summary_fact in summary_facts:
        alignment = align_fact(
            summary_fact, source_facts,
            fuzzy_threshold, embedding_threshold,
            embedding_model
        )
        alignments.append(alignment)

    return alignments


def get_aligned_count(alignments: List) -> int:
    """
    Count alignments that found a match.

    Args:
        alignments: List of EntityAlignment or FactAlignment objects

    Returns:
        Number of aligned items
    """
    return sum(1 for a in alignments if a.method != 'none')


def get_unaligned_items(alignments: List) -> List:
    """
    Get items that were not aligned.

    Args:
        alignments: List of EntityAlignment or FactAlignment objects

    Returns:
        List of unaligned items
    """
    unaligned = []
    for alignment in alignments:
        if alignment.method == 'none':
            if isinstance(alignment, EntityAlignment):
                unaligned.append(alignment.summary_entity)
            elif isinstance(alignment, FactAlignment):
                unaligned.append(alignment.summary_fact)
    return unaligned

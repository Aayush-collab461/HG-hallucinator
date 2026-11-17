"""
Scoring module.

Computes factual consistency scores and identifies hallucinations.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import logging

from .align import EntityAlignment, FactAlignment, get_unaligned_items

logger = logging.getLogger(__name__)


@dataclass
class ScoreResult:
    """Complete scoring result for a summary."""
    entity_precision: float
    fact_precision: float
    consistency_score: float
    unsupported_entities: List[str]
    unsupported_facts: List[Tuple[str, str, str]]
    aligned_entities: int
    aligned_facts: int
    total_entities: int
    total_facts: int


def compute_entity_score(
    entity_alignments: List[EntityAlignment]
) -> Tuple[float, int, int]:
    """
    Compute entity-level precision score.

    Args:
        entity_alignments: List of EntityAlignment objects

    Returns:
        Tuple of (precision score, aligned count, total count)
    """
    if not entity_alignments:
        return np.nan, 0, 0

    total = len(entity_alignments)
    aligned = sum(1 for a in entity_alignments if a.method != 'none')

    if total == 0:
        return np.nan, aligned, total

    return aligned / total, aligned, total


def compute_fact_score(
    fact_alignments: List[FactAlignment]
) -> Tuple[float, int, int]:
    """
    Compute fact-level precision score.

    Args:
        fact_alignments: List of FactAlignment objects

    Returns:
        Tuple of (precision score, aligned count, total count)
    """
    if not fact_alignments:
        return np.nan, 0, 0

    total = len(fact_alignments)
    aligned = sum(1 for a in fact_alignments if a.method != 'none')

    if total == 0:
        return np.nan, aligned, total

    return aligned / total, aligned, total


def compute_consistency_score(
    entity_score: float,
    fact_score: float,
    weights: Dict[str, float]
) -> float:
    """
    Compute weighted factual consistency score.

    Args:
        entity_score: Entity-level precision score
        fact_score: Fact-level precision score
        weights: Dict with 'entity_weight' and 'fact_weight'

    Returns:
        Weighted combination score (NaN if both inputs are NaN)
    """
    # Handle edge cases
    if np.isnan(entity_score) and np.isnan(fact_score):
        return np.nan
    if np.isnan(entity_score):
        return fact_score
    if np.isnan(fact_score):
        return entity_score

    # Weighted combination
    entity_weight = weights.get('entity_weight', 0.5)
    fact_weight = weights.get('fact_weight', 0.5)

    # Normalize weights to sum to 1
    total_weight = entity_weight + fact_weight
    if total_weight > 0:
        entity_weight /= total_weight
        fact_weight /= total_weight

    return entity_weight * entity_score + fact_weight * fact_score


def identify_hallucinations(
    entity_alignments: List[EntityAlignment],
    fact_alignments: List[FactAlignment]
) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """
    Identify unsupported entities and facts.

    Args:
        entity_alignments: List of EntityAlignment objects
        fact_alignments: List of FactAlignment objects

    Returns:
        Tuple of (unsupported_entities, unsupported_facts)
    """
    # Get unaligned entities
    unsupported_entities = get_unaligned_items(entity_alignments)

    # Get unaligned facts
    unsupported_facts = []
    for alignment in fact_alignments:
        if alignment.method == 'none':
            unsupported_facts.append(alignment.summary_fact)

    return unsupported_entities, unsupported_facts


def compute_scores(
    entity_alignments: List[EntityAlignment],
    fact_alignments: List[FactAlignment],
    weights: Dict[str, float]
) -> ScoreResult:
    """
    Compute all scores and identify hallucinations.

    Args:
        entity_alignments: List of EntityAlignment objects
        fact_alignments: List of FactAlignment objects
        weights: Dict with 'entity_weight' and 'fact_weight'

    Returns:
        ScoreResult with all metrics
    """
    entity_score, aligned_entities, total_entities = compute_entity_score(entity_alignments)
    fact_score, aligned_facts, total_facts = compute_fact_score(fact_alignments)

    consistency_score = compute_consistency_score(entity_score, fact_score, weights)

    unsupported_entities, unsupported_facts = identify_hallucinations(
        entity_alignments, fact_alignments
    )

    return ScoreResult(
        entity_precision=entity_score if not np.isnan(entity_score) else 0.0,
        fact_precision=fact_score if not np.isnan(fact_score) else 0.0,
        consistency_score=consistency_score if not np.isnan(consistency_score) else 0.0,
        unsupported_entities=unsupported_entities,
        unsupported_facts=unsupported_facts,
        aligned_entities=aligned_entities,
        aligned_facts=aligned_facts,
        total_entities=total_entities,
        total_facts=total_facts
    )


def compute_entity_precision(
    entity_alignments: List[EntityAlignment]
) -> float:
    """
    Compute entity precision (deprecated: use compute_entity_score).

    Args:
        entity_alignments: List of EntityAlignment objects

    Returns:
        Precision score (0-1)
    """
    score, _, _ = compute_entity_score(entity_alignments)
    return score if not np.isnan(score) else 0.0


def compute_fact_precision(
    fact_alignments: List[FactAlignment]
) -> float:
    """
    Compute fact precision (deprecated: use compute_fact_score).

    Args:
        fact_alignments: List of FactAlignment objects

    Returns:
        Precision score (0-1)
    """
    score, _, _ = compute_fact_score(fact_alignments)
    return score if not np.isnan(score) else 0.0

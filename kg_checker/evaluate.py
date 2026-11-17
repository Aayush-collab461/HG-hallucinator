"""
Evaluation module.

Evaluates summarization quality using ROUGE and factual consistency metrics.
"""

import logging
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from pathlib import Path

from rouge_score import rouge_scorer
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import pearsonr, spearmanr

from .extract import extract_entities, extract_facts, Entity, Fact
from .align import align_entities, align_facts
from .score import compute_scores, ScoreResult

logger = logging.getLogger(__name__)


def compute_rouge_scores(summaries: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores for summaries.

    Args:
        summaries: List of generated summaries
        references: List of reference summaries

    Returns:
        Dict with ROUGE-1, ROUGE-2, ROUGE-L F1 scores
    """
    if not summaries or not references:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for summary, reference in zip(summaries, references):
        score = scorer.score(reference, summary)
        scores['rouge1'].append(score['rouge1'].fmeasure)
        scores['rouge2'].append(score['rouge2'].fmeasure)
        scores['rougeL'].append(score['rougeL'].fmeasure)

    return {
        'rouge1': np.mean(scores['rouge1']),
        'rouge2': np.mean(scores['rouge2']),
        'rougeL': np.mean(scores['rougeL'])
    }


def evaluate_single_summary(
    summary: str,
    source: str,
    nlp_model,
    fuzzy_threshold: float,
    embedding_threshold: float,
    weights: Dict[str, float],
    embedding_model=None
) -> Tuple[ScoreResult, List[Entity], List[Fact], List[Entity], List[Fact]]:
    """
    Evaluate a single summary against its source.

    Returns:
        Tuple of (score_result, summary_entities, summary_facts, source_entities, source_facts)
    """
    # Extract from source
    source_entities = extract_entities(source, nlp_model)
    source_facts = extract_facts(source, nlp_model)

    # Extract from summary
    summary_entities = extract_entities(summary, nlp_model)
    summary_facts = extract_facts(summary, nlp_model)

    # Align
    entity_alignments = align_entities(
        summary_entities, source_entities,
        fuzzy_threshold, embedding_threshold,
        embedding_model
    )

    fact_alignments = align_facts(
        summary_facts, source_facts,
        fuzzy_threshold
    )

    # Score
    scores = compute_scores(entity_alignments, fact_alignments, weights)

    return scores, summary_entities, summary_facts, source_entities, source_facts


def evaluate_factual_consistency(
    summaries: List[str],
    sources: List[str],
    nlp_model,
    config: Dict,
    embedding_model=None
) -> Dict:
    """
    Evaluate factual consistency of summaries.

    Args:
        summaries: List of generated summaries
        sources: List of source documents
        nlp_model: spaCy model
        config: Configuration dict with thresholds and weights
        embedding_model: Optional sentence transformer model

    Returns:
        Dict with evaluation results and statistics
    """
    fuzzy_threshold = config['alignment']['fuzzy_threshold']
    embedding_threshold = config['alignment']['embedding_threshold']
    weights = {
        'entity_weight': config['scoring']['entity_weight'],
        'fact_weight': config['scoring']['fact_weight']
    }

    results = []
    all_scores = []
    entity_types_count = defaultdict(int)
    entity_types_unsupported = defaultdict(int)

    summaries_with_hallucinations = 0

    for idx, (summary, source) in enumerate(zip(summaries, sources)):
        try:
            score_result, sum_ents, sum_facts, src_ents, src_facts = evaluate_single_summary(
                summary, source, nlp_model,
                fuzzy_threshold, embedding_threshold, weights,
                embedding_model
            )

            all_scores.append(score_result)

            # Track entity types
            for entity in sum_ents:
                entity_types_count[entity.type] += 1

            for unsupported_ent_text in score_result.unsupported_entities:
                # Find the entity type
                for entity in sum_ents:
                    if entity.normalized_text == unsupported_ent_text:
                        entity_types_unsupported[entity.type] += 1
                        break

            # Track summaries with at least one hallucination
            if score_result.unsupported_entities or score_result.unsupported_facts:
                summaries_with_hallucinations += 1

            results.append({
                'index': idx,
                'entity_precision': score_result.entity_precision,
                'fact_precision': score_result.fact_precision,
                'consistency_score': score_result.consistency_score,
                'num_entities': score_result.total_entities,
                'num_facts': score_result.total_facts,
                'num_unsupported_entities': len(score_result.unsupported_entities),
                'num_unsupported_facts': len(score_result.unsupported_facts),
                'summary': summary,
                'source': source
            })

        except Exception as e:
            logger.error(f"Error evaluating sample {idx}: {e}")
            results.append({
                'index': idx,
                'entity_precision': 0.0,
                'fact_precision': 0.0,
                'consistency_score': 0.0,
                'num_entities': 0,
                'num_facts': 0,
                'num_unsupported_entities': 0,
                'num_unsupported_facts': 0,
                'summary': summary,
                'source': source
            })

    # Compute statistics
    valid_scores = [s for s in all_scores if s.total_entities > 0 or s.total_facts > 0]

    entity_precisions = [s.entity_precision for s in valid_scores if s.total_entities > 0]
    fact_precisions = [s.fact_precision for s in valid_scores if s.total_facts > 0]
    consistency_scores = [s.consistency_score for s in valid_scores]

    # Compute hallucination rates by entity type
    hallucination_rates_by_type = {}
    for ent_type in entity_types_count:
        if entity_types_count[ent_type] > 0:
            rate = entity_types_unsupported[ent_type] / entity_types_count[ent_type]
            hallucination_rates_by_type[ent_type] = rate

    return {
        'results': results,
        'mean_entity_precision': np.mean(entity_precisions) if entity_precisions else 0.0,
        'mean_fact_precision': np.mean(fact_precisions) if fact_precisions else 0.0,
        'mean_consistency_score': np.mean(consistency_scores) if consistency_scores else 0.0,
        'pct_summaries_with_hallucinations': (summaries_with_hallucinations / len(summaries) * 100) if summaries else 0.0,
        'hallucination_rates_by_type': hallucination_rates_by_type,
        'entity_types_count': dict(entity_types_count),
        'total_evaluated': len(results)
    }


def compare_abstractive_extractive(
    abs_summaries: List[str],
    ext_summaries: List[str],
    sources: List[str],
    references: List[str],
    nlp_model,
    config: Dict,
    embedding_model=None
) -> Dict:
    """
    Compare abstractive vs extractive summarization.

    Args:
        abs_summaries: Abstractive summaries
        ext_summaries: Extractive summaries
        sources: Source documents
        references: Reference summaries
        nlp_model: spaCy model
        config: Configuration dict
        embedding_model: Optional sentence transformer model

    Returns:
        Dict with comparison results
    """
    logger.info("Evaluating abstractive summaries...")
    abs_results = evaluate_factual_consistency(
        abs_summaries, sources, nlp_model, config, embedding_model
    )

    logger.info("Evaluating extractive summaries...")
    ext_results = evaluate_factual_consistency(
        ext_summaries, sources, nlp_model, config, embedding_model
    )

    logger.info("Computing ROUGE scores...")
    abs_rouge = compute_rouge_scores(abs_summaries, references)
    ext_rouge = compute_rouge_scores(ext_summaries, references)

    return {
        'abstractive': {
            'factual_consistency': abs_results,
            'rouge': abs_rouge
        },
        'extractive': {
            'factual_consistency': ext_results,
            'rouge': ext_rouge
        }
    }


def compute_ablation_scores(
    summaries: List[str],
    sources: List[str],
    nlp_model,
    config: Dict,
    embedding_model=None
) -> Dict:
    """
    Compute ablation study: entity-only, fact-only, and combined scores.

    Args:
        summaries: List of summaries
        sources: List of source documents
        nlp_model: spaCy model
        config: Configuration dict
        embedding_model: Optional sentence transformer model

    Returns:
        Dict with ablation results
    """
    fuzzy_threshold = config['alignment']['fuzzy_threshold']
    embedding_threshold = config['alignment']['embedding_threshold']

    # Entity-only weights
    entity_only_weights = {'entity_weight': 1.0, 'fact_weight': 0.0}
    # Fact-only weights
    fact_only_weights = {'entity_weight': 0.0, 'fact_weight': 1.0}
    # Combined weights
    combined_weights = {
        'entity_weight': config['scoring']['entity_weight'],
        'fact_weight': config['scoring']['fact_weight']
    }

    entity_only_scores = []
    fact_only_scores = []
    combined_scores = []

    for summary, source in zip(summaries, sources):
        try:
            # Extract
            source_entities = extract_entities(source, nlp_model)
            source_facts = extract_facts(source, nlp_model)
            summary_entities = extract_entities(summary, nlp_model)
            summary_facts = extract_facts(summary, nlp_model)

            # Align
            entity_alignments = align_entities(
                summary_entities, source_entities,
                fuzzy_threshold, embedding_threshold,
                embedding_model
            )
            fact_alignments = align_facts(
                summary_facts, source_facts,
                fuzzy_threshold
            )

            # Score with different weights
            entity_only = compute_scores(entity_alignments, fact_alignments, entity_only_weights)
            fact_only = compute_scores(entity_alignments, fact_alignments, fact_only_weights)
            combined = compute_scores(entity_alignments, fact_alignments, combined_weights)

            entity_only_scores.append(entity_only.consistency_score)
            fact_only_scores.append(fact_only.consistency_score)
            combined_scores.append(combined.consistency_score)

        except Exception as e:
            logger.error(f"Error in ablation: {e}")
            entity_only_scores.append(0.0)
            fact_only_scores.append(0.0)
            combined_scores.append(0.0)

    return {
        'entity_only': {
            'mean_score': np.mean(entity_only_scores),
            'std_score': np.std(entity_only_scores)
        },
        'fact_only': {
            'mean_score': np.mean(fact_only_scores),
            'std_score': np.std(fact_only_scores)
        },
        'combined': {
            'mean_score': np.mean(combined_scores),
            'std_score': np.std(combined_scores)
        }
    }


def compute_human_eval_metrics(
    predicted_scores: List[float],
    human_labels: List[float]
) -> Dict:
    """
    Compute metrics against human factuality annotations.

    Args:
        predicted_scores: Predicted consistency scores
        human_labels: Human factuality labels

    Returns:
        Dict with ROC-AUC, Pearson, Spearman, balanced accuracy
    """
    if not predicted_scores or not human_labels:
        return {}

    # Filter out NaN values
    valid_pairs = [(p, h) for p, h in zip(predicted_scores, human_labels)
                   if not np.isnan(p) and not np.isnan(h)]

    if not valid_pairs:
        return {}

    predicted, human = zip(*valid_pairs)
    predicted = np.array(predicted)
    human = np.array(human)

    metrics = {}

    # Pearson and Spearman correlation
    try:
        pearson_corr, pearson_p = pearsonr(predicted, human)
        metrics['pearson_r'] = pearson_corr
        metrics['pearson_p'] = pearson_p
    except Exception as e:
        logger.warning(f"Could not compute Pearson: {e}")

    try:
        spearman_corr, spearman_p = spearmanr(predicted, human)
        metrics['spearman_r'] = spearman_corr
        metrics['spearman_p'] = spearman_p
    except Exception as e:
        logger.warning(f"Could not compute Spearman: {e}")

    # ROC-AUC (if human labels are binary)
    unique_labels = np.unique(human)
    if len(unique_labels) == 2:
        try:
            roc_auc = roc_auc_score(human, predicted)
            metrics['roc_auc'] = roc_auc

            # Balanced accuracy with threshold at 0.5
            predicted_binary = (predicted >= 0.5).astype(int)
            bal_acc = accuracy_score(human, predicted_binary)
            metrics['balanced_accuracy'] = bal_acc
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    return metrics


def print_error_analysis(
    results: List[Dict],
    human_labels: Optional[List[float]] = None,
    n_examples: int = 10
):
    """
    Print false positive and false negative examples for error analysis.

    Args:
        results: List of evaluation results
        human_labels: Optional human factuality labels
        n_examples: Number of examples to print per category
    """
    if not human_labels:
        logger.info("No human labels provided for error analysis")
        return

    # Identify FP and FN
    # FP: model says high consistency (>0.5) but human says low (<0.5)
    # FN: model says low consistency (<0.5) but human says high (>0.5)

    false_positives = []
    false_negatives = []

    for idx, (result, human_label) in enumerate(zip(results, human_labels)):
        pred_score = result['consistency_score']

        if pred_score >= 0.5 and human_label < 0.5:
            false_positives.append((idx, result, pred_score, human_label))
        elif pred_score < 0.5 and human_label >= 0.5:
            false_negatives.append((idx, result, pred_score, human_label))

    # Sort by absolute difference
    false_positives.sort(key=lambda x: abs(x[2] - x[3]), reverse=True)
    false_negatives.sort(key=lambda x: abs(x[2] - x[3]), reverse=True)

    print("\n" + "="*80)
    print(f"ERROR ANALYSIS: Top {n_examples} False Positives and False Negatives")
    print("="*80)

    if false_positives:
        print(f"\n{'='*80}")
        print(f"FALSE POSITIVES (Model overestimates consistency): {len(false_positives)} total")
        print(f"{'='*80}\n")

        for i, (idx, result, pred, human) in enumerate(false_positives[:n_examples]):
            print(f"Example {i+1} (Index {idx})")
            print(f"  Predicted: {pred:.3f} | Human: {human:.3f} | Diff: {pred-human:.3f}")
            print(f"  Summary: {result['summary'][:200]}...")
            print(f"  Unsupported entities: {result['num_unsupported_entities']}")
            print(f"  Unsupported facts: {result['num_unsupported_facts']}")
            print()

    if false_negatives:
        print(f"{'='*80}")
        print(f"FALSE NEGATIVES (Model underestimates consistency): {len(false_negatives)} total")
        print(f"{'='*80}\n")

        for i, (idx, result, pred, human) in enumerate(false_negatives[:n_examples]):
            print(f"Example {i+1} (Index {idx})")
            print(f"  Predicted: {pred:.3f} | Human: {human:.3f} | Diff: {pred-human:.3f}")
            print(f"  Summary: {result['summary'][:200]}...")
            print(f"  Unsupported entities: {result['num_unsupported_entities']}")
            print(f"  Unsupported facts: {result['num_unsupported_facts']}")
            print()


def save_results(results: Dict, output_dir: str, prefix: str = ""):
    """
    Save evaluation results to CSV files.

    Args:
        results: Dict with evaluation results
        output_dir: Output directory path
        prefix: Optional prefix for filenames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    if 'results' in results:
        df = pd.DataFrame(results['results'])
        filename = f"{prefix}_detailed.csv" if prefix else "detailed.csv"
        df.to_csv(output_path / filename, index=False)
        logger.info(f"Saved detailed results to {output_path / filename}")

    # Save summary statistics
    summary_data = {
        'metric': [],
        'value': []
    }

    for key, value in results.items():
        if key != 'results' and isinstance(value, (int, float)):
            summary_data['metric'].append(key)
            summary_data['value'].append(value)

    if summary_data['metric']:
        df_summary = pd.DataFrame(summary_data)
        filename = f"{prefix}_summary.csv" if prefix else "summary.csv"
        df_summary.to_csv(output_path / filename, index=False)
        logger.info(f"Saved summary to {output_path / filename}")


def generate_evaluation_report(results: Dict) -> str:
    """
    Generate comprehensive evaluation report.

    Args:
        results: Dict with evaluation results

    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 80)
    report.append("EVALUATION REPORT")
    report.append("=" * 80)

    if 'abstractive' in results and 'extractive' in results:
        # Comparison report
        report.append("\nABSTRACTIVE SUMMARIZATION")
        report.append("-" * 80)
        abs_fc = results['abstractive']['factual_consistency']
        report.append(f"Mean Consistency Score: {abs_fc['mean_consistency_score']:.4f}")
        report.append(f"Mean Entity Precision: {abs_fc['mean_entity_precision']:.4f}")
        report.append(f"Mean Fact Precision: {abs_fc['mean_fact_precision']:.4f}")
        report.append(f"% with Hallucinations: {abs_fc['pct_summaries_with_hallucinations']:.2f}%")

        abs_rouge = results['abstractive']['rouge']
        report.append(f"\nROUGE Scores:")
        report.append(f"  ROUGE-1: {abs_rouge['rouge1']:.4f}")
        report.append(f"  ROUGE-2: {abs_rouge['rouge2']:.4f}")
        report.append(f"  ROUGE-L: {abs_rouge['rougeL']:.4f}")

        report.append("\n\nEXTRACTIVE SUMMARIZATION")
        report.append("-" * 80)
        ext_fc = results['extractive']['factual_consistency']
        report.append(f"Mean Consistency Score: {ext_fc['mean_consistency_score']:.4f}")
        report.append(f"Mean Entity Precision: {ext_fc['mean_entity_precision']:.4f}")
        report.append(f"Mean Fact Precision: {ext_fc['mean_fact_precision']:.4f}")
        report.append(f"% with Hallucinations: {ext_fc['pct_summaries_with_hallucinations']:.2f}%")

        ext_rouge = results['extractive']['rouge']
        report.append(f"\nROUGE Scores:")
        report.append(f"  ROUGE-1: {ext_rouge['rouge1']:.4f}")
        report.append(f"  ROUGE-2: {ext_rouge['rouge2']:.4f}")
        report.append(f"  ROUGE-L: {ext_rouge['rougeL']:.4f}")

    else:
        # Single system report
        report.append(f"\nMean Consistency Score: {results.get('mean_consistency_score', 0):.4f}")
        report.append(f"Mean Entity Precision: {results.get('mean_entity_precision', 0):.4f}")
        report.append(f"Mean Fact Precision: {results.get('mean_fact_precision', 0):.4f}")
        report.append(f"% with Hallucinations: {results.get('pct_summaries_with_hallucinations', 0):.2f}%")

    report.append("\n" + "=" * 80)

    return "\n".join(report)

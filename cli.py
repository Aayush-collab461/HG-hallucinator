"""
Command-line interface.

CLI for running evaluation pipelines and generating reports.
"""

import argparse
import logging
import yaml
from pathlib import Path
import spacy
from sentence_transformers import SentenceTransformer

from kg_checker.data import load_dataset, preprocess_text
from kg_checker.summarizers import AbstractiveSummarizer, ExtractiveSummarizer
from kg_checker.evaluate import (
    compare_abstractive_extractive,
    compute_ablation_scores,
    compute_human_eval_metrics,
    print_error_analysis,
    save_results,
    generate_evaluation_report
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Run CLI application."""
    parser = argparse.ArgumentParser(
        description="KG Hallucination Checker - Fact-check abstractive summaries via knowledge graphs"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cnn_dailymail",
        help="Dataset name (cnn_dailymail or xsum)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bart",
        help="Abstractive model to use (bart)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: results/)"
    )
    parser.add_argument(
        "--skip-extractive",
        action="store_true",
        help="Skip extractive summarization"
    )
    parser.add_argument(
        "--skip-ablation",
        action="store_true",
        help="Skip ablation study"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set output directory
    if args.output_dir is None:
        args.output_dir = config['output']['results_dir']

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("KG HALLUCINATION CHECKER - EVALUATION PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Samples: {args.n}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Output: {args.output_dir}")
    logger.info("=" * 80)

    # Load models
    logger.info("Loading spaCy model...")
    nlp = spacy.load(config['models']['spacy_model'])

    logger.info("Loading embedding model...")
    embedding_model = SentenceTransformer(config['models']['embedding_model'])

    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    dataset = load_dataset(args.dataset, split='test', n_samples=args.n, seed=args.seed)

    sources = [preprocess_text(item['document']) for item in dataset]
    references = [preprocess_text(item['summary']) for item in dataset]

    logger.info(f"Loaded {len(sources)} examples")

    # Initialize summarizers
    logger.info("Initializing abstractive summarizer...")
    abs_summarizer = AbstractiveSummarizer(
        model_name=config['models']['abstractive_model'],
        cache_dir=config['dataset']['cache_dir']
    )

    # Generate abstractive summaries
    logger.info("Generating abstractive summaries...")
    abs_summaries = []
    for i, source in enumerate(sources):
        if i % 10 == 0:
            logger.info(f"  Progress: {i}/{len(sources)}")
        summary = abs_summarizer.summarize(source)
        abs_summaries.append(summary)

    logger.info(f"Generated {len(abs_summaries)} abstractive summaries")

    # Generate extractive summaries
    ext_summaries = []
    if not args.skip_extractive:
        logger.info("Initializing extractive summarizer...")
        ext_summarizer = ExtractiveSummarizer(method='textrank')

        logger.info("Generating extractive summaries...")
        for i, source in enumerate(sources):
            if i % 10 == 0:
                logger.info(f"  Progress: {i}/{len(sources)}")
            summary = ext_summarizer.summarize(source)
            ext_summaries.append(summary)

        logger.info(f"Generated {len(ext_summaries)} extractive summaries")

    # Run comparison evaluation
    if ext_summaries:
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING COMPARISON EVALUATION")
        logger.info("=" * 80)

        comparison_results = compare_abstractive_extractive(
            abs_summaries,
            ext_summaries,
            sources,
            references,
            nlp,
            config,
            embedding_model
        )

        # Print and save comparison results
        report = generate_evaluation_report(comparison_results)
        print("\n" + report)

        # Save results
        save_results(
            comparison_results['abstractive']['factual_consistency'],
            args.output_dir,
            prefix="abstractive"
        )
        save_results(
            comparison_results['extractive']['factual_consistency'],
            args.output_dir,
            prefix="extractive"
        )

        # Save comparison summary
        import pandas as pd
        comparison_df = pd.DataFrame([
            {
                'system': 'abstractive',
                'consistency_score': comparison_results['abstractive']['factual_consistency']['mean_consistency_score'],
                'entity_precision': comparison_results['abstractive']['factual_consistency']['mean_entity_precision'],
                'fact_precision': comparison_results['abstractive']['factual_consistency']['mean_fact_precision'],
                'pct_with_hallucinations': comparison_results['abstractive']['factual_consistency']['pct_summaries_with_hallucinations'],
                'rouge1': comparison_results['abstractive']['rouge']['rouge1'],
                'rouge2': comparison_results['abstractive']['rouge']['rouge2'],
                'rougeL': comparison_results['abstractive']['rouge']['rougeL']
            },
            {
                'system': 'extractive',
                'consistency_score': comparison_results['extractive']['factual_consistency']['mean_consistency_score'],
                'entity_precision': comparison_results['extractive']['factual_consistency']['mean_entity_precision'],
                'fact_precision': comparison_results['extractive']['factual_consistency']['mean_fact_precision'],
                'pct_with_hallucinations': comparison_results['extractive']['factual_consistency']['pct_summaries_with_hallucinations'],
                'rouge1': comparison_results['extractive']['rouge']['rouge1'],
                'rouge2': comparison_results['extractive']['rouge']['rouge2'],
                'rougeL': comparison_results['extractive']['rouge']['rougeL']
            }
        ])
        comparison_df.to_csv(output_dir / "comparison_summary.csv", index=False)
        logger.info(f"Saved comparison summary to {output_dir / 'comparison_summary.csv'}")

        # Print hallucination rates by entity type
        print("\n" + "=" * 80)
        print("HALLUCINATION RATES BY ENTITY TYPE")
        print("=" * 80)

        abs_rates = comparison_results['abstractive']['factual_consistency']['hallucination_rates_by_type']
        ext_rates = comparison_results['extractive']['factual_consistency']['hallucination_rates_by_type']

        all_types = sorted(set(list(abs_rates.keys()) + list(ext_rates.keys())))

        print(f"\n{'Entity Type':<15} {'Abstractive':<15} {'Extractive':<15}")
        print("-" * 45)
        for ent_type in all_types:
            abs_rate = abs_rates.get(ent_type, 0.0) * 100
            ext_rate = ext_rates.get(ent_type, 0.0) * 100
            print(f"{ent_type:<15} {abs_rate:>6.2f}%{'':<8} {ext_rate:>6.2f}%")

    # Run ablation study
    if not args.skip_ablation:
        logger.info("\n" + "=" * 80)
        logger.info("RUNNING ABLATION STUDY")
        logger.info("=" * 80)

        ablation_results = compute_ablation_scores(
            abs_summaries,
            sources,
            nlp,
            config,
            embedding_model
        )

        print("\n" + "=" * 80)
        print("ABLATION STUDY: Entity-only vs Fact-only vs Combined")
        print("=" * 80)
        print(f"\nEntity-only:  {ablation_results['entity_only']['mean_score']:.4f} (±{ablation_results['entity_only']['std_score']:.4f})")
        print(f"Fact-only:    {ablation_results['fact_only']['mean_score']:.4f} (±{ablation_results['fact_only']['std_score']:.4f})")
        print(f"Combined:     {ablation_results['combined']['mean_score']:.4f} (±{ablation_results['combined']['std_score']:.4f})")

        # Save ablation results
        import pandas as pd
        ablation_df = pd.DataFrame([
            {
                'configuration': 'entity_only',
                'mean_score': ablation_results['entity_only']['mean_score'],
                'std_score': ablation_results['entity_only']['std_score']
            },
            {
                'configuration': 'fact_only',
                'mean_score': ablation_results['fact_only']['mean_score'],
                'std_score': ablation_results['fact_only']['std_score']
            },
            {
                'configuration': 'combined',
                'mean_score': ablation_results['combined']['mean_score'],
                'std_score': ablation_results['combined']['std_score']
            }
        ])
        ablation_df.to_csv(output_dir / "ablation_study.csv", index=False)
        logger.info(f"Saved ablation results to {output_dir / 'ablation_study.csv'}")

    # Check for human labels
    human_labels = None
    try:
        # Try to load human factuality labels if available
        human_labels = [item.get('factuality_label', None) for item in dataset]
        if all(label is None for label in human_labels):
            human_labels = None
    except Exception:
        pass

    # Compute human evaluation metrics if labels exist
    if human_labels:
        logger.info("\n" + "=" * 80)
        logger.info("COMPUTING HUMAN EVALUATION METRICS")
        logger.info("=" * 80)

        abs_fc_results = comparison_results['abstractive']['factual_consistency']['results'] if ext_summaries else []
        predicted_scores = [r['consistency_score'] for r in abs_fc_results]

        human_metrics = compute_human_eval_metrics(predicted_scores, human_labels)

        if human_metrics:
            print("\nHuman Evaluation Metrics:")
            print("-" * 40)
            if 'pearson_r' in human_metrics:
                print(f"Pearson r:          {human_metrics['pearson_r']:.4f} (p={human_metrics['pearson_p']:.4e})")
            if 'spearman_r' in human_metrics:
                print(f"Spearman r:         {human_metrics['spearman_r']:.4f} (p={human_metrics['spearman_p']:.4e})")
            if 'roc_auc' in human_metrics:
                print(f"ROC-AUC:            {human_metrics['roc_auc']:.4f}")
            if 'balanced_accuracy' in human_metrics:
                print(f"Balanced Accuracy:  {human_metrics['balanced_accuracy']:.4f}")

            # Error analysis
            print_error_analysis(abs_fc_results, human_labels, n_examples=10)
    else:
        logger.info("\nNo human factuality labels available for this dataset")

    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

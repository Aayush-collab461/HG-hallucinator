"""
Data loading and preprocessing module.

Handles dataset loading, text preprocessing, and data preparation for summarization
and evaluation pipelines.
"""

import logging
from typing import Optional
from pathlib import Path

import datasets
from datasets import load_dataset as hf_load_dataset

logger = logging.getLogger(__name__)

# Cache directory for datasets
CACHE_DIR = Path("./cache")


def load_dataset(name: str, split: str = "train", n_samples: Optional[int] = None):
    """
    Load dataset from HuggingFace datasets.

    Args:
        name: Dataset name (e.g., "cnn_dailymail", "xsum")
        split: Dataset split to load (e.g., "train", "validation", "test")
        n_samples: Number of samples to load (None for all)

    Returns:
        Dataset dict with 'article' and 'highlights'/'summary' fields
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Use correct namespace/name format for HuggingFace datasets
    dataset_map = {
        "cnn_dailymail": "abisee/cnn_dailymail",
        "cnn": "abisee/cnn_dailymail",
        "xsum": "EdinburghNLP/xsum",
    }

    if name.lower() not in dataset_map:
        raise ValueError(f"Unknown dataset: {name}. Supported: {list(dataset_map.keys())}")

    dataset_name = dataset_map[name.lower()]
    logger.info(f"Loading {dataset_name}, split={split}")

    dataset = hf_load_dataset(
        dataset_name,
        "3.0.0" if "cnn" in name.lower() else None,
        split=split,
        cache_dir=str(CACHE_DIR)
    )

    # Select n_samples if specified
    if n_samples is not None:
        dataset = dataset.select(range(min(n_samples, len(dataset))))
        logger.info(f"Selected {n_samples} samples")

    return dataset


def load_factuality_annotations(name: str = "xsum"):
    """
    Load human factuality annotations (Maynez et al. XSum / FRANK).

    These datasets contain human annotations for factual consistency.

    Args:
        name: Annotation dataset name ("xsum" or "frank")

    Returns:
        Dict with annotations or None if unavailable (logs warning)
    """
    try:
        if name.lower() == "xsum":
            # Maynez et al. XSum factuality annotations
            # Try to load from HuggingFace or local file
            logger.info("Attempting to load XSum factuality annotations")
            # Placeholder: actual implementation depends on data availability
            logger.warning("XSum factuality annotations not available")
            return None

        elif name.lower() == "frank":
            # FRANK dataset (Pagnoni et al.)
            logger.info("Attempting to load FRANK annotations")
            logger.warning("FRANK annotations not available")
            return None

        else:
            logger.warning(f"Unknown factuality annotation dataset: {name}")
            return None

    except Exception as e:
        logger.warning(f"Failed to load factuality annotations: {e}")
        return None


def preprocess_text(text: str) -> str:
    """
    Preprocess and clean input text.

    Args:
        text: Raw input text

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def prepare_batch(texts: list, tokenizer=None, max_length: int = 1024):
    """
    Prepare a batch of texts for model processing.

    Args:
        texts: List of input texts
        tokenizer: Optional tokenizer for encoding
        max_length: Maximum sequence length

    Returns:
        Dict with tensors ready for model input
    """
    if tokenizer is not None:
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        )
    else:
        return {"texts": texts}


def get_column_names(dataset_name: str) -> tuple:
    """
    Get dataset column names for source and summary.

    Args:
        dataset_name: Name of the dataset

    Returns:
        Tuple of (source_column, summary_column)
    """
    column_map = {
        "cnn_dailymail": ("article", "highlights"),
        "cnn": ("article", "highlights"),
        "xsum": ("document", "summary"),
    }

    if dataset_name.lower() not in column_map:
        raise ValueError(f"No column mapping for dataset: {dataset_name}")

    return column_map[dataset_name.lower()]


def extract_samples(dataset, source_col: str, summary_col: str, n: int = 5):
    """
    Extract n samples from dataset for testing.

    Args:
        dataset: HuggingFace dataset
        source_col: Source text column name
        summary_col: Summary text column name
        n: Number of samples to extract

    Returns:
        List of dicts with 'source' and 'summary' keys
    """
    samples = []
    for i in range(min(n, len(dataset))):
        sample = dataset[i]
        samples.append({
            "source": preprocess_text(sample[source_col]),
            "summary": preprocess_text(sample[summary_col]),
            "id": i
        })
    return samples
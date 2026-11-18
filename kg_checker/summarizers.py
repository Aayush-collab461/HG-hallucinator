"""
Summarization module.

Implements abstractive and extractive summarization models for generating
and comparing summaries.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Cache directory
CACHE_DIR = Path("./cache/summaries")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class SummaryResult:
    """Container for summarization result."""
    summary: str
    method: str
    source_len: int
    summary_len: int
    compression_ratio: float


class AbstractiveSummarizer:
    """Abstractive summarizer using transformer models."""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: str = "auto",
        cache: bool = True
    ):
        """
        Initialize abstractive summarizer.

        Args:
            model_name: HuggingFace model name
            device: Device to use ("cuda", "cpu", "auto")
            cache: Whether to cache summaries to disk
        """
        self.model_name = model_name
        self.cache = cache
        self.cache_path = CACHE_DIR / f"abstractive_{model_name.replace('/', '_')}.json"
        self.cache_data: Dict[str, str] = {}

        self._load_model()
        self._load_cache()

    def _load_model(self):
        """Load the transformer model and tokenizer."""
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            logger.info(f"Loading abstractive model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

            # Set device
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"

            self.model.to(self.device)
            logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise

    def _load_cache(self):
        """Load cached summaries from disk."""
        if self.cache and self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    self.cache_data = json.load(f)
                logger.info(f"Loaded {len(self.cache_data)} cached summaries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache_data = {}

    def _save_cache(self):
        """Save cached summaries to disk."""
        if self.cache:
            try:
                with open(self.cache_path, 'w') as f:
                    json.dump(self.cache_data, f, indent=2)
                logger.info(f"Saved {len(self.cache_data)} summaries to cache")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30) -> SummaryResult:
        """
        Generate abstractive summary.

        Args:
            text: Source text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length

        Returns:
            SummaryResult with generated summary
        """
        # Check cache
        cache_key = text[:500]  # Use first 500 chars as key
        if cache_key in self.cache_data:
            logger.debug("Returning cached summary")
            summary = self.cache_data[cache_key]
            return SummaryResult(
                summary=summary,
                method="abstractive",
                source_len=len(text),
                summary_len=len(summary),
                compression_ratio=len(summary) / max(1, len(text))
            )

        # Generate summary
        import torch

        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        # Adjust max/min lengths dynamically for shorter inputs to avoid verbatim copying
        num_tokens = len(text.split())
        gen_max = min(max_length, max(25, int(num_tokens * 0.75)))
        gen_min = min(min_length, max(8, int(num_tokens * 0.25)))
        if gen_max <= gen_min:
            gen_max = gen_min + 15

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=gen_max,
                min_new_tokens=gen_min,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )

        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Cache the result
        self.cache_data[cache_key] = summary
        self._save_cache()

        return SummaryResult(
            summary=summary,
            method="abstractive",
            source_len=len(text),
            summary_len=len(summary),
            compression_ratio=len(summary) / max(1, len(text))
        )

    def summarize_batch(self, texts: List[str], batch_size: int = 4) -> List[SummaryResult]:
        """
        Generate summaries for a batch of texts.

        Args:
            texts: List of source texts
            batch_size: Batch size for processing

        Returns:
            List of SummaryResult objects
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                results.append(self.summarize(text))
        return results


class ExtractiveSummarizer:
    """Extractive summarizer using various strategies."""

    def __init__(self, method: str = "lead3", cache: bool = True):
        """
        Initialize extractive summarizer.

        Args:
            method: Summarization method ("lead3" or "textrank")
            cache: Whether to cache summaries to disk
        """
        self.method = method
        self.cache = cache
        self.cache_path = CACHE_DIR / f"extractive_{method}.json"
        self.cache_data: Dict[str, str] = {}

        if method == "textrank":
            self._init_textrank()

        self._load_cache()

    def _init_textrank(self):
        """Initialize TextRank components."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            self.TfidfVectorizer = TfidfVectorizer
            self.cosine_similarity = cosine_similarity
        except ImportError as e:
            logger.error("sklearn required for TextRank")
            raise

    def _load_cache(self):
        """Load cached summaries from disk."""
        if self.cache and self.cache_path.exists():
            try:
                with open(self.cache_path, 'r') as f:
                    self.cache_data = json.load(f)
                logger.info(f"Loaded {len(self.cache_data)} cached summaries")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache_data = {}

    def _save_cache(self):
        """Save cached summaries to disk."""
        if self.cache:
            try:
                with open(self.cache_path, 'w') as f:
                    json.dump(self.cache_data, f, indent=2)
                logger.info(f"Saved {len(self.cache_data)} summaries to cache")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

    def summarize(self, text: str, num_sentences: int = 3) -> SummaryResult:
        """
        Generate extractive summary.

        Args:
            text: Source text to summarize
            num_sentences: Number of sentences to include

        Returns:
            SummaryResult with selected summary
        """
        # Check cache
        cache_key = text[:500]
        if cache_key in self.cache_data:
            summary = self.cache_data[cache_key]
            return SummaryResult(
                summary=summary,
                method=f"extractive_{self.method}",
                source_len=len(text),
                summary_len=len(summary),
                compression_ratio=len(summary) / max(1, len(text))
            )

        # Split into sentences - handle NLTK data download issues
        import re
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt_tab')
                from nltk.tokenize import sent_tokenize
            except LookupError:
                # Try alternative punkt resource
                try:
                    nltk.data.find('tokenizers/punkt')
                    from nltk.tokenize import sent_tokenize
                except LookupError:
                    # Fallback: use regex-based sentence splitting
                    sent_tokenize = None
        except Exception:
            sent_tokenize = None

        # Fallback regex for sentence splitting if NLTK fails
        if sent_tokenize is None:
            def sent_tokenize(text):
                # Simple regex-based sentence splitting
                return re.split(r'(?<=[.!?])\s+', text)
        sentences = sent_tokenize(text)

        if len(sentences) <= num_sentences:
            summary = text
        elif self.method == "lead3":
            summary = " ".join(sentences[:num_sentences])
        elif self.method == "textrank":
            summary = self._textrank_summarize(sentences, num_sentences)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Cache result
        self.cache_data[cache_key] = summary
        self._save_cache()

        return SummaryResult(
            summary=summary,
            method=f"extractive_{self.method}",
            source_len=len(text),
            summary_len=len(summary),
            compression_ratio=len(summary) / max(1, len(text))
        )

    def _textrank_summarize(self, sentences: List[str], num_sentences: int) -> str:
        """
        Generate summary using TextRank algorithm.

        Args:
            sentences: List of sentences
            num_sentences: Number of sentences to select

        Returns:
            Selected sentences joined as summary
        """
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        # Build sentence similarity matrix using TF-IDF
        vectorizer = self.TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(sentences)

        # Compute similarity matrix
        sim_matrix = self.cosine_similarity(tfidf_matrix)

        # Apply PageRank
        import numpy as np
        n = len(sentences)
        scores = np.ones(n) / n

        # Power iteration for PageRank
        damping = 0.85
        for _ in range(50):
            new_scores = (1 - damping) / n + damping * sim_matrix.T.dot(scores)
            scores = new_scores / new_scores.sum()

        # Select top sentences (in order)
        top_indices = np.argsort(scores)[::-1][:num_sentences]
        top_indices_sorted = sorted(top_indices)

        return " ".join([sentences[i] for i in top_indices_sorted])

    def summarize_batch(self, texts: List[str], num_sentences: int = 3) -> List[SummaryResult]:
        """
        Generate summaries for a batch of texts.

        Args:
            texts: List of source texts
            num_sentences: Number of sentences per summary

        Returns:
            List of SummaryResult objects
        """
        return [self.summarize(text, num_sentences) for text in texts]


def create_summarizers(config: dict) -> Dict[str, tuple]:
    """
    Create all summarizers from configuration.

    Args:
        config: Configuration dict with model settings

    Returns:
        Dict of summarizer name to (summarizer, method) tuples
    """
    models = config.get("models", {})
    cache = config.get("dataset", {}).get("cache_dir", "./cache") is not None

    summarizers = {
        "abstractive": (
            AbstractiveSummarizer(
                model_name=models.get("abstractive_model", "facebook/bart-large-cnn"),
                cache=cache
            ),
            "abstractive"
        ),
        "lead3": (
            ExtractiveSummarizer(method="lead3", cache=cache),
            "extractive_lead3"
        ),
        "textrank": (
            ExtractiveSummarizer(method="textrank", cache=cache),
            "extractive_textrank"
        ),
    }

    return summarizers
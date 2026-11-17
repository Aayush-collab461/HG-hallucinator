# Development Progress

## Project Goal

Fact-check abstractive summaries via knowledge graphs: align generated summary entities/facts against source knowledge graphs to quantify factual consistency, detect unsupported entities/facts, and compare with an extractive baseline.

## Design Summary

### Module Overview

- **data.py**: Load datasets (CNN/DailyMail, XSum) and preprocess text
- **summarizers.py**: Generate abstractive (BART) and extractive summaries
- **extract.py**: Extract entities (NER) and facts (SVO triples) using spaCy
- **kg.py**: Build knowledge graphs from entities/facts using NetworkX
- **align.py**: Align summary entities/facts to source KG via fuzzy matching + embeddings
- **score.py**: Compute factual consistency scores and identify hallucinations
- **evaluate.py**: Calculate ROUGE scores and aggregate evaluation metrics
- **visualize.py**: Generate KG visualizations and result plots

### Architecture

1. Load source documents → Generate abstractive + extractive summaries
2. Extract entities/facts from sources and summaries → Build KGs
3. Align summary KG elements to source KG (fuzzy + embedding similarity)
4. Score consistency: weighted combination of entity/fact alignment
5. Compare abstractive vs extractive: ROUGE + factual consistency

## Implementation Checklist

### Part 1: Setup ✓
- [x] Project structure created
- [x] Dependencies defined (requirements.txt)
- [x] Configuration file (config.yaml)
- [x] Stub files with docstrings
- [x] Dependencies installed
- [x] spaCy model downloaded
- [x] Import verification

### Part 2: Data Loading & Preprocessing ✓
- [x] Implement `load_dataset()` for HuggingFace datasets
- [x] Implement `preprocess_text()` for text cleaning
- [x] Implement `prepare_batch()` for batch processing
- [x] Add dataset caching
- [x] Add factuality annotation loader (graceful fallback)
- [x] Test data loading with sample datasets

### Part 3: Summarization ✓
- [x] Implement `AbstractiveSummarizer` with BART-large-CNN
- [x] Implement `ExtractiveSummarizer` with Lead-3 and TextRank
- [x] Add JSON-based output caching for fast reruns
- [x] Test summarizers on sample texts

### Part 4: Entity & Fact Extraction ✓
- [x] Implement `extract_entities()` with spaCy NER
- [x] Implement `extract_facts()` for SVO triple extraction
- [x] Implement `normalize_text()` for numbers and dates
- [x] Implement pronoun filtering
- [x] Test extraction with 5 hand-crafted sentences (all 24 tests pass)

### Part 4b: Knowledge Graph Construction & Alignment ✓
- [x] Implement `KnowledgeGraph` class with NetworkX
- [x] Implement `add_entity()` and `add_fact()` methods
- [x] Implement `get_subgraph()`, `get_neighbors()`, `get_entities()`, `get_facts()`
- [x] Implement `build_kg_from_text()` pipeline
- [x] Implement `build_kg_from_entities_and_facts()` helper

### Part 5: Alignment Module ✓
- [x] Implement `align_entity()` - exact → fuzzy → embedding cascade
- [x] Implement `align_entities()` - batch entity alignment
- [x] Implement `align_fact()` - fact matching with fuzzy components
- [x] Implement `align_facts()` - batch fact alignment
- [x] Implement `get_aligned_count()` and `get_unaligned_items()` helpers
- [x] Test with 5 hand-crafted scenarios (40 tests pass)
- [x] **Fully faithful summary**: Entity/fact precision near 1.0
- [x] **Wrong entity name**: Flagged as unsupported entity
- [x] **Wrong number**: Flagged as unsupported entity
- [x] **Paraphrased summary**: Matches via fuzzy/embedding
- [x] **Empty summary**: Returns NaN scores appropriately

### Part 5b: Scoring Module ✓
- [x] Implement `compute_entity_score()` - entity precision (0-1 or NaN)
- [x] Implement `compute_fact_score()` - fact precision (0-1 or NaN)
- [x] Implement `compute_consistency_score()` - weighted combination from config.yaml
- [x] Implement `identify_hallucinations()` - returns unsupported entities/facts
- [x] Implement `compute_scores()` - complete ScoreResult dataclass
- [x] **Entity precision**: Supported entities / total entities
- [x] **Fact precision**: Supported facts / total facts
- [x] **Consistency score**: Weighted (entity_weight=0.4, fact_weight=0.6)
- [x] **Hallucination detection**: Unsupported entities & facts
- [x] Handle edge cases: empty lists, NaN scores, normalization

### Part 6: Evaluation Module ✓
- [x] Implement `compute_rouge_scores()` — ROUGE-1/2/L scores
- [x] Implement `evaluate_factual_consistency()` — Full pipeline per system
- [x] Implement `compare_abstractive_extractive()` — Side-by-side comparison
- [x] Implement `compute_ablation_scores()` — Entity-only, fact-only, combined
- [x] Implement `compute_human_eval_metrics()` — ROC-AUC, Pearson, Spearman, balanced accuracy
- [x] Implement `print_error_analysis()` — False positive/negative examples
- [x] Implement `save_results()` — CSV outputs
- [x] Implement `generate_evaluation_report()` — Formatted reports
- [x] **Hallucination statistics**: % summaries with hallucinations, rates by entity type (PERSON, ORG, DATE, NUMBER, GPE)
- [x] **Test suite**: 11 unit tests (ROUGE, ablation, reporting) all passing

### Part 7: CLI Interface ✓
- [x] Implement `cli.py` with argparse
- [x] Support `--dataset xsum|cnn_dailymail`
- [x] Support `--n N_SAMPLES` (tested with n=20, ready for n=200)
- [x] Support `--model bart`
- [x] Support `--config config.yaml`
- [x] Support `--output-dir`, `--skip-extractive`, `--skip-ablation`
- [x] Load config, models, dataset
- [x] Run full pipeline: abstractive + extractive + comparison
- [x] Print results table (system, consistency, precision, hallucination %, ROUGE)
- [x] Save CSV outputs (comparison_summary.csv, ablation_study.csv, detailed.csv)
- [x] Verification: all 5 tests passed

### Part 8: Visualization Module ✓
- [x] Implement `plot_system_comparison()` — Bar chart abstractive vs extractive
- [x] Implement `plot_consistency_histogram()` — Distribution of scores
- [x] Implement `plot_hallucination_rates_by_type()` — Hallucination rates by entity type
- [x] Implement `visualize_kg_comparison()` — Source vs summary KG with unsupported in red
- [x] Implement `plot_ablation_study()` — Entity/fact/combined scores
- [x] Implement `generate_all_visualizations()` — Generate all charts from CSVs
- [x] **Output files**: 5 PNG charts saved to results/figures/
  - 01_system_comparison.png (70.7 KB) — Consistency & precision metrics, ROUGE scores
  - 02_consistency_histogram.png (47.7 KB) — Score distribution with mean/median
  - 03_hallucination_rates.png (49.3 KB) — Rates by entity type (PERSON, ORG, GPE, DATE, MONEY, PRODUCT)
  - 04_kg_comparison_example_0.png (123.8 KB) — Source KG vs summary KG (unsupported in red)
  - 05_ablation_study.png (48.1 KB) — Entity-only, fact-only, combined with error bars

### Part 9: Streamlit Web App ✓
- [x] Implement `app.py` with Streamlit interface
- [x] **Input modes**:
  - Paste article + summary
  - Auto-generate summary with BART
- [x] **Output display**:
  - Three scores (entity precision, fact precision, consistency)
  - Unsupported entities highlighted in red
  - Source KG and summary KG visualized
  - Breakdown tables for entities/facts
- [x] **Interactive controls**:
  - Adjustable fuzzy/embedding thresholds
  - Configurable entity/fact weights
  - Model caching for performance
- [x] **Verification**: All imports successful, extraction working

### Part 10: Documentation ✓
- [x] Write comprehensive README.md
- [x] **Sections**:
  - Quick start (installation, usage)
  - Feature overview (extraction, alignment, scoring, evaluation)
  - Architecture diagram with pipeline flow
  - Configuration reference
  - Results examples with CSV/figures
  - Python API usage examples
  - Streamlit app guide
  - Design decisions documented
  - **Limitations clearly stated**:
    - Parser errors on out-of-domain text
    - Paraphrase mismatch with fuzzy threshold
    - Missing coreference resolution
    - Known gaps (numbers, negation, temporal reasoning)
  - **Test coverage**: 104/104 tests ✅
  - **Performance table**: n=20, n=200, n=1000+ timing
  - Future work roadmap
  - Citation format
  - License and contribution guidelines

### Part 11: Project Complete ✓
- [x] All 8 modules implemented and tested
- [x] 104+ unit tests passing
- [x] Full evaluation pipeline working end-to-end
- [x] CLI interface functional with multiple datasets
- [x] 5 professional visualizations generated and verified
- [x] Streamlit web app ready to launch
- [x] Comprehensive README with all documentation
- [x] All requirements met and verified

## Notes

- Use CNN/DailyMail or XSum as primary datasets
- BART-large-CNN recommended for abstractive summarization
- Consider both fuzzy matching and embedding similarity for robustness
- Weighted scoring allows tuning entity vs fact importance
- Visualization critical for understanding alignment quality

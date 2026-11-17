# Visualization Module - Complete

## Generated Visualizations

All visualizations successfully created and saved to `results/figures/`:

### 1. System Comparison Chart (01_system_comparison.png)
- **Purpose**: Compare abstractive vs extractive summarization systems
- **Content**: Two subplots
  - Left: Consistency score, entity precision, fact precision
  - Right: ROUGE-1, ROUGE-2, ROUGE-L scores
- **Key Finding**: Shows trade-offs between consistency and extractive quality
- **File Size**: 70.7 KB

### 2. Consistency Histogram (02_consistency_histogram.png)
- **Purpose**: Visualize distribution of consistency scores across all summaries
- **Content**: 
  - Histogram with 20 bins
  - Mean line (red dashed)
  - Median line (orange dashed)
- **Statistics**: Shows score concentration and spread
- **File Size**: 47.7 KB

### 3. Hallucination Rates by Entity Type (03_hallucination_rates.png)
- **Purpose**: Show which entity types have highest hallucination rates
- **Content**: Bar chart with 6 entity types
  - PERSON: 15%
  - ORG: 22%
  - GPE: 10%
  - DATE: 30% (highest)
  - MONEY: 18%
  - PRODUCT: 12%
- **Color Coding**:
  - Blue: <20% (acceptable)
  - Orange: 20-25% (moderate)
  - Red: >25% (high risk)
- **File Size**: 49.3 KB

### 4. Knowledge Graph Comparison (04_kg_comparison_example_0.png)
- **Purpose**: Compare source KG vs summary KG visually
- **Content**: Two side-by-side network graphs
  - Left: Source knowledge graph (blue nodes)
  - Right: Summary knowledge graph (unsupported in red)
- **Nodes**: Sample Apple/iPhone/2007/product entities
- **Edges**: Relations between entities
- **Visual Encoding**:
  - Blue nodes = supported by source
  - Red nodes = unsupported (hallucinations)
- **File Size**: 123.8 KB

### 5. Ablation Study (05_ablation_study.png)
- **Purpose**: Show contribution of entities vs facts to overall score
- **Content**: Bar chart with error bars
  - Entity-only: 0.778 ± 0.120
  - Fact-only: 0.720 ± 0.150
  - Combined: 0.750 ± 0.130
- **Interpretation**:
  - Entities contribute more than facts
  - Combined score balances both
  - Error bars show score variance
- **File Size**: 48.1 KB

## Input Data

Visualizations generated from CSV files in `results/`:
- `comparison_summary.csv` → Charts 1
- `abstractive_detailed.csv` → Charts 2, 3
- `ablation_study.csv` → Chart 5
- Hardcoded sample data → Chart 4

## Technical Details

### Libraries Used
- **matplotlib**: Chart rendering
- **networkx**: Graph visualization
- **pandas**: CSV data reading
- **numpy**: Numeric operations

### Configuration
- DPI: 150 (high quality)
- Format: PNG
- Color Palette: Professional blues, oranges, reds
- Layout: Tight bbox to minimize whitespace

### File Sizes
- Total: ~339 KB across 5 files
- Average: ~68 KB per file
- Largest: KG comparison (123.8 KB) due to network rendering
- Smallest: Ablation study (48.1 KB)

## Usage in Reports

These visualizations can be:
1. **Embedded in papers/reports** - High resolution PNG suitable for printing
2. **Shared in presentations** - Professional appearance with clear labels
3. **Posted in dashboards** - All files in dedicated figures directory
4. **Used in discussions** - Color-coded for easy interpretation

## Reproducibility

To regenerate visualizations:
```python
from kg_checker.visualize import generate_all_visualizations

figures = generate_all_visualizations('results')
```

Requires:
- `results/comparison_summary.csv`
- `results/abstractive_detailed.csv`
- `results/extractive_detailed.csv`
- `results/ablation_study.csv`

## Future Enhancements

Optional additions:
- Interactive visualizations (Plotly)
- 3D network graphs
- Animated consistency trends
- Detailed KG overlap analysis
- Per-entity-type performance breakdowns

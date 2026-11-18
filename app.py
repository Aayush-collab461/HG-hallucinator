"""
Streamlit app for KG Hallucination Checker.

Interactive interface to check factual consistency of summaries using knowledge graphs.
"""

import os
import re
import logging
import yaml
from pathlib import Path
from typing import List, Set, Tuple

import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from kg_checker.extract import extract_entities, extract_facts, Entity, Fact
from kg_checker.kg import build_kg_from_text
from kg_checker.align import align_entities, align_facts
from kg_checker.score import compute_scores, ScoreResult
from kg_checker.summarizers import AbstractiveSummarizer, ExtractiveSummarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure cached models are used without network delays
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ─── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KG Hallucination Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom Styles ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Clean container layout */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    /* Header styling */
    .app-header {
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0;
    }

    /* Score summary cards */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-card.score-card-high {
        border-top: 4px solid #10b981;
    }
    .metric-card.score-card-medium {
        border-top: 4px solid #f59e0b;
    }
    .metric-card.score-card-low {
        border-top: 4px solid #ef4444;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.4rem;
    }
    .val-green { color: #10b981; }
    .val-amber { color: #f59e0b; }
    .val-red { color: #ef4444; }

    /* Entity chips */
    .chip {
        display: inline-flex;
        align-items: center;
        padding: 0.3rem 0.75rem;
        margin: 0.25rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .chip-supported {
        background-color: #064e3b;
        color: #6ee7b7;
        border: 1px solid #047857;
    }
    .chip-unsupported {
        background-color: #450a0a;
        color: #fca5a5;
        border: 1px solid #b91c1c;
    }
    .chip-type {
        font-size: 0.7rem;
        opacity: 0.85;
        margin-left: 0.35rem;
        padding: 0.1rem 0.3rem;
        background: rgba(0, 0, 0, 0.25);
        border-radius: 4px;
    }

    /* Fact cards */
    .fact-card {
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 8px;
        font-size: 0.92rem;
        border-left: 4px solid;
    }
    .fact-card-supported {
        background: #0f2e22;
        border-left-color: #10b981;
        color: #d1fae5;
    }
    .fact-card-unsupported {
        background: #2e1010;
        border-left-color: #ef4444;
        color: #fee2e2;
    }

    /* Highlighted text mark */
    mark.hallucinated {
        background-color: rgba(239, 68, 68, 0.25);
        color: #fca5a5;
        border-bottom: 2px solid #ef4444;
        padding: 0.1rem 0.3rem;
        border-radius: 3px;
        font-weight: 600;
    }

    /* Section card */
    .section-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1.5rem;
    }

    /* Preset buttons */
    .preset-btn {
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Demo Presets ───────────────────────────────────────────────────────────────
PRESET_ARTICLE = (
    "Barack Obama was born in Hawaii in 1961. "
    "He served as the 44th President of the United States from 2009 to 2017. "
    "Obama graduated from Harvard Law School and was a senator from Illinois before becoming president."
)

PRESET_HALLUCINATED = (
    "Barack Obama was born in Texas in 1965. "
    "He served as the 45th President of the United States."
)

PRESET_FAITHFUL = (
    "Barack Obama was born in Hawaii in 1961. "
    "He was the 44th President of the United States."
)


# ─── Cached Model Loading ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner=False)
def load_nlp():
    return spacy.load('en_core_web_sm')


@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource(show_spinner=False)
def load_bart_summarizer():
    config = load_config()
    return AbstractiveSummarizer(
        model_name=config['models']['abstractive_model'],
        cache=True
    )


# ─── Helper Functions ───────────────────────────────────────────────────────────
def highlight_unsupported_entities(text: str, unsupported_entities: List[str]) -> str:
    """Highlight unsupported entities with styled mark tags, preserving whitespace."""
    if not unsupported_entities:
        return text

    highlighted = text
    # Sort by length descending so longer phrases match first
    for ent in sorted(unsupported_entities, key=len, reverse=True):
        if not ent.strip():
            continue
        # Use word boundary if alphanumeric
        pattern = re.compile(rf'\b({re.escape(ent)})\b', re.IGNORECASE) if ent.isalnum() else None
        replacement = r'<mark class="hallucinated">⚠ \1</mark>'
        if pattern:
            highlighted = pattern.sub(replacement, highlighted)
        else:
            highlighted = highlighted.replace(ent, f'<mark class="hallucinated">⚠ {ent}</mark>')

    return highlighted


def render_kg_comparison(source_entities: List[Entity], source_facts: List[Fact],
                         summary_entities: List[Entity], summary_facts: List[Fact],
                         unsupported_entities: Set[str]):
    """Render side-by-side NetworkX knowledge graphs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor='#0f172a')

    # Source graph
    g_src = nx.DiGraph()
    for e in source_entities:
        g_src.add_node(e.normalized_text)
    for f in source_facts:
        g_src.add_edge(f.subject, f.object, relation=f.relation)

    axes[0].set_facecolor('#0f172a')
    axes[0].set_title('Source Knowledge Graph', fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
    if len(g_src) > 0:
        pos1 = nx.spring_layout(g_src, k=1.8, seed=42)
        nx.draw_networkx_nodes(g_src, pos1, ax=axes[0], node_color='#3b82f6', node_size=1100, alpha=0.9)
        nx.draw_networkx_edges(g_src, pos1, ax=axes[0], edge_color='#64748b', width=1.5, alpha=0.8, arrowsize=14)
        nx.draw_networkx_labels(g_src, pos1, ax=axes[0], font_size=8, font_color='#ffffff', font_weight='bold')
    axes[0].axis('off')

    # Summary graph
    g_sum = nx.DiGraph()
    for e in summary_entities:
        g_sum.add_node(e.normalized_text)
    for f in summary_facts:
        g_sum.add_edge(f.subject, f.object, relation=f.relation)

    axes[1].set_facecolor('#0f172a')
    axes[1].set_title('Summary Knowledge Graph (Red = Unsupported)', fontsize=12, fontweight='bold', color='#f8fafc', pad=12)
    if len(g_sum) > 0:
        pos2 = nx.spring_layout(g_sum, k=1.8, seed=42)
        node_colors = ['#ef4444' if n in unsupported_entities else '#10b981' for n in g_sum.nodes()]
        nx.draw_networkx_nodes(g_sum, pos2, ax=axes[1], node_color=node_colors, node_size=1100, alpha=0.9)
        nx.draw_networkx_edges(g_sum, pos2, ax=axes[1], edge_color='#64748b', width=1.5, alpha=0.8, arrowsize=14)
        nx.draw_networkx_labels(g_sum, pos2, ax=axes[1], font_size=8, font_color='#ffffff', font_weight='bold')
    axes[1].axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#10b981', label='Supported by Source'),
        mpatches.Patch(color='#ef4444', label='Unsupported (Hallucinated)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, frameon=False,
               fontsize=10, labelcolor='#cbd5e1')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    return fig


# ─── Main Application ───────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🔍 KG Hallucination Checker</h1>
        <p>Detect factual inconsistencies in summaries by extracting and aligning knowledge graphs.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load models
    with st.spinner("Initializing models..."):
        config = load_config()
        nlp = load_nlp()
        embedding_model = load_embedding_model()

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")

        mode = st.radio(
            "Input Mode",
            ["Paste Article & Summary", "Generate Summary"],
            help="Choose whether to evaluate an existing summary or generate one on-the-fly."
        )

        st.markdown("---")
        st.subheader("Alignment Thresholds")
        fuzzy_threshold = st.slider(
            "Fuzzy Matching Threshold",
            0, 100, 80,
            help="Minimum character string similarity (0-100) for matching entities and facts."
        )
        embedding_threshold = st.slider(
            "Embedding Similarity Threshold",
            0.0, 1.0, 0.75,
            help="Minimum cosine similarity threshold (0-1) for semantic alignment via sentence transformers."
        )

        st.markdown("---")
        st.subheader("Scoring Weights")
        entity_weight = st.slider("Entity Precision Weight", 0.0, 1.0, 0.4)
        fact_weight = st.slider("Fact Precision Weight", 0.0, 1.0, 0.6)

        total_weight = entity_weight + fact_weight
        if total_weight > 0:
            st.caption(f"Normalized: Entities **{entity_weight/total_weight:.0%}**, Facts **{fact_weight/total_weight:.0%}**")

        st.markdown("---")
        st.subheader("Quick Presets")
        c_pre1, c_pre2 = st.columns(2)
        with c_pre1:
            if st.button("🚨 Load Hallucination", use_container_width=True):
                st.session_state['source_input'] = PRESET_ARTICLE
                st.session_state['summary_input'] = PRESET_HALLUCINATED
        with c_pre2:
            if st.button("✅ Load Faithful", use_container_width=True):
                st.session_state['source_input'] = PRESET_ARTICLE
                st.session_state['summary_input'] = PRESET_FAITHFUL

    # Initialize text state
    if 'source_input' not in st.session_state:
        st.session_state['source_input'] = ""
    if 'summary_input' not in st.session_state:
        st.session_state['summary_input'] = ""

    # Mode 1: Paste Article & Summary
    if mode == "Paste Article & Summary":
        st.subheader("📝 Input Texts")
        col_src, col_sum = st.columns(2)

        with col_src:
            source_text = st.text_area(
                "Source Article",
                value=st.session_state.get('source_input', ''),
                height=200,
                placeholder="Paste the ground-truth article or document here...",
                key="source_area"
            )

        with col_sum:
            summary_text = st.text_area(
                "Summary to Fact-Check",
                value=st.session_state.get('summary_input', ''),
                height=200,
                placeholder="Paste the machine-generated or human summary to evaluate...",
                key="summary_area"
            )

    # Mode 2: Generate Summary
    else:
        st.subheader("📝 Input Article & Model Selection")
        source_text = st.text_area(
            "Source Article",
            value=st.session_state.get('source_input', ''),
            height=180,
            placeholder="Paste the ground-truth article here...",
            key="source_gen_area"
        )

        gen_col1, gen_col2 = st.columns([2, 1])
        with gen_col1:
            gen_method = st.selectbox(
                "Summarization Model",
                ["BART (Abstractive - may hallucinate)", "TextRank (Extractive - faithful)", "Lead-3 (Extractive - faithful)"],
                help="Abstractive models synthesize text and can invent facts. Extractive models select sentences directly from the text."
            )
        with gen_col2:
            st.write("")
            st.write("")
            generate_clicked = st.button("🤖 Generate Summary", type="secondary", use_container_width=True)

        if generate_clicked:
            if not source_text.strip():
                st.warning("Please enter a source article first.")
            else:
                with st.spinner("Generating summary with model..."):
                    try:
                        if "BART" in gen_method:
                            bart_sum = load_bart_summarizer()
                            res = bart_sum.summarize(source_text)
                            st.session_state['generated_text'] = res.summary
                        elif "TextRank" in gen_method:
                            ext_sum = ExtractiveSummarizer(method="textrank")
                            res = ext_sum.summarize(source_text)
                            st.session_state['generated_text'] = res.summary
                        else:
                            ext_sum = ExtractiveSummarizer(method="lead3")
                            res = ext_sum.summarize(source_text)
                            st.session_state['generated_text'] = res.summary
                    except Exception as err:
                        st.error(f"Generation error: {err}")
                        logger.exception("Summarization error")

        summary_text = st.session_state.get('generated_text', '')

        if summary_text:
            st.markdown("**Generated Summary:**")
            st.info(summary_text)

    # Run Analysis Button
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyze Factual Consistency", type="primary", use_container_width=True)

    if analyze_clicked:
        if not source_text.strip():
            st.error("Please provide a source article.")
            return
        if not summary_text.strip():
            st.error("Please provide a summary to fact-check.")
            return

        with st.spinner("Analyzing knowledge graphs, aligning entities & facts..."):
            # 1. Extraction
            source_entities = extract_entities(source_text, nlp)
            source_facts = extract_facts(source_text, nlp)

            summary_entities = extract_entities(summary_text, nlp)
            summary_facts = extract_facts(summary_text, nlp)

            # 2. Alignment
            entity_alignments = align_entities(
                summary_entities, source_entities,
                fuzzy_threshold, embedding_threshold,
                embedding_model
            )

            fact_alignments = align_facts(
                summary_facts, source_facts,
                fuzzy_threshold, embedding_threshold,
                embedding_model
            )

            # 3. Scoring
            weights = {
                'entity_weight': entity_weight,
                'fact_weight': fact_weight
            }
            scores = compute_scores(entity_alignments, fact_alignments, weights)

        # Store in session state for inspection
        st.session_state['last_analysis'] = {
            'scores': scores,
            'source_entities': source_entities,
            'source_facts': source_facts,
            'summary_entities': summary_entities,
            'summary_facts': summary_facts,
            'entity_alignments': entity_alignments,
            'fact_alignments': fact_alignments,
            'summary_text': summary_text,
            'source_text': source_text
        }

    # Display Results if available
    if 'last_analysis' in st.session_state:
        data = st.session_state['last_analysis']
        scores: ScoreResult = data['scores']
        unsupported_entity_set = set(scores.unsupported_entities)
        unsupported_fact_set = set(scores.unsupported_facts)

        st.markdown("---")
        st.header("📊 Evaluation Results")

        # Top metric cards
        consistency = scores.consistency_score
        c_class = "score-card-high" if consistency >= 0.75 else "score-card-medium" if consistency >= 0.50 else "score-card-low"
        c_val_color = "val-green" if consistency >= 0.75 else "val-amber" if consistency >= 0.50 else "val-red"
        c_label = "🟢 Highly Faithful" if consistency >= 0.75 else "🟡 Moderate Inconsistency" if consistency >= 0.50 else "🔴 Hallucination Detected"

        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-title">Entity Precision</div>
                <div class="metric-value {c_val_color}">{scores.entity_precision:.3f}</div>
                <div class="metric-sub">{scores.aligned_entities} of {scores.total_entities} entities supported</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Fact Precision</div>
                <div class="metric-value {c_val_color}">{scores.fact_precision:.3f}</div>
                <div class="metric-sub">{scores.aligned_facts} of {scores.total_facts} facts supported</div>
            </div>
            <div class="metric-card {c_class}">
                <div class="metric-title">Consistency Score</div>
                <div class="metric-value {c_val_color}">{scores.consistency_score:.3f}</div>
                <div class="metric-sub">{c_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Highlighted text view
        st.subheader("📄 Summary with Detected Hallucinations")
        highlighted_html = highlight_unsupported_entities(data['summary_text'], scores.unsupported_entities)
        st.markdown(f"""
        <div style="background:#1e293b;padding:1.25rem;border-radius:10px;border:1px solid #334155;line-height:1.7;font-size:1.05rem;">
            {highlighted_html}
        </div>
        """, unsafe_allow_html=True)

        # Issues Breakdown Banner
        if scores.unsupported_entities or scores.unsupported_facts:
            st.markdown("<br>", unsafe_allow_html=True)
            col_ue, col_uf = st.columns(2)
            with col_ue:
                st.markdown(f"**⚠️ Unsupported Entities ({len(scores.unsupported_entities)})**")
                if scores.unsupported_entities:
                    for ent in scores.unsupported_entities:
                        st.markdown(f"- 🔴 `{ent}` *(not supported by source)*")
                else:
                    st.success("All entities are verified by the source.")

            with col_uf:
                st.markdown(f"**⚠️ Unsupported Facts ({len(scores.unsupported_facts)})**")
                if scores.unsupported_facts:
                    for s, r, o in scores.unsupported_facts:
                        st.markdown(f"- 🔴 **{s}** → *{r}* → **{o}**")
                else:
                    st.success("All facts are verified by the source.")
        else:
            st.success("✅ Perfect consistency! All extracted entities and facts are grounded in the source text.")

        # Detailed breakdown tabs
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("🔬 Knowledge Graph & Alignment Details")

        tab_kg, tab_se, tab_sf, tab_align = st.tabs([
            "Knowledge Graphs",
            "Summary Entities",
            "Summary Facts",
            "Alignment Trace"
        ])

        with tab_kg:
            st.markdown("Visual comparison of extracted knowledge graphs. Red nodes represent unsupported entities:")
            fig = render_kg_comparison(
                data['source_entities'], data['source_facts'],
                data['summary_entities'], data['summary_facts'],
                unsupported_entity_set
            )
            st.pyplot(fig)

        with tab_se:
            st.markdown("All entities extracted from the summary:")
            if not data['summary_entities']:
                st.info("No entities were extracted from the summary.")
            else:
                chips_html = []
                for e in data['summary_entities']:
                    is_bad = e.normalized_text in unsupported_entity_set
                    cls = "chip-unsupported" if is_bad else "chip-supported"
                    icon = "✗" if is_bad else "✓"
                    chips_html.append(
                        f'<span class="chip {cls}">'
                        f'{icon} {e.text}'
                        f'<span class="chip-type">{e.type}</span>'
                        f'</span>'
                    )
                st.markdown(" ".join(chips_html), unsafe_allow_html=True)

        with tab_sf:
            st.markdown("All subject-relation-object facts extracted from the summary:")
            if not data['summary_facts']:
                st.info("No relational facts were extracted from the summary.")
            else:
                for f in data['summary_facts']:
                    is_bad = (f.subject, f.relation, f.object) in unsupported_fact_set
                    cls = "fact-card-unsupported" if is_bad else "fact-card-supported"
                    icon = "✗ Unsupported" if is_bad else "✓ Supported"
                    st.markdown(f"""
                    <div class="fact-card {cls}">
                        <strong>{icon}</strong> &nbsp;
                        <b>{f.subject}</b> ──[ <em>{f.relation}</em> ]──▶ <b>{f.object}</b>
                    </div>
                    """, unsafe_allow_html=True)

        with tab_align:
            st.markdown("Detailed alignment methods and confidence scores for summary elements:")
            col_at1, col_at2 = st.columns(2)
            with col_at1:
                st.markdown("**Entity Alignments:**")
                for ea in data['entity_alignments']:
                    status = "🔴" if ea.method == 'none' else "🟢"
                    match_info = f"aligned with `{ea.source_entity}` ({ea.method}, {ea.score:.2f})" if ea.source_entity else "no match"
                    st.markdown(f"- {status} `{ea.summary_entity}` → {match_info}")

            with col_at2:
                st.markdown("**Fact Alignments:**")
                for fa in data['fact_alignments']:
                    status = "🔴" if fa.method == 'none' else "🟢"
                    match_info = f"aligned with `{fa.source_fact}` ({fa.method}, {fa.score:.2f})" if fa.source_fact else "no match"
                    st.markdown(f"- {status} `{fa.summary_fact}` → {match_info}")


if __name__ == "__main__":
    main()

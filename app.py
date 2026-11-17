"""
Streamlit app for KG Hallucination Checker.

Interactive interface to check factual consistency of summaries using knowledge graphs.
"""

import streamlit as st
import spacy
from sentence_transformers import SentenceTransformer
import yaml
from pathlib import Path
import logging

from kg_checker.extract import extract_entities, extract_facts
from kg_checker.kg import build_kg_from_text
from kg_checker.align import align_entities, align_facts
from kg_checker.score import compute_scores
from kg_checker.summarizers import AbstractiveSummarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="KG Hallucination Checker",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
@st.cache_resource
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

@st.cache_resource
def load_models():
    """Load spaCy and embedding models."""
    nlp = spacy.load('en_core_web_sm')
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return nlp, embedding_model

@st.cache_resource
def load_summarizer():
    """Load BART summarizer."""
    config = load_config()
    return AbstractiveSummarizer(
        model_name=config['models']['abstractive_model'],
        cache=True
    )

def highlight_unsupported_entities(summary_text, unsupported_entities):
    """Highlight unsupported entities in summary text."""
    highlighted = summary_text
    for entity in unsupported_entities:
        # Simple highlighting - in real app would use proper tokenization
        highlighted = highlighted.replace(
            entity,
            f"🔴 **{entity}**"
        )
    return highlighted

def visualize_entities(entities, unsupported):
    """Create a visual representation of entities."""
    cols = st.columns(len(entities) if entities else 1)
    if not entities:
        st.info("No entities extracted")
        return

    for col, entity in zip(cols, entities):
        is_unsupported = entity.normalized_text in unsupported
        color = "🔴" if is_unsupported else "🟢"
        with col:
            st.markdown(f"""
            {color} **{entity.text}**

            Type: `{entity.type}`
            """)

def visualize_facts(facts, unsupported):
    """Create a visual representation of facts."""
    if not facts:
        st.info("No facts extracted")
        return

    for fact in facts:
        is_unsupported = (fact.subject, fact.relation, fact.object) in unsupported
        status = "🔴 Unsupported" if is_unsupported else "🟢 Supported"

        st.markdown(f"""
        {status}

        **{fact.subject}** → *{fact.relation}* → **{fact.object}**
        """)

def main():
    st.title("🔍 KG Hallucination Checker")
    st.markdown("""
    Fact-check summaries using knowledge graphs. This tool extracts entities and facts
    from both source documents and their summaries, aligns them, and computes a
    **factual consistency score** to detect hallucinations.
    """)

    # Load models
    with st.spinner("Loading models..."):
        config = load_config()
        nlp, embedding_model = load_models()
        summarizer = load_summarizer()

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        mode = st.radio(
            "Mode",
            ["Paste Article & Summary", "Generate Summary"],
            help="Choose how to provide the article and summary"
        )

        st.subheader("Alignment Thresholds")
        fuzzy_threshold = st.slider(
            "Fuzzy matching threshold",
            0, 100, 85,
            help="Similarity threshold for fuzzy string matching (0-100)"
        )

        embedding_threshold = st.slider(
            "Embedding similarity threshold",
            0.0, 1.0, 0.75,
            help="Cosine similarity threshold for embedding matching (0-1)"
        )

        st.subheader("Scoring Weights")
        entity_weight = st.slider(
            "Entity weight",
            0.0, 1.0, 0.4,
            help="Weight for entity-level consistency"
        )
        fact_weight = st.slider(
            "Fact weight",
            0.0, 1.0, 0.6,
            help="Weight for fact-level consistency"
        )

    # Main content
    if mode == "Paste Article & Summary":
        st.header("📝 Input")

        col1, col2 = st.columns(2)

        with col1:
            source_text = st.text_area(
                "Source Article",
                height=200,
                placeholder="Paste the original article here...",
                key="source"
            )

        with col2:
            summary_text = st.text_area(
                "Summary",
                height=200,
                placeholder="Paste the summary here...",
                key="summary"
            )

    else:  # Generate Summary
        st.header("📝 Input")

        source_text = st.text_area(
            "Source Article",
            height=200,
            placeholder="Paste the original article here...",
            key="source_gen"
        )

        gen_method = st.selectbox(
            "Summarization Method",
            ["BART (Abstractive)", "TextRank (Extractive)", "Lead-3 (Extractive)"],
            help="Extractive methods prevent hallucination by selecting sentences directly from the text."
        )

        if st.button("🤖 Generate Summary"):
            with st.spinner("Generating summary..."):
                if "BART" in gen_method:
                    result = summarizer.summarize(source_text)
                    summary_text = result.summary
                elif "TextRank" in gen_method:
                    ext_sum = ExtractiveSummarizer(method="textrank")
                    result = ext_sum.summarize(source_text)
                    summary_text = result.summary
                else:
                    ext_sum = ExtractiveSummarizer(method="lead3")
                    result = ext_sum.summarize(source_text)
                    summary_text = result.summary

                st.session_state['summary_gen'] = summary_text

        summary_text = st.session_state.get('summary_gen', '')

        if summary_text:
            st.info(f"Generated summary ({len(summary_text)} chars)")
            st.text_area(
                "Generated Summary",
                summary_text,
                height=150,
                disabled=True,
                key="summary_display"
            )

    # Analysis button
    if st.button("🔍 Analyze", type="primary"):
        if not source_text or not summary_text:
            st.error("Please provide both article and summary")
            return

        with st.spinner("Analyzing..."):
            # Extract entities and facts
            source_entities = extract_entities(source_text, nlp)
            source_facts = extract_facts(source_text, nlp)

            summary_entities = extract_entities(summary_text, nlp)
            summary_facts = extract_facts(summary_text, nlp)

            # Align
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

            # Score
            weights = {
                'entity_weight': entity_weight,
                'fact_weight': fact_weight
            }
            scores = compute_scores(entity_alignments, fact_alignments, weights)

            # Display results
            st.header("📊 Results")

            # Scores
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Entity Precision",
                    f"{scores.entity_precision:.3f}",
                    help="Proportion of summary entities supported by source"
                )

            with col2:
                st.metric(
                    "Fact Precision",
                    f"{scores.fact_precision:.3f}",
                    help="Proportion of summary facts supported by source"
                )

            with col3:
                color = "🟢" if scores.consistency_score >= 0.7 else "🟡" if scores.consistency_score >= 0.5 else "🔴"
                st.metric(
                    "Consistency Score",
                    f"{scores.consistency_score:.3f}",
                    help=f"{color} Higher is better (1.0 = fully consistent)"
                )

            # Hallucinations
            st.subheader("⚠️ Detected Issues")

            if scores.unsupported_entities or scores.unsupported_facts:
                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Unsupported Entities ({len(scores.unsupported_entities)}):**")
                    if scores.unsupported_entities:
                        for entity in scores.unsupported_entities:
                            st.markdown(f"- 🔴 {entity}")
                    else:
                        st.info("All entities are supported")

                with col2:
                    st.write(f"**Unsupported Facts ({len(scores.unsupported_facts)}):**")
                    if scores.unsupported_facts:
                        for subj, rel, obj in scores.unsupported_facts:
                            st.markdown(f"- 🔴 {subj} → {rel} → {obj}")
                    else:
                        st.info("All facts are supported")
            else:
                st.success("✅ No hallucinations detected!")

            # Summary with highlights
            st.subheader("📄 Summary with Highlights")
            highlighted_summary = highlight_unsupported_entities(
                summary_text,
                scores.unsupported_entities
            )
            st.markdown(highlighted_summary)

            # Entities and facts breakdown
            st.subheader("🔬 Extracted Information")

            tab1, tab2, tab3, tab4 = st.tabs([
                "Summary Entities",
                "Summary Facts",
                "Source Entities",
                "Source Facts"
            ])

            with tab1:
                visualize_entities(summary_entities, scores.unsupported_entities)

            with tab2:
                visualize_facts(summary_facts, scores.unsupported_facts)

            with tab3:
                visualize_entities(source_entities, [])

            with tab4:
                visualize_facts(source_facts, [])

            # Statistics
            st.subheader("📈 Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Summary Entities", scores.total_entities)

            with col2:
                st.metric("Summary Facts", scores.total_facts)

            with col3:
                st.metric("Aligned Entities", scores.aligned_entities)

            with col4:
                st.metric("Aligned Facts", scores.aligned_facts)

if __name__ == "__main__":
    main()

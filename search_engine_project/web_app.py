import html
import os

import matplotlib.pyplot as plt
import streamlit as st

from MoteurRecherche import SearchEngine
from evaluation import SAMPLE_RELEVANCE, precision_recall_points


BOOLEAN_OPERATORS = {"AND", "OR", "NOT"}


def apply_custom_styles():
    """Inject a dark visual style for the Streamlit interface."""
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(34, 197, 94, 0.10), transparent 25%),
                    radial-gradient(circle at top right, rgba(245, 158, 11, 0.12), transparent 22%),
                    linear-gradient(180deg, #050b16 0%, #0a1220 55%, #07111d 100%);
                color: #e5eefb;
            }

            .main .block-container {
                max-width: 1180px;
                padding-top: 2rem;
                padding-bottom: 2rem;
            }

            .hero-card,
            .search-panel,
            .agent-card,
            .result-card,
            .evaluation-card,
            .footer-card {
                background: rgba(11, 19, 34, 0.88);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 24px;
                box-shadow: 0 20px 45px rgba(0, 0, 0, 0.30);
            }

            .hero-card {
                padding: 2rem 2.2rem;
                margin-bottom: 1.25rem;
                background:
                    linear-gradient(135deg, rgba(180, 83, 9, 0.20), rgba(11, 19, 34, 0.92) 36%),
                    linear-gradient(135deg, #0b1322 0%, #101b33 52%, #132238 100%);
            }

            .hero-title {
                font-size: 2.5rem;
                font-weight: 800;
                letter-spacing: -0.03em;
                color: #f8fafc;
                margin-bottom: 0.45rem;
            }

            .hero-text,
            .agent-text,
            .result-preview,
            .footer-text {
                color: #d5dfef;
                line-height: 1.7;
            }

            .section-label {
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                font-weight: 700;
                color: #fbbf24;
                margin-bottom: 0.5rem;
            }

            .search-panel,
            .agent-card,
            .evaluation-card,
            .footer-card {
                padding: 1.2rem;
                margin-bottom: 1rem;
            }

            .agent-title,
            .results-title {
                font-size: 1.15rem;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 0.5rem;
            }

            .agent-chip {
                display: inline-block;
                padding: 0.28rem 0.7rem;
                border-radius: 999px;
                background: rgba(250, 204, 21, 0.14);
                border: 1px solid rgba(250, 204, 21, 0.25);
                color: #fde68a;
                font-size: 0.82rem;
                font-weight: 700;
                margin-right: 0.5rem;
                margin-bottom: 0.5rem;
            }

            .result-card {
                padding: 1.1rem 1.15rem;
                margin-bottom: 1rem;
            }

            .result-topline {
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: center;
                margin-bottom: 0.55rem;
            }

            .result-name {
                font-size: 1.06rem;
                font-weight: 700;
                color: #f8fafc;
            }

            .result-score {
                background: rgba(34, 197, 94, 0.14);
                color: #bbf7d0;
                padding: 0.28rem 0.72rem;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 700;
                white-space: nowrap;
            }

            .image-caption {
                text-align: center;
                color: #cbd5e1;
                font-size: 0.88rem;
                margin-top: 0.45rem;
            }

            h3, p, label, .stCaption, .stMarkdown, .stText {
                color: #e5eefb !important;
            }

            [data-testid="stTextInput"] label {
                color: #f8fafc !important;
                font-weight: 600;
            }

            [data-testid="stTextInput"] input {
                background: rgba(15, 23, 42, 0.92) !important;
                color: #f8fafc !important;
                border: 1px solid rgba(148, 163, 184, 0.28) !important;
                border-radius: 14px !important;
            }

            [data-testid="stTextInput"] input::placeholder {
                color: #94a3b8 !important;
            }

            .stButton > button,
            .stFormSubmitButton > button {
                background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
                color: #1f2937 !important;
                border: none;
                border-radius: 14px;
                font-weight: 800;
                min-height: 3rem;
            }

            .stButton > button:hover,
            .stFormSubmitButton > button:hover {
                background: linear-gradient(135deg, #fbbf24 0%, #ea580c 100%);
                color: #111827 !important;
            }

            [data-testid="metric-container"] {
                background: #f4efe6;
                border: 1px solid rgba(15, 23, 42, 0.12);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: none;
            }

            [data-testid="metric-container"] label,
            [data-testid="metric-container"] [data-testid="stMetricLabel"],
            [data-testid="metric-container"] [data-testid="stMetricValue"],
            [data-testid="metric-container"] [data-testid="stMetricDelta"] {
                color: #111827 !important;
            }

            [data-testid="stMetricValue"] {
                font-weight: 800;
            }

            [data-testid="stExpander"] {
                background: rgba(15, 23, 42, 0.78);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 16px;
                color: #e5eefb !important;
            }

            .footer-card {
                margin-top: 1.5rem;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def resolve_image_directory(base_dir):
    """Return the preferred image directory for the web interface."""
    dataset_images_dir = os.path.join(base_dir, "dataset", "images")
    fallback_images_dir = os.path.join(base_dir, "images")

    if os.path.isdir(dataset_images_dir):
        return dataset_images_dir
    return fallback_images_dir


@st.cache_resource
def load_search_engine():
    """Create the search engine once and reuse it across Streamlit reruns."""
    return SearchEngine()


def resolve_display_images(engine, query):
    """Get related images and remap them to dataset/images/ when available."""
    image_directory = resolve_image_directory(engine.base_dir)
    related_images = []

    for image_path in engine.get_related_images(query):
        image_name = os.path.basename(image_path)
        display_path = os.path.join(image_directory, image_name)
        if os.path.exists(display_path):
            related_images.append(display_path)

    return related_images


def decide_search_strategy(query, improved_query, suggestions):
    """Pick the search strategy from the submitted query and AI guidance."""
    tokens = query.split()
    has_boolean_query = any(token.upper() in BOOLEAN_OPERATORS for token in tokens)

    if has_boolean_query:
        return {
            "model_name": "Boolean",
            "search_query": query,
            "strategy": "Boolean retrieval on the original query because boolean operators were detected.",
        }

    if suggestions:
        return {
            "model_name": "Vector",
            "search_query": improved_query,
            "strategy": "Vector retrieval on an AI-expanded query to capture related concepts.",
        }

    return {
        "model_name": "Vector",
        "search_query": query,
        "strategy": "Vector retrieval on the original query because no helpful expansion was available.",
    }


def run_ai_guided_search(engine, query):
    """Process a query through the AI agent, pick a strategy, and return results."""
    suggestions = engine.ai_agent.suggest_related_keywords(query)
    improved_query = engine.ai_agent.improve_query(query)
    strategy = decide_search_strategy(query, improved_query, suggestions)
    results = engine.search(strategy["search_query"], strategy["model_name"])
    ai_response = engine.get_ai_response(query, results, strategy["model_name"])

    return {
        "results": results,
        "suggestions": ai_response["suggestions"],
        "improved_query": ai_response["improved_query"],
        "explanation": ai_response["explanation"],
        "model_name": strategy["model_name"],
        "strategy": strategy["strategy"],
        "search_query": strategy["search_query"],
    }


def render_agent_summary(search_payload, original_query):
    """Display how the AI agent handled the user query."""
    suggestions = search_payload["suggestions"]
    suggestions_markup = "".join(
        f'<span class="agent-chip">{html.escape(suggestion)}</span>'
        for suggestion in suggestions
    )

    st.markdown(
        f"""
        <div class="agent-card">
            <div class="section-label">AI Agent</div>
            <div class="agent-title">Search strategy selected by the agent</div>
            <div class="agent-text"><strong>Original query:</strong> {html.escape(original_query)}</div>
            <div class="agent-text"><strong>Chosen model:</strong> {html.escape(search_payload["model_name"])}</div>
            <div class="agent-text"><strong>Executed query:</strong> {html.escape(search_payload["search_query"])}</div>
            <div class="agent-text"><strong>Why this strategy:</strong> {html.escape(search_payload["strategy"])}</div>
            <div class="agent-text"><strong>Agent explanation:</strong> {html.escape(search_payload["explanation"])}</div>
            <div class="agent-text"><strong>Improved query:</strong> {html.escape(search_payload["improved_query"])}</div>
            <div class="agent-text"><strong>Suggested keywords:</strong></div>
            <div>{suggestions_markup if suggestions_markup else '<span class="agent-chip">No suggestions available</span>'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_results(engine, original_query, search_payload):
    """Display documents and images returned by the AI-guided search."""
    results = search_payload["results"]

    overview_columns = st.columns(3)
    overview_columns[0].metric("Submitted Query", original_query)
    overview_columns[1].metric("Model Used", search_payload["model_name"])
    overview_columns[2].metric("Results Found", len(results))

    st.markdown("### Search Results")

    for rank, (document_name, score) in enumerate(results, start=1):
        document_content = engine.get_document_content(document_name)
        preview = document_content.strip()
        if len(preview) > 400:
            preview = preview[:400].rstrip() + "..."

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-topline">
                    <div class="result-name">{rank}. {html.escape(document_name)}</div>
                    <div class="result-score">Score {score:.3f}</div>
                </div>
                <div class="result-preview">
                    {html.escape(preview) if preview else "No preview available for this document."}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander(f"View full document: {document_name}"):
            st.text(document_content if document_content else "No content available.")

    related_images = resolve_display_images(engine, original_query)
    st.markdown("### Related Images")

    if not related_images:
        st.info("No related image found for this query.")
        return

    columns = st.columns(min(len(related_images), 3))
    for column, image_path in zip(columns, related_images):
        with column:
            st.image(image_path, caption=os.path.basename(image_path), use_container_width=True)
            st.markdown(
                f'<div class="image-caption">{html.escape(os.path.basename(image_path))}</div>',
                unsafe_allow_html=True,
            )


def render_evaluation(engine, query, results):
    """Display precision, recall, and a precision-recall curve when available."""
    st.markdown("### Evaluation")

    if not engine.has_evaluation_reference(query):
        available_queries = ", ".join(sorted(SAMPLE_RELEVANCE.keys()))
        st.info(
            "No predefined evaluation set exists for this query. "
            f"Try one of these queries: {available_queries}"
        )
        return

    metrics = engine.evaluate_results(query, results)
    points = precision_recall_points(results, metrics["relevant_documents"])

    st.markdown('<div class="evaluation-card">', unsafe_allow_html=True)
    metric_columns = st.columns(2)
    metric_columns[0].metric("Precision", f"{metrics['precision']:.2f}")
    metric_columns[1].metric("Recall", f"{metrics['recall']:.2f}")

    st.caption(
        "Precision and Recall are computed with the existing project evaluation logic."
    )

    if points:
        recalls = [point[0] for point in points]
        precisions = [point[1] for point in points]

        figure, axis = plt.subplots(figsize=(7, 4))
        figure.patch.set_facecolor("#0b1322")
        axis.set_facecolor("#111c2e")
        axis.plot(recalls, precisions, marker="o", color="#f59e0b", linewidth=2)
        axis.set_title(f"Precision-Recall Curve for '{query}'", color="#f8fafc")
        axis.set_xlabel("Recall", color="#e5eefb")
        axis.set_ylabel("Precision", color="#e5eefb")
        axis.set_xlim(0, 1.05)
        axis.set_ylim(0, 1.05)
        axis.grid(True, alpha=0.22, color="#94a3b8")
        axis.tick_params(colors="#e5eefb")
        for spine in axis.spines.values():
            spine.set_color("#94a3b8")
        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

    st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    """Display developer credits."""
    st.markdown(
        """
        <div class="footer-card">
            <div class="section-label">Credits</div>
            <div class="footer-text">Developed by Yosri Khedher and Omrane Khedher</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    """Build the Streamlit interface for the multimedia search engine."""
    st.set_page_config(page_title="Multimedia Search Engine", layout="wide")
    apply_custom_styles()

    engine = load_search_engine()

    st.markdown(
        """
        <div class="hero-card">
            <div class="section-label">AI-Powered Multimedia Retrieval</div>
            <div class="hero-title">Multimedia Search Engine</div>
            <div class="hero-text">
                Submit a query and let the AI agent choose the best search strategy, expand the request when useful,
                and return the most relevant documents with supporting multimedia content.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="search-panel">', unsafe_allow_html=True)
    with st.form("search_form"):
        query = st.text_input(
            "Enter your query",
            placeholder="Example: machine learning OR computer vision",
        )
        submitted = st.form_submit_button("Search with AI Agent", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        cleaned_query = query.strip()
        if not cleaned_query:
            st.warning("Please enter a query before searching.")
        else:
            search_payload = run_ai_guided_search(engine, cleaned_query)
            results = search_payload["results"]

            render_agent_summary(search_payload, cleaned_query)

            if not results:
                st.warning("The AI agent completed the search, but no results were found for this query.")
            else:
                render_results(engine, cleaned_query, search_payload)
                render_evaluation(engine, cleaned_query, results)

    render_footer()


if __name__ == "__main__":
    main()

import collections
import re

import numpy as np
import streamlit as st

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from crewai import Crew, Process

from agents import (
    get_agent_llm,
    create_triage_agent,
    create_compiler_agent,
    create_router_agent,
    create_automation_agent,
)

from tasks import define_workflow_tasks


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SOP-Orchestrator AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


APP_TITLE = "SOP-Orchestrator AI"
APP_SUBTITLE = (
    "Multi-Agent Supply Chain Contingency Planner"
)

MODEL_DISPLAY_NAME = "Gemini 3.8 Flash"

CHUNK_SIZE = 500
TOP_K = 3
SIMILARITY_THRESHOLD = 0.05


# ============================================================
# SESSION STATE
# ============================================================

if "sop_chunks" not in st.session_state:
    st.session_state.sop_chunks = []

if "sop_sources" not in st.session_state:
    st.session_state.sop_sources = []

if "retrieved_context" not in st.session_state:
    st.session_state.retrieved_context = []

if "incident" not in st.session_state:
    st.session_state.incident = ""

if "workflow_output" not in st.session_state:
    st.session_state.workflow_output = None

if "workflow_approved" not in st.session_state:
    st.session_state.workflow_approved = False

if "workflow_rejected" not in st.session_state:
    st.session_state.workflow_rejected = False


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        font-size: 1.15rem;
        color: #666666;
        margin-bottom: 1.2rem;
    }

    .status-card {
        border: 1px solid #d9d9d9;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        background-color: #fafafa;
    }

    .workflow-card {
        border: 1px solid #d9d9d9;
        border-radius: 12px;
        padding: 20px;
        background-color: #ffffff;
        margin-top: 15px;
    }

    .approval-card {
        border: 2px solid #999999;
        border-radius: 12px;
        padding: 20px;
        background-color: #fafafa;
        margin-top: 20px;
    }

    .small-label {
        font-size: 0.85rem;
        color: #666666;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 SOP-Orchestrator AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Multi-Agent Supply Chain Contingency Planner'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    "Converts operational incidents into SOP-grounded "
    "action plans using Local RAG, CrewAI multi-agent "
    "workflow and mandatory human approval."
)

st.divider()

st.markdown(
    """
    **Project Leader:** Masood Ur Rehman

    **Technology:** CrewAI + Gemini 3.8 Flash + Local TF-IDF RAG

    **Cost Model:** Free-tier / no paid API dependency
    """
)


# ============================================================
# SIDEBAR — WORKFLOW STATUS
# ============================================================

with st.sidebar:

    st.header("⚙️ System Configuration")

    st.markdown("### Workflow")

    workflow_steps = [
        "Upload approved SOP",
        "Describe incident",
        "Retrieve evidence",
        "Triage incident",
        "Compile SOP actions",
        "Route departments",
        "Prepare executive brief",
        "Human approval",
    ]

    for index, step in enumerate(workflow_steps, start=1):
        st.write(f"**{index}.** {step}")

    st.divider()

    st.caption(
        "The AI prepares a grounded workflow for human review. "
        "It does not independently execute business transactions."
    )

    st.divider()

    st.markdown("### 💰 Cost Control")

    st.caption(
        "Local PDF processing and TF-IDF retrieval require "
        "no external API calls."
    )

    st.caption(
        "Only the four CrewAI agents use Gemini."
    )

    st.caption(
        "Agents are configured for one-pass execution and "
        "no automatic agent retries."
    )


# ============================================================
# SECTION 1 — SOP UPLOAD
# ============================================================

st.header("1. 📚 Upload Approved SOP")

uploaded_files = st.file_uploader(
    "Upload one or more approved SOP PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help=(
        "The system extracts text locally and creates a "
        "TF-IDF searchable knowledge base."
    )
)

# SAFEGUARD GUARDRAIL: Reset context loops if uploader is cleared by user
if not uploaded_files:
    if st.session_state.sop_chunks or st.session_state.sop_sources:
        st.session_state.sop_chunks = []
        st.session_state.sop_sources = []
        st.session_state.retrieved_context = []
        st.session_state.workflow_output = None
        st.session_state.workflow_approved = False
        st.session_state.workflow_rejected = False


def split_text_into_chunks(text, chunk_size=500):

    words = text.split()

    chunks = []

    for start in range(0, len(words), chunk_size):
        chunk = " ".join(
            words[start:start + chunk_size]
        )

        if chunk.strip():
            chunks.append(chunk.strip())

    return chunks


def build_local_knowledge_base(files):

    all_chunks = []
    all_sources = []

    for uploaded_file in files:

        try:

            reader = PdfReader(uploaded_file)

            full_text = []

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                try:
                    page_text = page.extract_text() or ""

                except Exception:
                    page_text = ""

                if page_text.strip():
                    full_text.append(
                        f"[Page {page_number}]\n"
                        f"{page_text.strip()}"
                    )

            combined_text = "\n\n".join(full_text)

            file_chunks = split_text_into_chunks(
                combined_text,
                CHUNK_SIZE
            )

            for chunk_number, chunk in enumerate(
                file_chunks,
                start=1
            ):

                all_chunks.append(chunk)

                all_sources.append(
                    {
                        "file": uploaded_file.name,
                        "chunk": chunk_number,
                    }
                )

        except Exception as exc:

            st.error(
                f"Could not read {uploaded_file.name}: {exc}"
            )

    return all_chunks, all_sources


if uploaded_files:

    if st.button(
        "🔄 Build Local SOP Knowledge Base",
        type="primary"
    ):

        with st.spinner(
            "Extracting SOP text and building local TF-IDF index..."
        ):

            chunks, sources = build_local_knowledge_base(
                uploaded_files
            )

            st.session_state.sop_chunks = chunks
            st.session_state.sop_sources = sources
            st.session_state.workflow_output = None
            st.session_state.workflow_approved = False
            st.session_state.workflow_rejected = False

        if chunks:

            st.success(
                f"Knowledge base ready: "
                f"{len(uploaded_files)} SOP file(s), "
                f"{len(chunks)} searchable chunk(s)."
            )

        else:

            st.error(
                "No readable text was extracted from the uploaded SOP files."
            )


if st.session_state.sop_chunks:

    st.success(
        f"✅ Local SOP knowledge base active — "
        f"{len(st.session_state.sop_chunks)} chunks"
    )


# ============================================================
# SECTION 2 — INCIDENT
# ============================================================

st.header("2. 🚨 Describe Operational Incident")

incident = st.text_area(
    "Describe the operational situation",
    value=st.session_state.incident,
    height=150,
    placeholder=(
        "Example:\n"
        "During manual inspection of an incoming marine cargo "
        "container, our operator discovered that the high-security "
        "bolt seal shows physical deformation and un-logged "
        "tracking marks. What are the applicable procedures?"
    )
)

st.session_state.incident = incident


# ============================================================
# LOCAL RAG RETRIEVAL
# ============================================================

def retrieve_sop_evidence(
    query,
    chunks,
    sources,
    top_k=3,
    threshold=0.05
):

    if not query.strip():
        return []

    if not chunks:
        return []

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    try:
        matrix = vectorizer.fit_transform(chunks)
        query_vector = vectorizer.transform([query])
        # 💎 FIX: Explicitly extract and flatten the similarity array securely
        similarities = np.array(cosine_similarity(query_vector, matrix)).flatten()
    except Exception:
        return []

    ranked_indices = np.argsort(similarities)[::-1]
    results = []

    for index in ranked_indices[:top_k]:
        score = float(similarities[index])
        if score < threshold:
            continue

        results.append(
            {
                "chunk": chunks[index],
                "source": sources[index],
                "score": score,
            }
        )

    return results
    for index in ranked_indices[:top_k]:

        score = float(similarities[index])

        if score < threshold:
            continue

        results.append(
            {

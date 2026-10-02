import os
import collections

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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SOP-Orchestrator AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# SESSION STATE
# ============================================================

if "crew_output" not in st.session_state:
    st.session_state.crew_output = None

if "approval_status" not in st.session_state:
    st.session_state.approval_status = "Pending Human Review"

if "retrieved_chunks" not in st.session_state:
    st.session_state.retrieved_chunks = []

if "retrieval_scores" not in st.session_state:
    st.session_state.retrieval_scores = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div style="
        padding: 20px;
        border: 2px solid #444;
        border-radius: 12px;
        margin-bottom: 20px;
    ">
        <h1>🤖 SOP-Orchestrator AI</h1>

        <h3>
        Multi-Agent Supply Chain Contingency Planner
        </h3>

        <p>
        Converts operational incidents into SOP-grounded action plans
        using Local RAG, CrewAI multi-agent workflow and mandatory
        human approval.
        </p>

        <hr>

        <p>
        <b>Project Leader:</b> Masood Ur Rehman
        &nbsp; | &nbsp;

        <b>Engineers:</b>
        Fatima Ijaz • Muhammad Aslam • Shakeel Ahmed •
        Sami Ur Rahman • Muhammad Haroon Jan
        </p>

        <p>
        <b>Technology:</b>
        CrewAI + Gemini 2.5 Flash + Local TF-IDF RAG
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System Configuration")

    st.markdown(
        """
        **Workflow**

        1. Upload approved SOP
        2. Describe incident
        3. Retrieve evidence
        4. Triage incident
        5. Compile SOP actions
        6. Route departments
        7. Prepare executive brief
        8. Human approval
        """
    )

    st.divider()

    st.info(
        "The AI prepares a grounded workflow for human review. "
        "It does not independently execute business transactions."
    )


# ============================================================
# SOP DOCUMENT INGESTION
# ============================================================

st.subheader("📚 1. Approved SOP Knowledge Base")

uploaded_files = st.file_uploader(
    "Upload one or more approved SOP PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf_chunks(uploaded_pdf):
    """
    Extract PDF text and divide it into approximately
    500-word chunks.
    """

    try:
        reader = PdfReader(uploaded_pdf)

        full_text = []

        for page_number, page in enumerate(reader.pages, start=1):

            try:
                text = page.extract_text() or ""

                if text.strip():
                    full_text.append(
                        f"[Page {page_number}]\n{text}"
                    )

            except Exception:
                continue

        combined_text = "\n\n".join(full_text).strip()

        if not combined_text:
            return []

        words = combined_text.split()

        chunk_size = 500

        chunks = []

        for start in range(0, len(words), chunk_size):

            chunk_words = words[start:start + chunk_size]

            if chunk_words:
                chunks.append(
                    " ".join(chunk_words)
                )

        return chunks

    except Exception as exc:
        st.error(
            f"Could not read {uploaded_pdf.name}: {exc}"
        )

        return []


# ============================================================
# BUILD KNOWLEDGE BASE
# ============================================================

all_chunks = []
chunk_sources = []

if uploaded_files:

    for uploaded_pdf in uploaded_files:

        pdf_chunks = extract_pdf_chunks(uploaded_pdf)

        for chunk_number, chunk in enumerate(
            pdf_chunks,
            start=1
        ):

            all_chunks.append(chunk)

            chunk_sources.append(
                {
                    "file": uploaded_pdf.name,
                    "chunk": chunk_number
                }
            )

    if all_chunks:

        st.success(
            f"Loaded {len(uploaded_files)} SOP file(s) "
            f"and created {len(all_chunks)} searchable evidence chunks."
        )

    else:

        st.warning(
            "The uploaded PDF files did not contain readable text."
        )


# ============================================================
# INCIDENT INPUT
# ============================================================

st.subheader("🚨 2. Operational Incident")

incident_description = st.text_area(
    "Describe the operational incident",
    height=160,
    placeholder=(
        "Example:\n"
        "Damaged material was received at the warehouse. "
        "Several cartons appear physically damaged during receiving. "
        "The receiving team has not yet released the material to production."
    )
)


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_relevant_chunks(
    query,
    chunks,
    sources,
    top_k=5,
    threshold=0.05
):
    """
    Local TF-IDF retrieval.

    This is the RAG grounding layer.
    It is not represented as an independent AI agent.
    """

    if not query.strip():
        return [], []

    if not chunks:
        return [], []

    try:

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )

        document_matrix = vectorizer.fit_transform(chunks)

        query_vector = vectorizer.transform(
            [query]
        )

        similarities = cosine_similarity(
            query_vector,
            document_matrix
        )[0]

        ranked_indices = np.argsort(
            similarities
        )[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            score = float(similarities[index])

            if score >= threshold:

                results.append(
                    {
                        "text": chunks[index],
                        "score": score,
                        "source": sources[index]
                    }
                )

        return results, vectorizer

    except Exception as exc:

        st.error(
            f"RAG retrieval failed: {exc}"
        )

        return [], []


# ============================================================
# RUN WORKFLOW
# ============================================================

st.subheader("🧠 3. Run SOP-Orchestrator")

run_workflow = st.button(
    "🚀 Analyze Incident & Prepare Action Plan",
    type="primary",
    use_container_width=True
)


if run_workflow:

    # --------------------------------------------------------
    # BASIC VALIDATION
    # --------------------------------------------------------

    if not incident_description.strip():

        st.warning(
            "Please describe the operational incident first."
        )

        st.stop()

    if not all_chunks:

        st.warning(
            "Please upload at least one readable approved SOP PDF."
        )

        st.stop()

    # --------------------------------------------------------
    # RAG RETRIEVAL
    # --------------------------------------------------------

    with st.spinner(
        "🔎 Searching approved SOP evidence..."
    ):

        retrieval_results, _ = retrieve_relevant_chunks(
            incident_description,
            all_chunks,
            chunk_sources,
            top_k=5,
            threshold=0.05
        )

    if not retrieval_results:

        st.error(
            "No sufficiently relevant SOP evidence was found. "
            "The system will not generate an action plan from weak or "
            "unsupported evidence."
        )

        st.info(
            "Try describing the incident using more specific operational "
            "terms or upload the applicable SOP."
        )

        st.stop()

    # --------------------------------------------------------
    # SAVE RETRIEVAL RESULTS
    # --------------------------------------------------------

    st.session_state.retrieved_chunks = retrieval_results

    st.session_state.retrieval_scores = [
        item["score"]
        for item in retrieval_results
    ]

    # --------------------------------------------------------
    # PREPARE GROUNDING CONTEXT
    # --------------------------------------------------------

    context_parts = []

    for number, item in enumerate(
        retrieval_results,
        start=1
    ):

        source = item["source"]

        context_parts.append(
            f"""
--- SOP EVIDENCE {number} ---
Source File: {source["file"]}
Chunk: {source["chunk"]}
Similarity Score: {item["score"]:.4f}

{item["text"]}
"""
        )

    context_chunks = "\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # DISPLAY RETRIEVAL
    # --------------------------------------------------------

    with st.expander(
        "🔍 View Retrieved SOP Evidence",
        expanded=False
    ):

        for number, item in enumerate(
            retrieval_results,
            start=1
        ):

            source = item["source"]

            st.markdown(
                f"""
                **Evidence {number}**

                **Source:** `{source["file"]}`  
                **Chunk:** `{source["chunk"]}`  
                **Similarity:** `{item["score"]:.4f}`
                """
            )

            st.write(
                item["text"]
            )

            st.divider()

    # --------------------------------------------------------
    # INITIALIZE GEMINI
    # --------------------------------------------------------

    with st.spinner(
        "🤖 Initializing CrewAI + Gemini 2.5 Flash..."
    ):

        shared_llm = get_agent_llm()

    if shared_llm is None:

        st.error(
            "Gemini could not be initialized. "
            "Check GEMINI_API_KEY configuration."
        )

        st.stop()

    # --------------------------------------------------------
    # CREATE AGENTS
    # --------------------------------------------------------

    with st.spinner(
        "🧩 Building multi-agent workflow..."
    ):

        triage_agent = create_triage_agent(
            shared_llm
        )

        compiler_agent = create_compiler_agent(
            shared_llm
        )

        router_agent = create_router_agent(
            shared_llm
        )

        automation_agent = create_automation_agent(
            shared_llm
        )

    # --------------------------------------------------------
    # CREATE TASKS
    # --------------------------------------------------------

    agent_tasks = define_workflow_tasks(
        triage_worker=triage_agent,
        compiler_worker=compiler_agent,
        router_worker=router_agent,
        automation_worker=automation_agent,
        incident_description=incident_description,
        context_chunks=context_chunks
    )

    # --------------------------------------------------------
    # CREATE CREW
    # --------------------------------------------------------

    operations_crew = Crew(
        agents=[
            triage_agent,
            compiler_agent,
            router_agent,
            automation_agent
        ],

        tasks=agent_tasks,

        process=Process.sequential,

        verbose=True
    )

    # --------------------------------------------------------
    # EXECUTE CREW
    # --------------------------------------------------------

    with st.spinner(
        "⚙️ Running multi-agent operational workflow..."
    ):

        try:

            crew_output = operations_crew.kickoff()

            st.session_state.crew_output = str(
                crew_output
            )

            st.session_state.approval_status = (
                "Pending Human Review"
            )

        except Exception as exc:

            st.error(
                "The multi-agent workflow failed."
            )

            st.exception(exc)

            st.stop()


# ============================================================
# EXECUTIVE ACTION BRIEF
# ============================================================

if st.session_state.crew_output:

    st.divider()

    st.subheader(
        "📋 4. Executive Action Brief"
    )

    st.markdown(
        """
        <div style="
            padding: 20px;
            border: 2px solid #444;
            border-radius: 12px;
            margin-bottom: 20px;
        ">
        <h3>AI-Generated Operational Action Map</h3>
        <p>
        The following brief has been prepared from the retrieved SOP
        evidence and multi-agent workflow. It requires human review
        before any operational release.
        </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state.crew_output
    )

    # ========================================================
    # VERIFIED SOURCES
    # ========================================================

    if st.session_state.retrieved_chunks:

        st.subheader(
            "📚 Verified SOP References"
        )

        for number, item in enumerate(
            st.session_state.retrieved_chunks,
            start=1
        ):

            source = item["source"]

            st.markdown(
                f"""
                **Reference {number}:**
                `{source["file"]}` —
                Chunk `{source["chunk"]}` —
                Similarity `{item["score"]:.4f}`
                """
            )

    # ========================================================
    # HUMAN-IN-THE-LOOP
    # ========================================================

    st.divider()

    st.subheader(
        "👤 5. Mandatory Human-in-the-Loop Approval"
    )

    st.warning(
        "The AI has prepared a recommended workflow. "
        "No workflow should be considered approved or released "
        "until an authorized human reviews it."
    )

    human_verified = st.checkbox(
        "I have reviewed the AI-generated action plan and the referenced SOP evidence."
    )

    approval_col1, approval_col2 = st.columns(2)

    with approval_col1:

        if st.button(
            "✅ Approve & Release Workflow",
            disabled=not human_verified,
            use_container_width=True
        ):

            st.session_state.approval_status = (
                "Approved & Released by Human Reviewer"
            )

            st.success(
                "Workflow approved by the human reviewer."
            )

    with approval_col2:

        if st.button(
            "❌ Reject / Request Review",
            use_container_width=True
        ):

            st.session_state.approval_status = (
                "Rejected / Review Required"
            )

            st.error(
                "Workflow rejected or sent for further human review."
            )

    # ========================================================
    # STATUS
    # ========================================================

    st.markdown(
        f"""
        ### Current Workflow Status

        **{st.session_state.approval_status}**
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SOP-Orchestrator AI | AI-Catalysts | "
    "CrewAI + Gemini 2.5 Flash + Local TF-IDF RAG | "
    "Human-Governed Operational Decision Support"
)

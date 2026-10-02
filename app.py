```python
import streamlit as st
import html

from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from crewai import Crew, Process

# ------------------------------------------------------------
# Local project imports
# ------------------------------------------------------------
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
    layout="wide"
)


# ============================================================
# PROFESSIONAL UI THEME
# ============================================================

st.markdown(
    """
    <style>

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }

    .header-banner {
        background: linear-gradient(
            135deg,
            #1E3A8A 0%,
            #0F172A 100%
        );

        padding: 26px;
        border-radius: 12px;
        margin-bottom: 28px;

        box-shadow:
            0 4px 15px rgba(0,0,0,0.20);
    }

    .header-banner h1 {
        color: #FFFFFF !important;
        font-weight: 800;
        font-size: 26px !important;
        margin: 0;
        text-align: center;
    }

    .header-banner p {
        color: #93C5FD !important;
        margin: 5px 0 12px 0;
        font-size: 14px;
        text-align: center;
    }

    .roster-grid {
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        padding: 10px;
        font-size: 13px;
        color: #F3F4F6;
        border-left: 4px solid #3B82F6;
        text-align: center;
    }

    .premium-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-left: 6px solid #2563EB;
        border-radius: 8px;
        padding: 22px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        color: #000000 !important;
    }

    .agent-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        color: #0F172A;
    }

    .status-card {
        background-color: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        color: #0F172A;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="header-banner">

        <h1>
            🤖 SOP-Orchestrator AI
        </h1>

        <p>
            Multi-Agent Supply Chain Contingency Planner |
            Powered by CrewAI + Gemini 2.5 Flash + Local RAG
        </p>

        <div class="roster-grid">

            👑 <b>Project Leader:</b>
            Hafiz Masood Ur Rehman

            &nbsp; | &nbsp;

            👥 <b>Team:</b>
            Fatima Ijaz • Muhammad Aslam • Shakeel Ahmed •
            Sami Ur Rahman • Muhammad Haroon Jan

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "sources" not in st.session_state:
    st.session_state.sources = []

if "vectorizer" not in st.session_state:
    st.session_state.vectorizer = None

if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

if "last_output" not in st.session_state:
    st.session_state.last_output = None

if "last_incident" not in st.session_state:
    st.session_state.last_incident = None

if "matched_citations" not in st.session_state:
    st.session_state.matched_citations = []

if "approval_status" not in st.session_state:
    st.session_state.approval_status = "Not Executed"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("### ⚙️ Core Infrastructure")

st.sidebar.success("⚡ CrewAI Multi-Agent Framework")

st.sidebar.info("📌 Local TF-IDF RAG Matrix")

st.sidebar.info("🧠 Gemini 2.5 Flash")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🤖 Agent Workflow")

st.sidebar.caption("1️⃣ Incident Triage Agent")
st.sidebar.caption("2️⃣ SOP Compliance Agent")
st.sidebar.caption("3️⃣ Department Routing Agent")
st.sidebar.caption("4️⃣ Executive Action Brief Agent")

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔐 Governance")

st.sidebar.warning(
    "AI recommendations require human review and approval "
    "before operational release."
)


# ============================================================
# MAIN TWO-PANEL LAYOUT
# ============================================================

col1, col2 = st.columns(2, gap="large")


# ============================================================
# LEFT PANEL — DOCUMENT INGESTION
# ============================================================

with col1:

    st.header("📄 SOP Ingestion Panel")

    uploaded_files = st.file_uploader(
        "Upload approved operational SOP manuals (PDF):",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_documents = st.button(
        "⚙️ Process & Index Documents",
        use_container_width=True
    )

    if process_documents:

        if not uploaded_files:

            st.warning(
                "Please upload at least one SOP PDF before indexing."
            )

        else:

            all_chunks = []
            all_sources = []

            with st.spinner(
                "Extracting SOP text and building local retrieval matrix..."
            ):

                for uploaded_file in uploaded_files:

                    try:

                        reader = PdfReader(uploaded_file)

                        file_text = ""

                        for page_number, page in enumerate(reader.pages):

                            page_text = page.extract_text()

                            if page_text:
                                file_text += page_text + "\n"

                        if not file_text.strip():

                            st.warning(
                                f"No readable text found in "
                                f"{uploaded_file.name}."
                            )

                            continue

                        # ------------------------------------------------
                        # 500-word chunking
                        # ------------------------------------------------

                        words = file_text.split()

                        chunk_size = 500

                        chunks = [
                            " ".join(
                                words[index:index + chunk_size]
                            )
                            for index in range(
                                0,
                                len(words),
                                chunk_size
                            )
                        ]

                        for index, chunk in enumerate(chunks):

                            if chunk.strip():

                                all_chunks.append(chunk)

                                all_sources.append(
                                    f"{uploaded_file.name} "
                                    f"(Segment {index + 1})"
                                )

                    except Exception as exc:

                        st.error(
                            f"Error processing "
                            f"{uploaded_file.name}: {exc}"
                        )

            # ------------------------------------------------------------
            # Build TF-IDF index
            # ------------------------------------------------------------

            if all_chunks:

                try:

                    vectorizer = TfidfVectorizer(
                        stop_words="english"
                    )

                    tfidf_matrix = vectorizer.fit_transform(
                        all_chunks
                    )

                    st.session_state.chunks = all_chunks

                    st.session_state.sources = all_sources

                    st.session_state.vectorizer = vectorizer

                    st.session_state.tfidf_matrix = tfidf_matrix

                    st.session_state.last_output = None

                    st.session_state.last_incident = None

                    st.session_state.matched_citations = []

                    st.session_state.approval_status = (
                        "Not Executed"
                    )

                    st.success(
                        f"Successfully indexed "
                        f"{len(uploaded_files)} document(s) "
                        f"into {len(all_chunks)} knowledge segments."
                    )

                    st.rerun()

                except Exception as exc:

                    st.error(
                        f"Index construction failed: {exc}"
                    )

            else:

                st.warning(
                    "No readable SOP content was available "
                    "for indexing."
                )


# ============================================================
# RIGHT PANEL — INCIDENT CONTROLLER
# ============================================================

with col2:

    st.header("💬 Agent Action Controller")

    incident_description = st.text_area(
        "Describe the operational incident:",
        placeholder=(
            "Example: "
            "We received a shipment of damaged material from "
            "a supplier. Several cartons are visibly damaged "
            "and some material may be unusable. What should we do?"
        ),
        height=160
    )

    trigger_workflow = st.button(
        "🚀 Trigger Multi-Agent Workflow",
        use_container_width=True
    )

    # ------------------------------------------------------------
    # Current RAG status
    # ------------------------------------------------------------

    if st.session_state.tfidf_matrix is not None:

        st.success(
            f"📂 RAG Status: "
            f"{len(st.session_state.chunks)} SOP segments indexed."
        )

    else:

        st.warning(
            "📥 RAG Status: No SOP knowledge base indexed."
        )


# ============================================================
# WORKFLOW EXECUTION
# ============================================================

if trigger_workflow:

    # ------------------------------------------------------------
    # Validate incident
    # ------------------------------------------------------------

    if not incident_description.strip():

        st.error(
            "Please enter an operational incident "
            "before starting the workflow."
        )

        st.stop()

    # ------------------------------------------------------------
    # Validate knowledge base
    # ------------------------------------------------------------

    if st.session_state.tfidf_matrix is None:

        st.error(
            "Please process at least one SOP document "
            "before triggering the workflow."
        )

        st.stop()

    # ------------------------------------------------------------
    # RAG RETRIEVAL
    # ------------------------------------------------------------

    with st.spinner(
        "Searching approved SOP knowledge base..."
    ):

        try:

            query_vector = (
                st.session_state.vectorizer.transform(
                    [incident_description]
                )
            )

            similarities = cosine_similarity(
                query_vector,
                st.session_state.tfidf_matrix
            ).flatten()

            # --------------------------------------------------------
            # Get top 3 matches
            # --------------------------------------------------------

            top_indices = np.argsort(
                similarities
            )[-3:][::-1]

            context_parts = []

            matched_citations = []

            for index in top_indices:

                score = float(similarities[index])

                # Minimum grounding threshold
                if score >= 0.05:

                    source = st.session_state.sources[index]

                    chunk = st.session_state.chunks[index]

                    context_parts.append(
                        f"Source Segment: {source}\n"
                        f"Similarity Score: {score:.4f}\n"
                        f"Reference Content:\n{chunk}\n"
                    )

                    if source not in matched_citations:

                        matched_citations.append(source)

            context_str = "\n\n".join(context_parts)

        except Exception as exc:

            st.error(
                f"RAG retrieval failed: {exc}"
            )

            st.stop()

    # ------------------------------------------------------------
    # Anti-hallucination gate
    # ------------------------------------------------------------

    if not context_str.strip():

        st.error(
            "🛑 Grounding Gate Activated"
        )

        st.warning(
            "No sufficiently relevant SOP reference was found. "
            "Multi-agent execution has been halted."
        )

        st.info(
            "Human review is required because the available "
            "knowledge base does not provide sufficient evidence."
        )

        st.session_state.approval_status = (
            "Halted — Insufficient SOP Evidence"
        )

        st.stop()

    # ------------------------------------------------------------
    # Grounding confirmation
    # ------------------------------------------------------------

    st.success(
        f"✅ Grounding reference matrix extracted: "
        f"{len(matched_citations)} supporting SOP segment(s)."
    )

    with st.expander(
        "🔎 View Retrieved SOP Evidence",
        expanded=False
    ):

        st.text(context_str)

    # ------------------------------------------------------------
    # Initialize Gemini/CrewAI
    # ------------------------------------------------------------

    with st.spinner(
        "Initializing Gemini-powered multi-agent workflow..."
    ):

        try:

            shared_llm = get_agent_llm()

        except Exception as exc:

            st.error(
                f"LLM initialization failed: {exc}"
            )

            st.stop()

    if shared_llm is None:

        st.error(
            "Gemini LLM could not be initialized. "
            "Check GEMINI_API_KEY configuration."
        )

        st.stop()

    # ------------------------------------------------------------
    # Create agents
    # ------------------------------------------------------------

    try:

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

    except Exception as exc:

        st.error(
            "CrewAI agent initialization failed."
        )

        st.exception(exc)

        st.stop()

    # ------------------------------------------------------------
    # Create tasks
    # ------------------------------------------------------------

    try:

        agent_tasks = define_workflow_tasks(
            triage_agent,
            compiler_agent,
            router_agent,
            automation_agent,
            incident_description,
            context_str
        )

    except Exception as exc:

        st.error(
            f"Workflow task construction failed: {exc}"
        )

        st.exception(exc)

        st.stop()

    # ------------------------------------------------------------
    # Create Crew
    # ------------------------------------------------------------

    try:

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

    except Exception as exc:

        st.error(
            f"Crew initialization failed: {exc}"
        )

        st.exception(exc)

        st.stop()

    # ------------------------------------------------------------
    # Execute Crew
    # ------------------------------------------------------------

    st.info(
        "🤖 Multi-Agent workflow is now executing sequentially."
    )

    try:

        with st.spinner(
            "Agents are analyzing, compiling, routing, "
            "and preparing the action brief..."
        ):

            crew_output = operations_crew.kickoff()

    except Exception as exc:

        st.error(
            "❌ Multi-Agent workflow execution failed."
        )

        st.exception(exc)

        st.session_state.approval_status = (
            "Execution Failed"
        )

        st.stop()

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    st.session_state.last_output = str(
        crew_output
    )

    st.session_state.last_incident = (
        incident_description
    )

    st.session_state.matched_citations = (
        matched_citations
    )

    st.session_state.approval_status = (
        "Pending Human Approval"
    )


# ============================================================
# OUTPUT SECTION
# ============================================================

if st.session_state.last_output:

    st.markdown("---")

    st.header("📋 Executive Action Map")

    st.markdown(
        '<div class="premium-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state.last_output
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Grounding citations
    # --------------------------------------------------------

    st.markdown(
        "### 📌 Verified SOP References"
    )

    if st.session_state.matched_citations:

        for source in st.session_state.matched_citations:

            st.caption(
                f"✔ {source}"
            )

    else:

        st.caption(
            "No citation references available."
        )

    # --------------------------------------------------------
    # Human Approval Gate
    # --------------------------------------------------------

    st.markdown("---")

    st.header(
        "🔐 Mandatory Human-in-the-Loop Approval"
    )

    st.warning(
        "The AI-generated action plan is advisory. "
        "No operational workflow should be released "
        "without authorized human review."
    )

    st.write(
        f"**Current Status:** "
        f"{st.session_state.approval_status}"
    )

    approval_confirmation = st.checkbox(
        "I have reviewed the AI-generated action plan "
        "and the supporting SOP references."
    )

    approval_col1, approval_col2 = st.columns(2)

    with approval_col1:

        approve_workflow = st.button(
            "✅ Approve & Release Workflow",
            use_container_width=True
        )

    with approval_col2:

        reject_workflow = st.button(
            "❌ Reject / Request Review",
            use_container_width=True
        )

    if approve_workflow:

        if not approval_confirmation:

            st.error(
                "Please confirm that you have reviewed "
                "the AI-generated action plan before approval."
            )

        else:

            st.session_state.approval_status = (
                "Approved by Human Reviewer"
            )

            st.success(
                "✅ Workflow approved by human reviewer."
            )

            st.info(
                "In the production architecture, this approval "
                "would release the downstream business workflow."
            )

    if reject_workflow:

        st.session_state.approval_status = (
            "Rejected — Human Review Required"
        )

        st.warning(
            "Workflow rejected. Human review is required "
            "before any operational action."
        )
```

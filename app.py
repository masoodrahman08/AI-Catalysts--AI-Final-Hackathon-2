```python
import os
import streamlit as st
import numpy as np

from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from crewai import Crew, Process

from agents import (
    get_agent_llm,
    create_triage_agent,
    create_compiler_agent,
    create_router_agent,
    create_automation_agent
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
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "chunks": [],
    "sources": [],
    "vectorizer": None,
    "tfidf_matrix": None,
    "last_output": None,
    "last_incident": "",
    "matched_citations": [],
    "approval_status": "Pending Human Review"
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HEADER
# ============================================================

st.title("🤖 SOP-Orchestrator AI")

st.subheader(
    "Multi-Agent Supply Chain Contingency Planner"
)

st.caption(
    "Powered by CrewAI + Gemini 2.5 Flash + Local TF-IDF RAG"
)

st.markdown(
    """
**Project Team: AI-Catalysts**

👑 **Project Leader:** Masood Ur Rehman

👥 **Engineers:** Fatima Ijaz • Muhammad Aslam • Shakeel Ahmed • Sami Ur Rahman • Muhammad Haroon Jan
"""
)


# ============================================================
# CORE INFRASTRUCTURE STATUS
# ============================================================

st.markdown("## ⚙️ Core Infrastructure")

col1, col2 = st.columns(2)

with col1:
    st.success("Swarm Framework: CrewAI Live")

with col2:
    if st.session_state.vectorizer is not None:
        st.success("Local Matrix Index: ACTIVE")
    else:
        st.info("Local Matrix Index: Waiting for SOP")


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🧠 Agent Architecture")

    st.markdown(
        """
### Agent 1
**Operational Incident Triage**

Identifies the incident and extracts important operational terms.

### Agent 2
**SOP Compliance Compiler**

Converts retrieved SOP evidence into an actionable checklist.

### Agent 3
**Department Routing**

Maps actions to responsible functions.

### Agent 4
**Executive Action Brief**

Prepares the final controlled action plan.

---

### 🔐 Governance

**Human-in-the-Loop**

AI recommendations are not automatically approved or executed.

An authorized human must review the action plan and approve release.
"""
    )


# ============================================================
# SECTION 1 — SOP INGESTION
# ============================================================

st.markdown("## 📄 Ingestion Panel")

uploaded_files = st.file_uploader(
    "Upload one or more approved SOP PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:

    all_chunks = []
    all_sources = []

    for uploaded_file in uploaded_files:

        try:
            reader = PdfReader(uploaded_file)

            file_text = ""

            for page in reader.pages:
                page_text = page.extract_text() or ""
                file_text += page_text + "\n"

            file_text = file_text.strip()

            if not file_text:
                st.warning(
                    f"Could not extract text from: {uploaded_file.name}"
                )
                continue

            words = file_text.split()

            # ------------------------------------------------
            # Local chunking
            # ------------------------------------------------

            chunk_size = 500

            file_chunks = [
                " ".join(words[i:i + chunk_size])
                for i in range(
                    0,
                    len(words),
                    chunk_size
                )
            ]

            for index, chunk in enumerate(file_chunks, start=1):

                all_chunks.append(chunk)

                all_sources.append(
                    f"{uploaded_file.name} | Segment {index}"
                )

            size_kb = uploaded_file.size / 1024

            st.info(
                f"📄 {uploaded_file.name} — "
                f"{size_kb:.1f} KB — "
                f"{len(file_chunks)} matrix chunks"
            )

        except Exception as exc:

            st.error(
                f"Failed to process {uploaded_file.name}"
            )

            st.exception(exc)


    # --------------------------------------------------------
    # Build TF-IDF matrix
    # --------------------------------------------------------

    if all_chunks:

        try:

            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(1, 2)
            )

            tfidf_matrix = vectorizer.fit_transform(
                all_chunks
            )

            st.session_state.chunks = all_chunks
            st.session_state.sources = all_sources
            st.session_state.vectorizer = vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix

            # Reset previous workflow
            st.session_state.last_output = None
            st.session_state.last_incident = ""
            st.session_state.matched_citations = []
            st.session_state.approval_status = (
                "Pending Human Review"
            )

            st.success(
                f"Grounding reference matrix extracted successfully. "
                f"{len(all_chunks)} chunks indexed."
            )

        except Exception as exc:

            st.error(
                "Failed to build the local TF-IDF matrix."
            )

            st.exception(exc)


# ============================================================
# CURRENT INDEX STATUS
# ============================================================

if st.session_state.chunks:

    st.caption(
        f"Deployed Framework Status: "
        f"{len(st.session_state.chunks)} matrix chunks "
        f"locked in session memory."
    )


# ============================================================
# SECTION 2 — INCIDENT INPUT
# ============================================================

st.markdown("## 💬 Agent Action Controller")

incident_description = st.text_area(
    "Describe the operational incident",
    height=160,
    placeholder=(
        "Example:\n"
        "Damaged material was received from a supplier. "
        "Several cartons were found damaged during warehouse "
        "receiving inspection."
    )
)


# ============================================================
# RUN WORKFLOW BUTTON
# ============================================================

run_workflow = st.button(
    "🚀 Analyze Incident & Build Action Plan",
    type="primary",
    use_container_width=True
)


# ============================================================
# VALIDATION
# ============================================================

if run_workflow:

    if not st.session_state.chunks:

        st.warning(
            "Please upload and index at least one approved SOP PDF first."
        )

        st.stop()

    if not incident_description.strip():

        st.warning(
            "Please enter an operational incident description."
        )

        st.stop()


    # ========================================================
    # LOCAL RAG RETRIEVAL
    # ========================================================

    st.markdown("### 🔎 Local SOP Retrieval")

    try:

        query_vec = (
            st.session_state.vectorizer.transform(
                [incident_description]
            )
        )

        similarities = cosine_similarity(
            query_vec,
            st.session_state.tfidf_matrix
        ).flatten()

        top_count = min(
            5,
            len(similarities)
        )

        top_indices = np.argsort(
            similarities
        )[-top_count:][::-1]


        # ----------------------------------------------------
        # Stronger grounding threshold
        # ----------------------------------------------------

        MIN_SIMILARITY = 0.05

        context_parts = []
        matched_citations = []

        for idx in top_indices:

            score = float(similarities[idx])

            if score >= MIN_SIMILARITY:

                source_name = (
                    st.session_state.sources[idx]
                )

                chunk_text = (
                    st.session_state.chunks[idx]
                )

                context_parts.append(
                    f"""
SOURCE: {source_name}
SIMILARITY SCORE: {score:.4f}

SOP CONTENT:
{chunk_text}
"""
                )

                matched_citations.append(
                    {
                        "source": source_name,
                        "score": score
                    }
                )


        # ----------------------------------------------------
        # Grounding gate
        # ----------------------------------------------------

        if not context_parts:

            st.error(
                "No sufficiently relevant SOP evidence was found "
                "for this incident."
            )

            st.info(
                "The AI workflow has been stopped to prevent "
                "ungrounded recommendations."
            )

            st.stop()


        context_str = "\n\n".join(
            context_parts
        )

        st.session_state.matched_citations = (
            matched_citations
        )

        # ----------------------------------------------------
        # Display retrieved references
        # ----------------------------------------------------

        with st.expander(
            "📚 View Retrieved SOP Evidence",
            expanded=False
        ):

            for citation in matched_citations:

                st.write(
                    f"**{citation['source']}**  "
                    f"(similarity: {citation['score']:.4f})"
                )


        # ====================================================
        # INITIALIZE CREWAI GEMINI
        # ====================================================

        st.markdown("### 🧠 Initializing Multi-Agent Workflow")

        shared_llm = get_agent_llm()

        if shared_llm is None:
            st.stop()


        # ====================================================
        # CREATE AGENTS
        # ====================================================

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


        # ====================================================
        # CREATE TASKS
        # ====================================================

        agent_tasks = define_workflow_tasks(
            triage_worker=triage_agent,
            compiler_worker=compiler_agent,
            router_worker=router_agent,
            automation_worker=automation_agent,
            incident_description=incident_description,
            context_chunks=context_str
        )


        # ====================================================
        # CREATE CREW
        # ====================================================

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


        # ====================================================
        # EXECUTE CREW
        # ====================================================

        st.info(
            "Grounding evidence verified. "
            "Initializing Multi-Agent task loop..."
        )

        crew_output = operations_crew.kickoff()


        # ====================================================
        # SAVE RESULT
        # ====================================================

        st.session_state.last_output = str(
            crew_output
        )

        st.session_state.last_incident = (
            incident_description
        )

        st.session_state.approval_status = (
            "Pending Human Review"
        )

        st.success(
            "Multi-Agent workflow completed successfully."
        )


    except Exception as exc:

        st.error(
            "The Multi-Agent workflow failed."
        )

        st.exception(exc)

        st.stop()


# ============================================================
# SECTION 3 — EXECUTIVE ACTION BRIEF
# ============================================================

if st.session_state.last_output:

    st.markdown("---")

    st.markdown(
        "## 📋 Executive Action Brief"
    )

    st.markdown(
        st.session_state.last_output
    )


    # ========================================================
    # SOURCE REFERENCES
    # ========================================================

    if st.session_state.matched_citations:

        st.markdown(
            "### 📚 Verified SOP References"
        )

        for citation in (
            st.session_state.matched_citations
        ):

            st.write(
                f"- **{citation['source']}** "
                f"| Similarity: "
                f"{citation['score']:.4f}"
            )


    # ========================================================
    # HUMAN APPROVAL GATE
    # ========================================================

    st.markdown("---")

    st.markdown(
        "## 🔐 Mandatory Human-in-the-Loop Approval"
    )

    st.warning(
        "The AI-generated action plan is advisory. "
        "No workflow should be released or executed until "
        "an authorized human reviews and approves it."
    )


    approval_check = st.checkbox(
        "I have reviewed the AI-generated action plan and "
        "the supporting SOP references."
    )


    col1, col2 = st.columns(2)


    with col1:

        if st.button(
            "✅ Approve & Release Workflow",
            disabled=not approval_check,
            use_container_width=True
        ):

            st.session_state.approval_status = (
                "APPROVED — Released by Human Reviewer"
            )

            st.success(
                "Workflow approved by the human reviewer."
            )


    with col2:

        if st.button(
            "❌ Reject / Request Review",
            use_container_width=True
        ):

            st.session_state.approval_status = (
                "REJECTED — Human Review Required"
            )

            st.error(
                "Workflow rejected. Human review is required."
            )


    # ========================================================
    # CURRENT APPROVAL STATUS
    # ========================================================

    st.markdown("### Current Governance Status")

    status = st.session_state.approval_status

    if status.startswith("APPROVED"):

        st.success(status)

    elif status.startswith("REJECTED"):

        st.error(status)

    else:

        st.info(status)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI-Catalysts | SOP-Orchestrator AI | "
    "Human-Governed Multi-Agent Operations"
)
```

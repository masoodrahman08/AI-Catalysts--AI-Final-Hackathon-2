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

        similarities = cosine_similarity(
            query_vector,
            matrix
        )[0]

    except Exception:
        return []

    ranked_indices = np.argsort(
        similarities
    )[::-1]

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


# ============================================================
# SECTION 3 — RETRIEVAL
# ============================================================

if st.button(
    "🔎 Retrieve Relevant SOP Evidence"
):

    if not st.session_state.sop_chunks:

        st.warning(
            "Please upload and build the SOP knowledge base first."
        )

    elif not incident.strip():

        st.warning(
            "Please describe the operational incident first."
        )

    else:

        with st.spinner(
            "Searching the local SOP knowledge base..."
        ):

            results = retrieve_sop_evidence(
                incident,
                st.session_state.sop_chunks,
                st.session_state.sop_sources,
                TOP_K,
                SIMILARITY_THRESHOLD
            )

            st.session_state.retrieved_context = results

            st.session_state.workflow_output = None
            st.session_state.workflow_approved = False
            st.session_state.workflow_rejected = False


# ============================================================
# DISPLAY RETRIEVED EVIDENCE
# ============================================================

if st.session_state.retrieved_context:

    st.subheader("📖 Retrieved SOP Evidence")

    for index, item in enumerate(
        st.session_state.retrieved_context,
        start=1
    ):

        source = item["source"]

        with st.expander(
            f"Evidence {index} — "
            f"{source['file']} / "
            f"Chunk {source['chunk']} / "
            f"Similarity {item['score']:.3f}"
        ):

            st.write(item["chunk"])

elif (
    incident.strip()
    and st.session_state.sop_chunks
):

    if st.session_state.retrieved_context == []:

        st.info(
            "No sufficiently relevant SOP evidence has been retrieved."
        )


# ============================================================
# BUILD CONTEXT FOR CREWAI
# ============================================================

def format_retrieved_context(results):

    if not results:
        return ""

    sections = []

    for index, result in enumerate(
        results,
        start=1
    ):

        source = result["source"]

        sections.append(
            (
                f"--- EVIDENCE {index} ---\n"
                f"Source: {source['file']}\n"
                f"Chunk: {source['chunk']}\n"
                f"Similarity: {result['score']:.4f}\n\n"
                f"{result['chunk']}"
            )
        )

    return "\n\n".join(sections)


# ============================================================
# SECTION 4 — RUN MULTI-AGENT WORKFLOW
# ============================================================

st.header("3. 🤖 Run Multi-Agent SOP Workflow")


def is_quota_error(exception_text):

    text = exception_text.lower()

    quota_terms = [
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "too many requests",
        "generate_content_free_tier"
    ]

    return any(
        term in text
        for term in quota_terms
    )


if st.button(
    "🚀 Generate SOP-to-Action Workflow",
    type="primary"
):

    if not st.session_state.sop_chunks:

        st.warning(
            "Please upload and build the SOP knowledge base first."
        )

    elif not incident.strip():

        st.warning(
            "Please describe the operational incident first."
        )

    elif not st.session_state.retrieved_context:

        st.warning(
            "Please retrieve SOP evidence before running "
            "the multi-agent workflow."
        )

    else:

        shared_llm = get_agent_llm()

        if shared_llm is None:

            st.stop()

        context_text = format_retrieved_context(
            st.session_state.retrieved_context
        )

        try:

            with st.status(
                "Running four-agent workflow...",
                expanded=True
            ) as workflow_status:

                st.write(
                    "Agent 1 — Operational Incident Triage"
                )

                triage_agent = create_triage_agent(
                    shared_llm
                )

                st.write(
                    "Agent 2 — SOP Compliance and Action Compiler"
                )

                compiler_agent = create_compiler_agent(
                    shared_llm
                )

                st.write(
                    "Agent 3 — Cross-Functional Department Routing"
                )

                router_agent = create_router_agent(
                    shared_llm
                )

                st.write(
                    "Agent 4 — Executive Action Brief"
                )

                automation_agent = create_automation_agent(
                    shared_llm
                )

                agent_tasks = define_workflow_tasks(
                    triage_agent,
                    compiler_agent,
                    router_agent,
                    automation_agent,
                    incident,
                    context_text
                )

                operations_crew = Crew(

                    agents=[
                        triage_agent,
                        compiler_agent,
                        router_agent,
                        automation_agent
                    ],

                    tasks=agent_tasks,

                    process=Process.sequential,

                    verbose=False,

                    cache=True,

                    # Keep total workflow request rate
                    # conservative for the free tier.
                    max_rpm=4
                )

                st.write(
                    "Executing sequential CrewAI workflow..."
                )

                crew_output = operations_crew.kickoff()

                st.session_state.workflow_output = str(
                    crew_output
                )

                st.session_state.workflow_approved = False
                st.session_state.workflow_rejected = False

                workflow_status.update(
                    label="Workflow generated successfully",
                    state="complete"
                )

        except Exception as exc:

            error_text = str(exc)

            if is_quota_error(error_text):

                st.error(
                    "### Gemini Free-Tier Rate Limit Reached"
                )

                st.warning(
                    "The workflow did not complete because the "
                    "Gemini free-tier request limit was reached. "
                    "No paid API service is being used."
                )

                st.info(
                    "Please wait for the Gemini request window "
                    "to reset and then run the workflow again. "
                    "Avoid running the separate Gemini test at "
                    "the same time because it also consumes a "
                    "free-tier request."
                )

            else:

                st.error(
                    "The multi-agent workflow encountered an error."
                )

                with st.expander(
                    "Technical error details"
                ):

                    st.code(
                        error_text
                    )


# ============================================================
# SECTION 5 — EXECUTIVE ACTION BRIEF
# ============================================================

if st.session_state.workflow_output:

    st.header("4. 📋 Executive Action Brief")

    st.markdown(
        '<div class="workflow-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state.workflow_output
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# SECTION 6 — SOURCE EVIDENCE
# ============================================================

if st.session_state.retrieved_context:

    st.header("5. 📚 Evidence Traceability")

    st.caption(
        "The following evidence was supplied to the SOP "
        "Compliance Agent before the workflow was generated."
    )

    for index, item in enumerate(
        st.session_state.retrieved_context,
        start=1
    ):

        source = item["source"]

        st.write(
            f"**Evidence {index}:** "
            f"{source['file']} — "
            f"Chunk {source['chunk']} — "
            f"Similarity {item['score']:.3f}"
        )


# ============================================================
# SECTION 7 — HUMAN APPROVAL
# ============================================================

if st.session_state.workflow_output:

    st.header("6. 👤 Human Approval Gate")

    st.markdown(
        '<div class="approval-card">',
        unsafe_allow_html=True
    )

    st.subheader(
        "MANDATORY HUMAN-IN-THE-LOOP APPROVAL"
    )

    st.write(
        "The AI-generated workflow is a recommendation. "
        "A responsible human must review the incident, "
        "SOP evidence, actions and responsibilities before "
        "the workflow can be considered approved."
    )

    st.markdown(
        """
        **AI does not:**
        - Execute procurement transactions
        - Release inventory
        - Contact suppliers
        - Approve financial claims
        - Change ERP records
        - Close the incident
        """,
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )

    approval_confirmation = st.checkbox(
        "I have reviewed the generated workflow and the supporting SOP evidence."
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "✅ Approve Workflow",
            disabled=not approval_confirmation,
            use_container_width=True
        ):

            st.session_state.workflow_approved = True
            st.session_state.workflow_rejected = False

    with col2:

        if st.button(
            "❌ Reject / Request Review",
            use_container_width=True
        ):

            st.session_state.workflow_approved = False
            st.session_state.workflow_rejected = True


# ============================================================
# APPROVAL STATUS
# ============================================================

if st.session_state.workflow_approved:

    st.success(
        "✅ Workflow approved by human reviewer."
    )

    st.info(
        "The approval represents human review of the "
        "AI-generated recommendation. No external business "
        "transaction has been executed by the application."
    )


elif st.session_state.workflow_rejected:

    st.warning(
        "⚠️ Workflow rejected / returned for human review."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "SOP-Orchestrator AI | AI-Catalysts | "
    "Local RAG + CrewAI + Gemini 3.8 Flash | "
    "Human-Governed Workflow"
)


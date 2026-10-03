import time

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
    initial_sidebar_state="expanded",
)


APP_TITLE = "SOP-Orchestrator AI"

MODEL_DISPLAY_NAME = "Gemini 3.8 Flash"

CHUNK_SIZE = 500

TOP_K = 3

SIMILARITY_THRESHOLD = 0.05

WORKFLOW_COOLDOWN_SECONDS = 20


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "sop_chunks": [],
    "sop_sources": [],
    "retrieved_context": [],
    "incident": "",
    "workflow_output": None,
    "workflow_approved": False,
    "workflow_rejected": False,
    "last_workflow_start": 0.0,
}


for key, default_value in DEFAULT_STATE.items():

    if key not in st.session_state:
        st.session_state[key] = default_value


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.35rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        font-size: 1.15rem;
        margin-bottom: 1rem;
    }

    .workflow-card {
        border: 1px solid #d9d9d9;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        background-color: #ffffff;
    }

    .approval-card {
        border: 2px solid #888888;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        background-color: #fafafa;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 SOP-Orchestrator AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-subtitle">'
    'Multi-Agent Supply Chain Contingency Planner'
    '</div>',
    unsafe_allow_html=True,
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
# SIDEBAR
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

    for number, step in enumerate(
        workflow_steps,
        start=1
    ):

        st.write(
            f"**{number}.** {step}"
        )

    st.divider()

    st.caption(
        "The AI prepares a grounded workflow for human review. "
        "It does not independently execute business transactions."
    )

    st.divider()

    st.markdown("### 💰 Cost Control")

    st.caption(
        "PDF extraction and TF-IDF retrieval run locally "
        "and require no API calls."
    )

    st.caption(
        "Only the four CrewAI agents use Gemini."
    )

    st.caption(
        "Agents are configured for one-pass execution, "
        "no delegation and no automatic agent retries."
    )

    st.caption(
        "A short cooldown prevents accidental repeated "
        "free-tier requests."
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
        "PDF content is processed locally and converted into "
        "a searchable TF-IDF knowledge base."
    ),
)


def split_text_into_chunks(
    text,
    chunk_size=500
):

    words = text.split()

    chunks = []

    for start in range(
        0,
        len(words),
        chunk_size
    ):

        chunk = " ".join(
            words[
                start:start + chunk_size
            ]
        ).strip()

        if chunk:
            chunks.append(chunk)

    return chunks


def build_local_knowledge_base(files):

    all_chunks = []

    all_sources = []

    for uploaded_file in files:

        try:

            reader = PdfReader(
                uploaded_file
            )

            page_texts = []

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                try:
                    page_text = (
                        page.extract_text()
                        or ""
                    )

                except Exception:
                    page_text = ""

                if page_text.strip():

                    page_texts.append(
                        f"[Page {page_number}]\n"
                        f"{page_text.strip()}"
                    )

            combined_text = "\n\n".join(
                page_texts
            )

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
                f"Could not read "
                f"{uploaded_file.name}: {exc}"
            )

    return (
        all_chunks,
        all_sources
    )


if uploaded_files:

    if st.button(
        "🔄 Build Local SOP Knowledge Base",
        type="primary"
    ):

        with st.spinner(
            "Extracting SOP text and building local TF-IDF index..."
        ):

            chunks, sources = (
                build_local_knowledge_base(
                    uploaded_files
                )
            )

            st.session_state.sop_chunks = chunks

            st.session_state.sop_sources = sources

            st.session_state.retrieved_context = []

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
                "No readable text was extracted "
                "from the uploaded SOP files."
            )


if st.session_state.sop_chunks:

    st.success(
        f"✅ Local SOP knowledge base active — "
        f"{len(st.session_state.sop_chunks)} searchable chunk(s)"
    )


# ============================================================
# SECTION 2 — INCIDENT INPUT
# ============================================================

st.header("2. 🚨 Describe Operational Incident")

incident = st.text_area(
    "Describe the operational situation",
    value=st.session_state.incident,
    height=150,
    placeholder=(
        "Example:\n"
        "During manual inspection of an incoming marine cargo "
        "container, the high-security bolt seal shows physical "
        "deformation and un-logged tracking marks. "
        "What are the applicable procedures?"
    ),
)

st.session_state.incident = incident


# ============================================================
# LOCAL TF-IDF RAG
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
        ngram_range=(1, 2),
    )

    try:

        matrix = vectorizer.fit_transform(
            chunks
        )

        query_vector = vectorizer.transform(
            [query]
        )

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

        score = float(
            similarities[index]
        )

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


def evidence_label(score):

    if score >= 0.40:
        return "🟢 Strong evidence"

    if score >= 0.20:
        return "🟡 Relevant evidence"

    return (
        "🟠 Weak evidence — "
        "human verification recommended"
    )


# ============================================================
# RETRIEVE BUTTON
# ============================================================

if st.button(
    "🔎 Retrieve Relevant SOP Evidence"
):

    if not st.session_state.sop_chunks:

        st.warning(
            "Please upload and build the SOP "
            "knowledge base first."
        )

    elif not incident.strip():

        st.warning(
            "Please describe the operational "
            "incident first."
        )

    else:

        with st.spinner(
            "Searching the local SOP knowledge base..."
        ):

            st.session_state.retrieved_context = (
                retrieve_sop_evidence(
                    incident,
                    st.session_state.sop_chunks,
                    st.session_state.sop_sources,
                    TOP_K,
                    SIMILARITY_THRESHOLD,
                )
            )

        st.session_state.workflow_output = None

        st.session_state.workflow_approved = False

        st.session_state.workflow_rejected = False


# ============================================================
# DISPLAY EVIDENCE
# ============================================================

if st.session_state.retrieved_context:

    st.subheader(
        "📖 Retrieved SOP Evidence"
    )

    for index, item in enumerate(
        st.session_state.retrieved_context,
        start=1
    ):

        source = item["source"]

        st.write(
            f"**Evidence {index}** — "
            f"{source['file']} / "
            f"Chunk {source['chunk']} / "
            f"Similarity {item['score']:.3f} / "
            f"{evidence_label(item['score'])}"
        )

        with st.expander(
            f"View Evidence {index}"
        ):

            st.write(
                item["chunk"]
            )

elif (
    incident.strip()
    and st.session_state.sop_chunks
):

    st.info(
        "No sufficiently relevant SOP evidence "
        "has been retrieved."
    )


# ============================================================
# FORMAT RAG CONTEXT
# ============================================================

def format_retrieved_context(results):

    sections = []

    for index, result in enumerate(
        results,
        start=1
    ):

        source = result["source"]

        sections.append(
            f"--- EVIDENCE {index} ---\n"
            f"Source: {source['file']}\n"
            f"Chunk: {source['chunk']}\n"
            f"Similarity: {result['score']:.4f}\n\n"
            f"{result['chunk']}"
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# ERROR CLASSIFICATION
# ============================================================

def is_quota_error(
    error_text
):

    text = error_text.lower()

    quota_terms = [
        "429",
        "resource_exhausted",
        "quota",
        "rate limit",
        "too many requests",
        "generate_content_free_tier",
    ]

    return any(
        term in text
        for term in quota_terms
    )


def is_service_unavailable(
    error_text
):

    text = error_text.lower()

    service_terms = [
        "503",
        "service unavailable",
        "currently experiencing high demand",
        "temporarily unavailable",
    ]

    return any(
        term in text
        for term in service_terms
    )


# ============================================================
# SECTION 3 — MULTI-AGENT WORKFLOW
# ============================================================

st.header(
    "3. 🤖 Run Multi-Agent SOP Workflow"
)


# ============================================================
# COOLDOWN
# ============================================================

elapsed = (
    time.time()
    - st.session_state.last_workflow_start
)

cooldown_remaining = max(
    0,
    int(
        WORKFLOW_COOLDOWN_SECONDS
        - elapsed
    )
)

if cooldown_remaining > 0:

    st.info(
        f"Free-tier protection: please wait "
        f"{cooldown_remaining} seconds before "
        f"starting another Gemini workflow."
    )


run_disabled = (
    cooldown_remaining > 0
)


# ============================================================
# WORKFLOW BUTTON
# ============================================================

if st.button(
    "🚀 Generate SOP-to-Action Workflow",
    type="primary",
    disabled=run_disabled,
):

    if not st.session_state.sop_chunks:

        st.warning(
            "Please upload and build the SOP "
            "knowledge base first."
        )

    elif not incident.strip():

        st.warning(
            "Please describe the operational "
            "incident first."
        )

    elif not st.session_state.retrieved_context:

        st.warning(
            "Please retrieve SOP evidence before "
            "running the multi-agent workflow."
        )

    else:

        st.session_state.last_workflow_start = (
            time.time()
        )

        shared_llm = get_agent_llm()

        if shared_llm is None:
            st.stop()

        context_text = (
            format_retrieved_context(
                st.session_state.retrieved_context
            )
        )

        try:

            with st.status(
                "Preparing four-agent workflow...",
                expanded=True
            ):

                st.write(
                    "🔹 Agent 1 — Operational Incident Triage"
                )

                triage_agent = (
                    create_triage_agent(
                        shared_llm
                    )
                )

                st.write(
                    "🔹 Agent 2 — SOP Compliance and Action Compiler"
                )

                compiler_agent = (
                    create_compiler_agent(
                        shared_llm
                    )
                )

                st.write(
                    "🔹 Agent 3 — Cross-Functional Department Routing"
                )

                router_agent = (
                    create_router_agent(
                        shared_llm
                    )
                )

                st.write(
                    "🔹 Agent 4 — Executive Action Brief"
                )

                automation_agent = (
                    create_automation_agent(
                        shared_llm
                    )
                )

                tasks = (
                    define_workflow_tasks(
                        triage_agent,
                        compiler_agent,
                        router_agent,
                        automation_agent,
                        incident,
                        context_text,
                    )
                )

                crew = Crew(
                    agents=[
                        triage_agent,
                        compiler_agent,
                        router_agent,
                        automation_agent,
                    ],

                    tasks=tasks,

                    process=Process.sequential,

                    verbose=False,

                    cache=True,

                    # Keep total workflow request rate
                    # conservative for the free tier.
                    max_rpm=4,
                )

                st.write(
                    "⚙️ Executing sequential CrewAI workflow..."
                )

                result = crew.kickoff()

                st.session_state.workflow_output = (
                    str(result)
                )

                st.session_state.workflow_approved = False

                st.session_state.workflow_rejected = False

            st.success(
                "✅ Four-agent workflow generated successfully."
            )

        except Exception as exc:

            error_text = str(exc)

            if is_quota_error(
                error_text
            ):

                st.error(
                    "### Gemini Free-Tier Rate Limit Reached"
                )

                st.warning(
                    "The workflow stopped because the "
                    "Gemini free-tier request limit was reached."
                )

                st.info(
                    "No paid service is being used. "
                    "Wait for the free-tier request window "
                    "to reset before trying again."
                )

            elif is_service_unavailable(
                error_text
            ):

                st.error(
                    "### Gemini Temporarily Unavailable"
                )

                st.warning(
                    "Gemini 3.8 Flash is temporarily unavailable "
                    "or experiencing high demand."
                )

                st.info(
                    "Your application, RAG system and API "
                    "configuration are still intact. "
                    "Please wait briefly and try the workflow "
                    "again once."
                )

            else:

                st.error(
                    "The multi-agent workflow encountered "
                    "an unexpected error."
                )

                with st.expander(
                    "Technical error details"
                ):

                    st.code(
                        error_text
                    )


# ============================================================
# SECTION 4 — EXECUTIVE ACTION BRIEF
# ============================================================

if st.session_state.workflow_output:

    st.header(
        "4. 📋 Executive Action Brief"
    )

    st.markdown(
        '<div class="workflow-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        st.session_state.workflow_output
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# SECTION 5 — EVIDENCE TRACEABILITY
# ============================================================

if st.session_state.workflow_output:

    st.header(
        "5. 📚 Evidence Traceability"
    )

    st.caption(
        "The following local SOP evidence was supplied "
        "to the SOP Compliance Agent before the workflow "
        "was generated."
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
# SECTION 6 — HUMAN APPROVAL
# ============================================================

if st.session_state.workflow_output:

    st.header(
        "6. 👤 Human Approval Gate"
    )

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
        "the workflow is considered approved."
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
"""
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    approval_confirmation = st.checkbox(
        "I have reviewed the generated workflow and supporting SOP evidence."
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "✅ Approve Workflow",
            disabled=not approval_confirmation,
            use_container_width=True,
        ):

            st.session_state.workflow_approved = True

            st.session_state.workflow_rejected = False


    with col2:

        if st.button(
            "❌ Reject / Request Review",
            use_container_width=True,
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
        "This approval records human review of the "
        "AI-generated recommendation. No external "
        "business transaction has been executed."
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

```python
import os
import streamlit as st
from crewai import Agent, LLM


# ============================================================
# GEMINI / CREWAI LLM
# ============================================================

def get_agent_llm():
    """
    Create a CrewAI-native Gemini LLM.

    IMPORTANT:
    Do not use LangChain's ChatGoogleGenerativeAI here.
    CrewAI 1.15.x expects its own LLM object or a supported
    model string.
    """

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # Streamlit Cloud / local secrets fallback
    if not api_key:
        try:
            api_key = str(
                st.secrets.get("GEMINI_API_KEY", "")
            ).strip()
        except Exception:
            api_key = ""

    if not api_key:
        st.error(
            "GEMINI_API_KEY was not found. "
            "Please configure it in Streamlit Secrets or "
            "your local environment variables."
        )
        return None

    try:
        llm = LLM(
            model="gemini/gemini-2.5-flash",
            api_key=api_key,
            temperature=0.0
        )

        return llm

    except Exception as exc:
        st.error(
            "CrewAI Gemini LLM initialization failed."
        )
        st.exception(exc)
        return None


# ============================================================
# AGENT 1 — INCIDENT TRIAGE
# ============================================================

def create_triage_agent(llm):

    return Agent(
        role="Operational Incident Triage Agent",

        goal=(
            "Analyze the user's operational incident description "
            "and identify the incident type, affected process, "
            "material or asset, operational risk indicators, "
            "and important search terms required to locate "
            "the applicable SOP evidence."
        ),

        backstory=(
            "You are the frontline operational triage specialist "
            "for a manufacturing and supply-chain organization. "
            "You convert an unstructured operational incident into "
            "clear operational information for downstream agents. "
            "Never invent facts. If information is missing, identify "
            "it as unknown rather than guessing."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# ============================================================
# AGENT 2 — SOP COMPLIANCE COMPILER
# ============================================================

def create_compiler_agent(llm):

    return Agent(
        role="SOP Compliance and Action Compiler",

        goal=(
            "Convert the supplied SOP evidence into a clear, "
            "sequential and operationally useful action checklist. "
            "Every recommended action must be supported by the "
            "retrieved SOP reference material."
        ),

        backstory=(
            "You are a strict operational compliance specialist. "
            "You work only with the SOP evidence supplied to you. "
            "Never invent company policies, procedures, approval "
            "limits, safety requirements, responsibilities, or "
            "documentation requirements. If the SOP evidence does "
            "not establish something, mark it as requiring "
            "human verification."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# ============================================================
# AGENT 3 — DEPARTMENT ROUTING
# ============================================================

def create_router_agent(llm):

    return Agent(
        role="Cross-Functional Department Routing Agent",

        goal=(
            "Map each required operational action to the most "
            "appropriate responsible department or organizational "
            "function using the incident information, SOP evidence, "
            "and preceding agent outputs."
        ),

        backstory=(
            "You are an organizational workflow specialist with "
            "experience in manufacturing, warehouse, quality, "
            "procurement and logistics operations. Identify the "
            "most appropriate responsible function for each action. "
            "Do not assume responsibility when it cannot be verified. "
            "Mark uncertain assignments as 'Human Verification Required'."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# ============================================================
# AGENT 4 — EXECUTIVE ACTION BRIEF
# ============================================================

def create_automation_agent(llm):

    return Agent(
        role="Executive Action Brief and Workflow Preparation Agent",

        goal=(
            "Combine the verified incident analysis, SOP action "
            "checklist, department responsibilities and source "
            "references into a concise Executive Action Brief "
            "prepared for human review and approval."
        ),

        backstory=(
            "You are a corporate operations documentation specialist. "
            "Your output must clearly distinguish verified SOP-based "
            "actions from information requiring human verification. "
            "The final recommendation must never bypass the mandatory "
            "human approval gate. You prepare the workflow for review; "
            "you do not independently authorize or execute it."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False
    )
```

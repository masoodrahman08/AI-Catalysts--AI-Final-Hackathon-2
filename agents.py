```python
import os
import streamlit as st

from crewai import Agent, LLM


# ============================================================
# GEMINI LLM CONFIGURATION
# ============================================================

def get_agent_llm():

    # --------------------------------------------------------
    # 1. Local machine environment variable
    # --------------------------------------------------------
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    # --------------------------------------------------------
    # 2. Streamlit Cloud Secrets fallback
    # --------------------------------------------------------
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
        except Exception:
            api_key = ""

    # --------------------------------------------------------
    # 3. Security check
    # --------------------------------------------------------
    if not api_key:
        st.error(
            "GEMINI_API_KEY was not found. "
            "Configure it as an environment variable or Streamlit Secret."
        )
        return None

    # --------------------------------------------------------
    # 4. CrewAI-native Gemini LLM
    # --------------------------------------------------------
    return LLM(
        model="gemini/gemini-2.5-flash",
        api_key=api_key,
        temperature=0.0
    )


# ============================================================
# AGENT 1 — INCIDENT TRIAGE
# ============================================================

def create_triage_agent(llm):

    return Agent(
        role="Incident Triage Agent",

        goal=(
            "Analyze the operational incident description and identify "
            "the key incident type, affected process, material or asset, "
            "risk indicators, and search terms required to locate the "
            "applicable SOP evidence."
        ),

        backstory=(
            "You are the frontline operational triage specialist. "
            "You convert an unstructured operational incident into "
            "structured and precise information for downstream agents. "
            "Never invent facts. If information is missing, explicitly "
            "identify it as unknown."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# ============================================================
# AGENT 2 — SOP COMPLIANCE
# ============================================================

def create_compiler_agent(llm):

    return Agent(
        role="SOP Compliance Agent",

        goal=(
            "Convert the retrieved SOP evidence into a clear, "
            "sequential and operationally useful action checklist. "
            "Every recommendation must be supported by the supplied "
            "reference material."
        ),

        backstory=(
            "You are a strict operational compliance specialist. "
            "You work only with the SOP evidence supplied to you. "
            "Never invent policies, procedures, approval limits, "
            "safety requirements, or responsibilities."
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
        role="Department Routing Agent",

        goal=(
            "Map each required action to the appropriate responsible "
            "department or organizational function based on the "
            "incident, SOP evidence, and preceding agent outputs."
        ),

        backstory=(
            "You are an organizational workflow specialist. "
            "Identify the most appropriate responsible function for "
            "each action. Do not assume responsibility when it cannot "
            "be verified. Mark uncertain assignments as "
            "Human Verification Required."
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
        role="Executive Action Brief Agent",

        goal=(
            "Combine the verified incident analysis, SOP action "
            "checklist, department responsibilities, and source "
            "references into a concise Executive Action Brief "
            "prepared for human review and approval."
        ),

        backstory=(
            "You are a corporate operations documentation specialist. "
            "Your output must clearly distinguish verified SOP-based "
            "actions from information requiring human verification. "
            "The final recommendation must never bypass the mandatory "
            "human approval gate."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False
    )
```

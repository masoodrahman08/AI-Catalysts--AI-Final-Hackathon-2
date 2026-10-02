```python
import os
import streamlit as st
from crewai import Agent, LLM


# ============================================================
# GEMINI / CREWAI LLM CONFIGURATION
# ============================================================

def get_agent_llm():
    """
    Create the CrewAI-native Gemini 3.8 Flash LLM.

    Authentication priority:
    1. GEMINI_API_KEY environment variable
    2. Streamlit Secrets

    The actual API key is never displayed or hard-coded.
    """

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
        except Exception:
            api_key = ""

    if not api_key:
        st.error(
            "🔒 Security Error: GEMINI_API_KEY was not found "
            "in the environment variables or Streamlit Secrets."
        )
        return None

    return LLM(
        model="gemini/gemini-3.8-flash",
        api_key=api_key
    )


# ============================================================
# AGENT 1 — OPERATIONAL INCIDENT TRIAGE
# ============================================================

def create_triage_agent(llm):

    return Agent(
        role="Operational Incident Triage Agent",

        goal=(
            "Analyze the reported operational incident and identify the "
            "incident type, affected process, material or asset involved, "
            "risk indicators, relevant search terms, and information gaps."
        ),

        backstory=(
            "You are the frontline operational triage specialist in a "
            "manufacturing, warehouse, logistics, and supply-chain "
            "environment. You convert an unstructured incident description "
            "into precise operational facts and search parameters. "
            "You must not invent facts or assume information that was not "
            "provided by the user."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# ============================================================
# AGENT 2 — SOP COMPLIANCE AND ACTION COMPILER
# ============================================================

def create_compiler_agent(llm):

    return Agent(
        role="SOP Compliance and Action Compiler",

        goal=(
            "Convert the retrieved and approved SOP evidence into a clear, "
            "sequential, operational action checklist without introducing "
            "rules, requirements, or procedures that are not supported by "
            "the retrieved evidence."
        ),

        backstory=(
            "You are a strict SOP compliance specialist. You work only "
            "with the evidence supplied from the approved corporate SOP "
            "knowledge base. You distinguish between documented procedures "
            "and information that requires human verification. "
            "If the SOP evidence does not establish a requirement, you "
            "must explicitly identify that gap rather than guessing."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# ============================================================
# AGENT 3 — CROSS-FUNCTIONAL DEPARTMENT ROUTING
# ============================================================

def create_router_agent(llm):

    return Agent(
        role="Cross-Functional Department Routing Agent",

        goal=(
            "Map each SOP-grounded action to the appropriate responsible "
            "function, identify dependencies and required documentation, "
            "and clearly flag any responsibility that cannot be established "
            "from the available evidence."
        ),

        backstory=(
            "You are an organizational workflow specialist working across "
            "Warehouse, Quality, Procurement, Logistics, Operations, and "
            "other business functions. You assign accountability based on "
            "the action and available evidence. You must not invent an "
            "organizational responsibility. When responsibility is unclear, "
            "mark it as 'Human Verification Required'."
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
            "Assemble the triage findings, SOP-grounded actions, department "
            "routing, documentation requirements, decision points, and "
            "source references into a concise executive action brief "
            "prepared for human review and approval."
        ),

        backstory=(
            "You are a corporate operations documentation specialist. "
            "You prepare controlled action briefs for managers and "
            "executives. Your output must clearly separate verified SOP "
            "requirements from information requiring human verification. "
            "You prepare the workflow for human approval but never claim "
            "that a business transaction has been executed."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False
    )
```

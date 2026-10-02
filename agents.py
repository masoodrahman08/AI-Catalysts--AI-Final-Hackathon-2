```python
import os
import streamlit as st
from crewai import Agent, LLM


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

MODEL_NAME = "gemini/gemini-3.8-flash"


def get_agent_llm():
    """
    Create the CrewAI-native Gemini LLM.

    The API key is read from:
    1. Environment variable GEMINI_API_KEY
    2. Streamlit Secrets

    No API key is hard-coded in the application.
    """

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        try:
            api_key = str(
                st.secrets.get("GEMINI_API_KEY", "")
            ).strip()
        except Exception:
            api_key = ""

    if not api_key:
        st.error(
            "Gemini API key was not found. "
            "Please configure GEMINI_API_KEY in Streamlit Secrets "
            "or as an environment variable."
        )
        return None

    try:
        return LLM(
            model=MODEL_NAME,
            api_key=api_key
        )

    except Exception as exc:
        st.error(
            f"Gemini LLM configuration error: {exc}"
        )
        return None


# ============================================================
# COMMON AGENT SETTINGS
# ============================================================

FREE_TIER_SETTINGS = {
    # One main execution cycle per agent.
    "max_iter": 1,

    # Prevent unnecessary automatic agent retries.
    "max_retry_limit": 0,

    # Agents must not delegate to other agents.
    "allow_delegation": False,

    # Keep the public application clean.
    "verbose": False,

    # We are not using tools in this MVP.
    "cache": True,

    # Conservative request rate.
    "max_rpm": 1,

    # Keep context management enabled.
    "respect_context_window": True,
}


# ============================================================
# AGENT 1
# OPERATIONAL INCIDENT TRIAGE
# ============================================================

def create_triage_agent(llm):

    return Agent(
        role="Operational Incident Triage Agent",

        goal=(
            "Analyze the user's operational incident and extract "
            "the essential facts, affected process, incident type, "
            "and useful search terms without inventing information."
        ),

        backstory=(
            "You are the first-line operational intake specialist. "
            "Your responsibility is to structure an unstructured "
            "incident description so that downstream SOP analysis "
            "can be performed accurately. "
            "Never invent facts, procedures, responsibilities, "
            "approvals, or decisions."
        ),

        llm=llm,
        **FREE_TIER_SETTINGS
    )


# ============================================================
# AGENT 2
# SOP COMPLIANCE AND ACTION COMPILER
# ============================================================

def create_compiler_agent(llm):

    return Agent(
        role="SOP Compliance and Action Compiler",

        goal=(
            "Convert retrieved SOP evidence into a concise, "
            "sequential and operationally useful action checklist."
        ),

        backstory=(
            "You are a strict SOP compliance specialist. "
            "You work only with the SOP evidence supplied to you. "
            "Do not create policies, procedures, approvals, "
            "responsibilities, or requirements that are absent "
            "from the evidence."
        ),

        llm=llm,
        **FREE_TIER_SETTINGS
    )


# ============================================================
# AGENT 3
# CROSS-FUNCTIONAL ROUTING
# ============================================================

def create_router_agent(llm):

    return Agent(
        role="Cross-Functional Department Routing Agent",

        goal=(
            "Map each validated operational action to the most "
            "appropriate responsible department or function."
        ),

        backstory=(
            "You are an enterprise workflow routing specialist. "
            "You examine the approved action checklist and identify "
            "responsible functions. If the evidence does not "
            "support a responsibility assignment, explicitly mark "
            "it as Human Verification Required rather than guessing."
        ),

        llm=llm,
        **FREE_TIER_SETTINGS
    )


# ============================================================
# AGENT 4
# EXECUTIVE ACTION BRIEF
# ============================================================

def create_automation_agent(llm):

    return Agent(
        role="Executive Action Brief and Workflow Preparation Agent",

        goal=(
            "Combine the validated incident, SOP actions and "
            "department routing into a concise executive action "
            "brief ready for human review and approval."
        ),

        backstory=(
            "You are a corporate operations documentation specialist. "
            "Your output must clearly identify the incident, applicable "
            "SOP actions, responsible functions, required documents, "
            "decision points and human approval requirement. "
            "You prepare recommendations only. "
            "You never claim that a real business transaction "
            "has been executed."
        ),

        llm=llm,
        **FREE_TIER_SETTINGS
    )
```

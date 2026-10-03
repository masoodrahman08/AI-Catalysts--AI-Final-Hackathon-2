
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

    API key sources:
    1. GEMINI_API_KEY environment variable
    2. Streamlit Secrets

    No API key is stored in source code.
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
# FREE-TIER AGENT SETTINGS
# ============================================================

FREE_TIER_SETTINGS = {
    # One main reasoning/execution iteration.
    "max_iter": 1,

    # Prevent automatic agent retries.
    "max_retry_limit": 0,

    # Agents cannot delegate to other agents.
    "allow_delegation": False,

    # Keep public application output clean.
    "verbose": False,

    # Allow CrewAI caching.
    "cache": True,

    # Conservative request rate.
    "max_rpm": 1,

    # Respect context limits.
    "respect_context_window": True,
}


# ============================================================
# AGENT 1 — INCIDENT TRIAGE
# ============================================================

def create_triage_agent(llm):

    return Agent(
        role="Operational Incident Triage Agent",

        goal=(
            "Analyze the operational incident and extract the "
            "essential facts, affected process, incident type, "
            "and useful technical search terms without inventing "
            "information."
        ),

        backstory=(
            "You are the first-line operational intake specialist. "
            "You structure an unstructured incident description "
            "for downstream SOP analysis. "
            "Never invent facts, procedures, responsibilities, "
            "approvals, or decisions."
        ),

        llm=llm,
        **FREE_TIER_SETTINGS
    )


# ============================================================
# AGENT 2 — SOP COMPLIANCE
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
            "You work only with the evidence supplied from the "
            "approved SOP knowledge base. "
            "Do not create unsupported procedures, policies, "
            "approvals, responsibilities, or requirements."
        ),

        llm=llm,
        **FREE_TIER_SETTINGS
    )


# ============================================================
# AGENT 3 — DEPARTMENT ROUTING
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
            "Assign responsibilities only when supported by the "
            "available information. "
            "If responsibility cannot be established confidently, "
            "mark it as Human Verification Required."
        ),

        llm=llm,
        **FREE_TIER_SETTINGS
    )


# ============================================================
# AGENT 4 — EXECUTIVE ACTION BRIEF
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
            "Clearly identify the incident, applicable SOP actions, "
            "responsible functions, required documents, decision "
            "points and human approval requirement. "
            "You prepare recommendations only. "
            "You never claim that a real business transaction "
            "has been executed."
        ),

        llm=llm,
        **FREE_TIER_SETTINGS
    )

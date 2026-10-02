import os
import streamlit as st
from crewai import Agent, LLM


# ============================================================
# GEMINI / CREWAI LLM
# ============================================================

def get_agent_llm():
    """
    Create a CrewAI-native LLM using Gemini 2.5 Flash.

    IMPORTANT:
    This intentionally uses CrewAI's native LLM class.
    Do NOT use ChatGoogleGenerativeAI here.
    """

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "").strip()
        except Exception:
            api_key = ""

    if not api_key:
        st.error(
            "🔒 GEMINI_API_KEY was not found. "
            "Please configure it in Streamlit Secrets or the local environment."
        )
        return None

    try:
        return LLM(
            model="gemini/gemini-2.5-flash",
            api_key=api_key,
            temperature=0.0
        )

    except Exception as exc:
        st.error(f"Unable to initialize Gemini through CrewAI: {exc}")
        return None


# ============================================================
# AGENT 1 — OPERATIONAL TRIAGE
# ============================================================

def create_triage_agent(llm):

    return Agent(
        role="Operational Incident Triage Agent",

        goal=(
            "Analyze the user's operational incident and identify the core "
            "business process, affected material or asset, incident type, "
            "risk indicators, and precise search terms needed to locate "
            "the applicable SOP evidence."
        ),

        backstory=(
            "You are the frontline operational intake specialist for a "
            "manufacturing and supply-chain organization. You convert "
            "unstructured incident descriptions into precise operational "
            "information. You must not invent facts, policies, causes, "
            "responsibilities, or decisions that are not supported by the "
            "incident description."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False
    )


# ============================================================
# AGENT 2 — SOP COMPLIANCE / ACTION COMPILER
# ============================================================

def create_compiler_agent(llm):

    return Agent(
        role="SOP Compliance and Action Compiler",

        goal=(
            "Convert retrieved SOP evidence into a clear, sequential and "
            "practical operational action checklist while remaining strictly "
            "grounded in the supplied SOP evidence."
        ),

        backstory=(
            "You are a strict SOP compliance specialist. You work only with "
            "the approved SOP evidence supplied to you. You distinguish "
            "documented requirements from missing information. You never "
            "invent procedures, approval limits, responsibilities, safety "
            "requirements, or business rules."
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
            "Map every SOP-grounded action to the appropriate responsible "
            "business function, identify dependencies and required documents, "
            "and clearly flag responsibilities that require human verification."
        ),

        backstory=(
            "You are an organizational process-routing specialist working "
            "across Warehouse, Quality, Procurement, Production, Logistics, "
            "Finance, Legal and other business functions. You assign "
            "responsibility only when supported by the available evidence. "
            "If the SOP evidence does not establish ownership, explicitly "
            "mark the responsibility as 'Human Verification Required'."
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
            "Assemble the grounded incident analysis, SOP actions, department "
            "routing, required documentation, decision points and human "
            "approval requirement into a concise executive operational brief."
        ),

        backstory=(
            "You are a corporate operations documentation specialist. You "
            "prepare controlled action briefs for management review. Your "
            "output must clearly distinguish facts, SOP-grounded actions, "
            "missing information, human verification requirements and "
            "management approval. You do not claim that an action has been "
            "executed when the system has only prepared a recommendation."
        ),

        llm=llm,
        verbose=True,
        allow_delegation=False
    )

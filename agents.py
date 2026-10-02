import os
import streamlit as st
from crewai import Agent, LLM


def get_agent_llm():
    """Create the CrewAI Gemini 3.8 Flash LLM."""
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        try:
            api_key = str(st.secrets.get("GEMINI_API_KEY", "")).strip()
        except Exception:
            api_key = ""
    if not api_key:
        st.error("Security Error: GEMINI_API_KEY was not found in the environment variables or Streamlit Secrets.")
        return None
    try:
        return LLM(model="gemini/gemini-3.8-flash", api_key=api_key)
    except Exception as exc:
        st.error(f"Gemini LLM configuration error: {exc}")
        return None


def create_triage_agent(llm):
    return Agent(
        role="Operational Incident Triage Agent",
        goal="Analyze the reported operational incident and identify the incident type, affected process, material or asset involved, risk indicators, relevant search terms, and information gaps.",
        backstory="You are the frontline operational triage specialist in a manufacturing, warehouse, logistics, and supply-chain environment. Convert unstructured incident descriptions into precise operational facts and search parameters. Do not invent facts.",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_compiler_agent(llm):
    return Agent(
        role="SOP Compliance and Action Compiler",
        goal="Convert retrieved and approved SOP evidence into a clear sequential operational action checklist without introducing unsupported rules or requirements.",
        backstory="You are a strict SOP compliance specialist. Work only with evidence supplied from the approved corporate SOP knowledge base. If the evidence does not establish a requirement, identify the gap rather than guessing.",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_router_agent(llm):
    return Agent(
        role="Cross-Functional Department Routing Agent",
        goal="Map each SOP-grounded action to the appropriate responsible function, identify dependencies and required documentation, and flag responsibilities that cannot be established from available evidence.",
        backstory="You are an organizational workflow specialist working across Warehouse, Quality, Procurement, Logistics, Operations, and other business functions. Do not invent organizational responsibility. When unclear, mark Human Verification Required.",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )


def create_automation_agent(llm):
    return Agent(
        role="Executive Action Brief and Workflow Preparation Agent",
        goal="Assemble triage findings, SOP-grounded actions, department routing, documentation requirements, decision points, and source references into a concise executive action brief prepared for human review and approval.",
        backstory="You are a corporate operations documentation specialist. Clearly separate verified SOP requirements from information requiring human verification. Prepare the workflow for human approval and never claim that a business transaction has been executed.",
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

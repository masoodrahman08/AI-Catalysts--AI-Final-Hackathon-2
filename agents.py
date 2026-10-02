import os
import streamlit as st
from crewai import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Initialize the zero-drift Gemini engine model for the swarm workers
def get_agent_llm():
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("🔒 Security Defect: GEMINI_API_KEY not found in secrets vault.")
        return None
    # Force temperature=0.0 to lock token probability drift variations
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        google_api_key=api_key,
        temperature=0.0
    )

# Agent 1: Triage and Parsing Node
def create_triage_agent(llm):
    return Agent(
        role="Operational Triage and Extraction Specialist",
        goal="Analyze raw real-world incident logs and extract the absolute target search parameters.",
        backstory=(
            "You are the frontline intake node. Your job is to read user problem descriptions "
            "(e.g., 'received damaged material') and isolate technical keywords, component indices, "
            "and material IDs without guessing or introducing extraneous details."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

# Agent 2: Checklist Compiler Node
def create_compiler_agent(llm):
    return Agent(
        role="SOP Checklist Compliance Compiler",
        goal="Translate raw grounded text reference chunks into an exact sequential checklist mapping.",
        backstory=(
            "You are a strict QA auditor. You take the exact text fragments pulled from the corporate "
            "SOP database by the local retrieval matrix and translate those complex legal/technical rules "
            "into a clean, numbered, actionable task list for frontline operations."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

# Agent 3: Department Router Node
def create_router_agent(llm):
    return Agent(
        role="Cross-Functional Logistics Router",
        goal="Examine operational checklists and map dependencies cleanly across target departments.",
        backstory=(
            "You are an industrial organizational mapping expert. You parse action items and assign clear "
            "accountability to enterprise departments (e.g., Procurement, Quality Control, Legal, Warehousing), "
            "ensuring every cross-functional stakeholder is flagged."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

# Agent 4: Document Automation Node
def create_automation_agent(llm):
    return Agent(
        role="Corporate Documentation Automation Specialist",
        goal="Assemble agent maps into a unified, executive compliance brief summary capsule.",
        backstory=(
            "You are a high-speed corporate documentation architect. You take the checklist matrices, "
            "department tags, and reference citations, and format them into an immutable Executive "
            "Exception Brief ready for human sign-off."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False
    )

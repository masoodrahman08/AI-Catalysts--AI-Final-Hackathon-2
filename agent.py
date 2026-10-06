import os
from crewai import Agent, Task, Crew, Process, LLM
from tools import (
    sop_search_rag, 
    generate_action_checklist, 
    draft_operational_artifacts
)

# --- FIX FOR GROQ CACHE BREAKPOINT ERROR ---
import crewai.llms.cache as _crewai_cache
_crewai_cache.mark_cache_breakpoint = lambda msg: msg
# -------------------------------------------

os.environ["PYTHONIOENCODING"] = "utf-8"

def sanitize_text(text: str) -> str:
    """Removes non-ASCII characters to prevent encoding issues."""
    return text.encode("ascii", "ignore").decode("ascii")

def run_sop_crew(user_incident: str, api_key: str, model_name: str = "groq/openai/gpt-oss-120b"):
    clean_incident = sanitize_text(user_incident)
    
    llm = LLM(
        model=model_name,
        api_key=api_key
    )
    
    sop_agent = Agent(
        role="Operations SOP and Execution Manager",
        goal="Analyze incidents using internal SOP tools and generate actionable resolution workflows.",
        backstory=(
            "An expert operational manager specializing in logistics, compliance, and automated risk response. "
            "You rely strictly on company SOPs and domain knowledge to formulate operational procedures."
        ),
        tools=[sop_search_rag, generate_action_checklist, draft_operational_artifacts],
        llm=llm,
        verbose=True
    )
    
    sop_task = Task(
        description=f"""
The following operational incident was reported:
'{clean_incident}'

Follow this exact sequence:
1. Use the 'SOP Policy Search Tool' to retrieve relevant procedures for this incident.
2. Identify and explicitly state the specific SOP document name (e.g., '[Document Source: Warehouse_SOP.pdf]') and section from which the retrieved policies originate.
3. Use the 'Action Checklist Generator' to turn retrieved guidelines into a structured task list.
4. Use the 'Operational Artifact Drafter' to prepare draft communications or tickets.
5. Consolidate everything into a clear executive response for human approval.
""",
        expected_output=(
            "A structured report containing: "
            "1) Cited SOP Document Name & Summary, "
            "2) Departmental Action Checklist, and "
            "3) Pre-filled Operational Drafts."
        ),
        agent=sop_agent
    )
    
    crew = Crew(
        agents=[sop_agent],
        tasks=[sop_task],
        process=Process.sequential
    )
    
    return crew.kickoff()

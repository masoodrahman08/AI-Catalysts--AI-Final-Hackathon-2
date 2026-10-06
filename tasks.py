from crewai import Task, Crew, Process
from agents import create_policy_analyst, create_operations_orchestrator

def run_sop_multi_agent_workflow(user_incident: str, sop_context: str, api_key: str, model_name: str = "groq/openai/gpt-oss-120b"):
    """Executes the two-agent sequential workflow."""
    
    policy_analyst = create_policy_analyst(api_key, model_name)
    operations_orchestrator = create_operations_orchestrator(api_key, model_name)

    # Task 1: Compliance Analysis
    policy_task = Task(
        description=(
            f"Examine the user incident report:\n'{user_incident}'\n\n"
            f"Using the retrieved SOP context below:\n{sop_context}\n\n"
            "1. Identify the exact SOP guidelines applicable to this situation.\n"
            "2. Note required emergency/compliance actions and cite the specific source document name.\n"
            "3. Highlight potential risks if compliance steps are delayed."
        ),
        expected_output="A structured policy report containing identified SOP rules, citations, and risk factors.",
        agent=policy_analyst
    )

    # Task 2: Action Plan Synthesis
    orchestration_task = Task(
        description=(
            "Review the compliance analysis provided by the Policy Analyst.\n"
            "Create a comprehensive operational execution package including:\n"
            "1. Executive Incident Summary (2-3 sentences)\n"
            "2. Prioritized Action Checklist (High / Medium / Low priority tasks)\n"
            "3. Pre-filled Communication Draft (e.g., supplier claim email, maintenance ticket, or incident log)\n"
            "4. Human-in-the-Loop Status Flag: Mark clearly as 'PENDING MANAGER APPROVAL'."
        ),
        expected_output="A clean, formatted markdown response containing the summary, prioritized checklist, ready-to-send draft artifact, and approval flag.",
        agent=operations_orchestrator
    )

    crew = Crew(
        agents=[policy_analyst, operations_orchestrator],
        tasks=[policy_task, orchestration_task],
        process=Process.sequential,
        verbose=True
    )

    return crew.kickoff()

from crewai import Agent, LLM

def create_policy_analyst(groq_api_key: str, model_name: str = "groq/openai/gpt-oss-120b") -> Agent:
    """Agent focused on compliance auditing and SOP source citation."""
    llm = LLM(
        model=model_name,
        api_key=groq_api_key
    )
    return Agent(
        role="Senior SOP Policy & Compliance Analyst",
        goal="Analyze reported operational incidents against retrieved SOP documentation to identify exact policy violations, safety protocols, and source citations.",
        backstory=(
            "You are an expert compliance auditor in industrial and logistics operations. "
            "Your duty is to meticulously scan company SOP manuals, cross-reference incident descriptions, "
            "and determine strict regulatory boundaries without missing critical safety or legal details."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

def create_operations_orchestrator(groq_api_key: str, model_name: str = "groq/openai/gpt-oss-120b") -> Agent:
    """Agent focused on converting findings into checklists and artifact drafts."""
    llm = LLM(
        model=model_name,
        api_key=groq_api_key
    )
    return Agent(
        role="Lead Operations & Execution Orchestrator",
        goal="Transform policy compliance analysis into prioritized, actionable operational checklists, emergency tasks, and communication drafts for executive approval.",
        backstory=(
            "You are an experienced warehouse and facility operations lead. "
            "You take raw compliance and safety analysis and turn it into swift, structured, real-world execution plans. "
            "You draft vendor emails, assign ticket priorities, and format executive briefs for manager sign-off."
        ),
        verbose=True,
        allow_delegation=False,
        llm=llm
    )

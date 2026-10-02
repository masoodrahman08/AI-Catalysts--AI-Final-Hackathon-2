```python
from crewai import Task


# ============================================================
# MULTI-AGENT WORKFLOW TASKS
# ============================================================

def define_workflow_tasks(
    triage_worker,
    compiler_worker,
    router_worker,
    automation_worker,
    incident_description,
    context_chunks
):
    """
    Build the sequential four-agent workflow.

    Workflow:

        Incident
            ↓
        Agent 1 — Triage
            ↓
        Agent 2 — SOP Compliance
            ↓
        Agent 3 — Department Routing
            ↓
        Agent 4 — Executive Action Brief
            ↓
        Human Approval
    """

    # ========================================================
    # TASK 1 — TRIAGE
    # ========================================================

    task_1_triage = Task(
        description=(
            "Analyze the following operational incident:\n\n"
            f"{incident_description}\n\n"

            "Identify:\n"
            "1. Incident type\n"
            "2. Affected process\n"
            "3. Material, equipment or asset involved\n"
            "4. Operational risk indicators\n"
            "5. Important technical/search terms\n"
            "6. Missing information that requires clarification\n\n"

            "Do not invent facts. "
            "If information is not available, explicitly state "
            "that it is unknown."
        ),

        expected_output=(
            "A concise operational triage summary containing the "
            "incident type, affected process, relevant entities, "
            "risk indicators, search terms and unknown information."
        ),

        agent=triage_worker
    )


    # ========================================================
    # TASK 2 — SOP COMPILATION
    # ========================================================

    task_2_compile = Task(
        description=(
            "Use the operational triage information from the "
            "previous task together with the following retrieved "
            "SOP evidence.\n\n"

            "================ SOP GROUNDING EVIDENCE ================\n"
            f"{context_chunks}\n"
            "==========================================================\n\n"

            "Convert the verified SOP evidence into a sequential "
            "operational action checklist.\n\n"

            "Rules:\n"
            "1. Use only information supported by the supplied SOP evidence.\n"
            "2. Do not create company procedures that are not present.\n"
            "3. Do not invent approval requirements.\n"
            "4. Do not invent department responsibilities.\n"
            "5. Clearly identify any item requiring human verification.\n"
            "6. Preserve important SOP controls and conditions.\n"
            "7. Prefer practical actions that frontline employees can follow."
        ),

        expected_output=(
            "A numbered SOP-grounded operational action checklist. "
            "Each action must be traceable to the supplied SOP evidence "
            "or explicitly marked as requiring human verification."
        ),

        agent=compiler_worker,

        context=[task_1_triage]
    )


    # ========================================================
    # TASK 3 — DEPARTMENT ROUTING
    # ========================================================

    task_3_route = Task(
        description=(
            "Review the SOP-grounded action checklist produced by "
            "the previous agent.\n\n"

            "For every action:\n"
            "1. Identify the most appropriate responsible department "
            "or organizational function.\n"
            "2. Identify dependencies on other departments.\n"
            "3. Identify required records or documents when supported "
            "by the available evidence.\n"
            "4. Mark responsibility as 'Human Verification Required' "
            "when it cannot be reliably established.\n\n"

            "Do not force every action into a department if the evidence "
            "does not support the assignment."
        ),

        expected_output=(
            "A clear task-to-department routing matrix containing "
            "Action, Responsible Function, Supporting Function, "
            "Required Document/Record where known, and Verification "
            "Status."
        ),

        agent=router_worker,

        context=[task_2_compile]
    )


    # ========================================================
    # TASK 4 — EXECUTIVE ACTION BRIEF
    # ========================================================

    task_4_automate = Task(
        description=(
            "Prepare the final Executive Action Brief using the "
            "incident analysis, SOP checklist and department routing "
            "information from the preceding agents.\n\n"

            "The final brief must contain these sections:\n\n"

            "1. INCIDENT SUMMARY\n"
            "2. SOP-GROUNDED IMMEDIATE ACTIONS\n"
            "3. RESPONSIBLE FUNCTIONS\n"
            "4. REQUIRED DOCUMENTS / RECORDS\n"
            "5. DECISION OR DISPOSITION REQUIRED\n"
            "6. ITEMS REQUIRING HUMAN VERIFICATION\n"
            "7. SOURCE / SOP REFERENCES\n"
            "8. HUMAN APPROVAL GATE\n\n"

            "At the end, explicitly state that the action plan is "
            "prepared for human review and must not be released or "
            "executed without authorized human approval.\n\n"

            "Do not claim that the AI has independently approved, "
            "released, executed, purchased, returned, quarantined, "
            "disposed of, or otherwise committed company resources."
        ),

        expected_output=(
            "A concise, professional Executive Action Brief in "
            "Markdown format, suitable for review by an authorized "
            "operations manager."
        ),

        agent=automation_worker,

        context=[task_3_route]
    )


    return [
        task_1_triage,
        task_2_compile,
        task_3_route,
        task_4_automate
    ]
```

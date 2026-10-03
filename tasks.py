
from crewai import Task


# ============================================================
# WORKFLOW TASK DEFINITIONS
# ============================================================

def define_workflow_tasks(
    triage_worker,
    compiler_worker,
    router_worker,
    automation_worker,
    incident_description,
    context_chunks
):

    # ========================================================
    # TASK 1 — INCIDENT TRIAGE
    # ========================================================

    task_1_triage = Task(

        description=(
            "Analyze the following operational incident.\n\n"

            f"INCIDENT:\n{incident_description}\n\n"

            "Extract only:\n"
            "1. Incident type\n"
            "2. Affected process, material or activity\n"
            "3. Important operational facts\n"
            "4. Useful technical/search terms\n\n"

            "Do not invent information that is not present "
            "in the incident description."
        ),

        expected_output=(
            "A concise structured incident summary containing "
            "incident type, affected process/material/activity, "
            "key facts and search terms."
        ),

        agent=triage_worker
    )


    # ========================================================
    # TASK 2 — SOP COMPLIANCE
    # ========================================================

    task_2_compile = Task(

        description=(
            "Review the incident information from the previous "
            "agent together with the retrieved SOP evidence below.\n\n"

            "RETRIEVED SOP EVIDENCE:\n"
            f"{context_chunks}\n\n"

            "Create a numbered operational action checklist.\n\n"

            "STRICT RULES:\n"
            "- Use only the supplied SOP evidence.\n"
            "- Do not invent procedures.\n"
            "- Do not invent safety requirements.\n"
            "- Do not invent approvals.\n"
            "- Do not invent responsibilities.\n"
            "- Clearly state when evidence is insufficient.\n"
            "- Keep actions practical and sequential."
        ),

        expected_output=(
            "A concise numbered SOP-grounded operational action "
            "checklist. Unsupported actions must not be presented "
            "as confirmed SOP requirements."
        ),

        agent=compiler_worker,

        context=[task_1_triage]
    )


    # ========================================================
    # TASK 3 — DEPARTMENT ROUTING
    # ========================================================

    task_3_route = Task(

        description=(
            "Review the SOP action checklist produced by the "
            "previous agent.\n\n"

            "For every action identify:\n"
            "- Responsible department or function\n"
            "- Required supporting activity\n"
            "- Human Verification Required if responsibility "
            "cannot be established from the available information.\n\n"

            "Do not invent organizational responsibility."
        ),

        expected_output=(
            "A concise action-to-department routing matrix "
            "showing action, responsible function and any "
            "human verification requirement."
        ),

        agent=router_worker,

        context=[task_2_compile]
    )


    # ========================================================
    # TASK 4 — EXECUTIVE ACTION BRIEF
    # ========================================================

    task_4_automate = Task(

        description=(
            "Prepare the final Executive Action Brief using "
            "the validated outputs from the previous agents.\n\n"

            "Include:\n"
            "1. Incident Summary\n"
            "2. Applicable SOP Evidence\n"
            "3. Required Actions\n"
            "4. Responsible Functions\n"
            "5. Required Documents or Records\n"
            "6. Decision / Approval Point\n"
            "7. Human Approval Requirement\n\n"

            "Important:\n"
            "- Do not claim that a transaction was executed.\n"
            "- Do not claim that an approval has already occurred.\n"
            "- Clearly distinguish SOP-supported actions from "
            "items requiring human verification.\n"
            "- The final output is a recommendation for human review."
        ),

        expected_output=(
            "A concise executive operational action brief "
            "ready for human review and approval."
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

from crewai import Task


# ============================================================
# CREWAI WORKFLOW TASKS
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
    Build the four sequential CrewAI tasks.

    Workflow:

        Incident
           ↓
        Triage
           ↓
        SOP Compliance
           ↓
        Department Routing
           ↓
        Executive Action Brief
    """

    # ========================================================
    # TASK 1 — INCIDENT TRIAGE
    # ========================================================

    task_1_triage = Task(
        description=(
            "Analyze the following raw operational incident:\n\n"
            f"{incident_description}\n\n"

            "Identify:\n"
            "1. Incident type\n"
            "2. Affected process\n"
            "3. Material, product, equipment or asset involved\n"
            "4. Observable problem or exception\n"
            "5. Potential operational risk indicators explicitly stated\n"
            "6. Important operational entities\n"
            "7. Precise search terms that should be used to locate "
            "the applicable SOP evidence\n"
            "8. Information that is missing or uncertain\n\n"

            "Do not invent causes, policies, responsibilities or decisions. "
            "Do not provide a final operational decision."
        ),

        expected_output=(
            "A structured incident triage containing incident type, "
            "affected process, relevant entities, search terms and "
            "unknown or missing information."
        ),

        agent=triage_worker
    )

    # ========================================================
    # TASK 2 — SOP COMPLIANCE
    # ========================================================

    task_2_compile = Task(
        description=(
            "Use the incident triage from the previous task together with "
            "the following locally retrieved SOP evidence:\n\n"

            f"{context_chunks}\n\n"

            "Create a practical sequential operational checklist.\n\n"

            "Rules:\n"
            "1. Use ONLY the supplied SOP evidence for procedural requirements.\n"
            "2. Do not invent policies or procedures.\n"
            "3. Do not invent approval limits.\n"
            "4. Do not invent department responsibilities.\n"
            "5. Clearly identify information that is not available in the SOP evidence.\n"
            "6. Where the evidence is insufficient, write "
            "'Human Verification Required'.\n"
            "7. Preserve important conditions, limits and exceptions contained "
            "in the supplied evidence.\n"
            "8. Each action should be practical and sequential."
        ),

        expected_output=(
            "A numbered SOP-grounded operational action checklist, including "
            "conditions, required records and clearly identified evidence gaps."
        ),

        agent=compiler_worker,
        context=[task_1_triage]
    )

    # ========================================================
    # TASK 3 — DEPARTMENT ROUTING
    # ========================================================

    task_3_route = Task(
        description=(
            "Review the SOP-grounded action checklist produced by the "
            "previous task.\n\n"

            "For every action, determine the responsible business function "
            "ONLY when the responsibility is supported by the incident "
            "evidence or SOP evidence.\n\n"

            "Possible functions may include:\n"
            "- Warehouse\n"
            "- Quality / QC\n"
            "- Procurement\n"
            "- Production\n"
            "- Logistics\n"
            "- Finance\n"
            "- Legal\n"
            "- Management\n"
            "- Other function supported by the evidence\n\n"

            "For each action identify:\n"
            "1. Action\n"
            "2. Responsible function\n"
            "3. Dependency, if any\n"
            "4. Required document or record, if stated\n"
            "5. Human verification requirement\n\n"

            "If responsibility cannot be established from the available "
            "evidence, write 'Human Verification Required'. "
            "Do not guess ownership."
        ),

        expected_output=(
            "A clear action-to-department routing matrix showing action, "
            "responsible function, dependency, required documentation and "
            "human verification requirements."
        ),

        agent=router_worker,
        context=[task_2_compile]
    )

    # ========================================================
    # TASK 4 — EXECUTIVE ACTION BRIEF
    # ========================================================

    task_4_automate = Task(
        description=(
            "Prepare the final Executive Action Brief using the previous "
            "workflow outputs.\n\n"

            "The brief must contain these sections:\n\n"

            "1. INCIDENT SUMMARY\n"
            "2. SOP-GROUNDED REQUIRED ACTIONS\n"
            "3. RESPONSIBLE FUNCTIONS\n"
            "4. REQUIRED DOCUMENTS / RECORDS\n"
            "5. DECISION OR DISPOSITION REQUIRED\n"
            "6. INFORMATION GAPS / HUMAN VERIFICATION\n"
            "7. VERIFIED SOP EVIDENCE\n"
            "8. HUMAN-IN-THE-LOOP APPROVAL GATE\n\n"

            "The final brief must clearly state that the system prepares "
            "a recommended operational workflow for human review. "
            "It must not claim that the workflow has been executed, "
            "approved or released.\n\n"

            "At the top of the approval section include exactly:\n\n"
            "=== MANDATORY HUMAN-IN-THE-LOOP APPROVAL ===\n\n"

            "The final output must remain concise, practical and suitable "
            "for management review."
        ),

        expected_output=(
            "A concise Executive Action Brief containing the incident "
            "summary, SOP-grounded actions, responsible functions, "
            "documents, decision points, evidence gaps, SOP references "
            "and mandatory human approval gate."
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

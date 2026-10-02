from crewai import Task

def define_workflow_tasks(triage_worker, compiler_worker, router_worker, automation_worker, incident_description, context_chunks):
    
    task_1_triage = Task(
        description=(
            f"Read this raw operational exception log: '{incident_description}'. "
            "Extract the core technical entities, operational failures, and context terms. "
            "Output only the optimized technical search tokens."
        ),
        expected_output="A brief, high-density listing of extracted technical incident tokens.",
        agent=triage_worker
    )

    task_2_compile = Task(
        description=(
            "Take the incident data and cross-examine it strictly using this retrieved corporate SOP "
            f"grounding data chunk: \n\n{context_chunks}\n\n. Translate these operational boundaries "
            "into a step-by-step sequential action checklist. Do not reference rules not present in this text."
        ),
        expected_output="A numbered, context-grounded operational compliance task checklist.",
        agent=compiler_worker,
        context=[task_1_triage]
    )

    task_3_route = Task(
        description=(
            "Evaluate the action checklist compiled in the previous step. For every checkbox entry, "
            "determine which department (Procurement, Legal, Warehousing, QA) holds primary accountability. "
            "Format the output as a clear task-to-department matrix map."
        ),
        expected_output="A clean department routing allocation map detailing active cross-functional owners.",
        agent=router_worker,
        context=[task_2_compile]
    )

    task_4_automate = Task(
        description=(
            "Combine the sequential checklist metrics, department matrix layouts, and raw text citations "
            "into a unified Executive Exception Brief layout capsule. Ensure a header block placeholder "
            "entitled '=== MANDATORY HUMAN-IN-THE-LOOP APPROVAL SIGN-OFF ===' is printed explicitly at the top."
        ),
        expected_output="The final finalized markdown Operational Action Map brief dossier.",
        agent=automation_worker,
        context=[task_3_route]
    )

    return [task_1_triage, task_2_compile, task_3_route, task_4_automate]

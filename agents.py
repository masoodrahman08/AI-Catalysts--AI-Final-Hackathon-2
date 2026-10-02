```python
import os
import streamlit as st
from crewai import Agent, LLM


# ============================================================
# GEMINI LLM CONFIGURATION
# FREE-TIER OPTIMIZED
# ============================================================

def get_agent_llm():
    """
    Configure CrewAI to use Gemini 3.8 Flash.

    The configuration is intentionally conservative because
    the project is running on the Gemini free tier.
    """

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()

    if not api_key:
        try:
            api_key = str(
                st.secrets.get("GEMINI_API_KEY", "")
            ).strip()
        except Exception:
            api_key = ""

    if not api_key:
        st.error(
            "Gemini API key was not found. "
            "Please configure GEMINI_API_KEY in the environment "
            "or Streamlit Secrets."
        )
        return None

    try:
        llm = LLM(
            model="gemini/gemini-3.8-flash",
            api_key=api_key
        )

        return llm

    except Exception as exc:
        st.error(
            f"Gemini LLM configuration error: {exc}"
        )
        return None


# ============================================================
# COMMON FREE-TIER AGENT SETTINGS
# ============================================================

FREE_TIER_AGENT_SETTINGS = {
    "verbose": False,
    "allow_delegation": False,

    # One execution cycle only.
    # This prevents unnecessary internal LLM calls.
    "max_iter": 1,

    # Do not automatically retry failed API requests.
    # Important for Gemini free-tier quota protection.
    "max_retry_limit": 0,

    # Conservative per-agent rate limit.
    "max_rpm": 1,

    # Allow CrewAI to reuse identical results where possible.
    "cache": True,
}


# ============================================================
# AGENT 1 — INCIDENT TRIAGE
# ============================================================

def create_triage_agent(llm):

    return Agent(
        role="Operational Incident Triage Agent",

        goal=(
            "Analyze the user's incident description and extract "
            "only the operational facts, incident type, affected "
            "process, material or activity, and key search terms."
        ),

        backstory=(
            "You are the first operational intake specialist. "
            "You do not invent facts, procedures, departments, "
            "or decisions. You only structure the information "
            "provided by the user."
        ),

        llm=llm,
        **FREE_TIER_AGENT_SETTINGS
    )


# ============================================================
# AGENT 2 — SOP COMPLIANCE
# ============================================================

def create_compiler_agent(llm):

    return Agent(
        role="SOP Compliance and Action Compiler",

        goal=(
            "Convert the retrieved SOP evidence into a short, "
            "sequential and evidence-grounded operational checklist."
        ),

        backstory=(
            "You are a strict SOP compliance specialist. "
            "Use only the supplied SOP evidence. "
            "Never invent a procedure, policy, approval, "
            "responsibility or requirement that is not supported "
            "by the retrieved evidence."
        ),

        llm=llm,
        **FREE_TIER_AGENT_SETTINGS
    )


# ============================================================
# AGENT 3 — DEPARTMENT ROUTING
# ============================================================

def create_router_agent(llm):

    return Agent(
        role="Cross-Functional Department Routing Agent",

        goal=(
            "Map each approved operational action to the most "
            "appropriate responsible function and identify "
            "where human verification is required."
        ),

        backstory=(
            "You are an organizational workflow specialist. "
            "You assign responsibilities based only on the "
            "action and available evidence. If responsibility "
            "cannot be established confidently, mark it as "
            "Human Verification Required."
        ),

        llm=llm,
        **FREE_TIER_AGENT_SETTINGS
    )


# ============================================================
# AGENT 4 — EXECUTIVE ACTION BRIEF
# ============================================================

def create_automation_agent(llm):

    return Agent(
        role="Executive Action Brief and Workflow Preparation Agent",

        goal=(
            "Combine the validated incident, SOP actions and "
            "department routing into a concise executive action "
            "brief ready for human review and approval."
        ),

        backstory=(
            "You are a corporate operations documentation specialist. "
            "Prepare a clear action brief containing the incident, "
            "applicable SOP actions, responsible functions, required "
            "documents, decision points and the mandatory human "
            "approval gate. Do not execute business transactions."
        ),

        llm=llm,
        **FREE_TIER_AGENT_SETTINGS
    )
```

### Why these settings matter

The important part is:

```python
"max_iter": 1,
"max_retry_limit": 0,
"max_rpm": 1,
```

CrewAI documents these as execution controls for limiting agent iterations, API request rate, and retries.

So our intended execution becomes:

**1 incident → 1 call Agent 1 → 1 call Agent 2 → 1 call Agent 3 → 1 call Agent 4**

Approximately **4 Gemini requests per workflow** rather than allowing the agents to repeatedly reason/retry.

---

## 2. Simplify `tasks.py`

I also recommend shortening the task prompts. This reduces token consumption and makes the workflow easier for Gemini to follow.

```python
from crewai import Task


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
            "Analyze this operational incident:\n\n"
            f"{incident_description}\n\n"

            "Extract only:\n"
            "1. Incident type\n"
            "2. Affected process/material/activity\n"
            "3. Key operational facts\n"
            "4. Search terms\n\n"

            "Do not invent missing information."
        ),

        expected_output=(
            "A concise structured incident summary containing "
            "incident type, affected process and search terms."
        ),

        agent=triage_worker
    )


    # ========================================================
    # TASK 2 — SOP COMPLIANCE
    # ========================================================

    task_2_compile = Task(
        description=(
            "Use the incident information from the previous agent "
            "and the following retrieved SOP evidence.\n\n"

            f"{context_chunks}\n\n"

            "Create a numbered action checklist.\n\n"

            "Rules:\n"
            "- Use only the supplied SOP evidence.\n"
            "- Do not invent procedures.\n"
            "- Do not add unsupported approvals.\n"
            "- Clearly identify when evidence is insufficient."
        ),

        expected_output=(
            "A concise numbered SOP-grounded action checklist "
            "with evidence-based actions only."
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

            "For each action identify:\n"
            "- Responsible department/function\n"
            "- Required supporting activity\n"
            "- Human Verification Required where responsibility "
            "is not supported by the available evidence.\n\n"

            "Do not invent organizational responsibilities."
        ),

        expected_output=(
            "A concise action-to-department routing table."
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
            "1. Incident\n"
            "2. Applicable SOP evidence\n"
            "3. Required actions\n"
            "4. Responsible functions\n"
            "5. Required documents\n"
            "6. Decision/approval point\n"
            "7. Human approval requirement\n\n"

            "The system must not claim that any business transaction "
            "has been executed."
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
```

---

# 3. Important: do NOT run `gemini_test.py` before every demo

Your direct Gemini test already proved something important:

**Gemini 3.8 Flash authentication works.**

So we don't need to consume another free-tier request every time we test the application.

Google confirms Gemini 3.8 Flash is currently a GA model and has a free tier.

For our hackathon:

**Do this once:**

`gemini_test.py` → confirms API

Then stop using it.

---

# 4. We should also change the application wording

Your current screen still says:

> Technology: CrewAI + Gemini 2.5 Flash + Local TF-IDF RAG

That is now incorrect.

Change it to:

**Technology: CrewAI + Gemini 3.8 Flash + Local TF-IDF RAG**

And change:

> autonomous

where it implies the system independently executes business activity.

I recommend:

> **Human-Governed Multi-Agent Supply Chain Contingency Planner**

This is actually stronger for a business/enterprise hackathon because your system generates the recommended workflow but a human remains responsible for approval.

---

# 5. One more important change: quota protection

We should modify `app.py` so that a Gemini 429 doesn't display a giant red Python traceback.

Instead, the user should see something like:

> **Gemini Free-Tier Rate Limit Reached**
>
> The free Gemini API quota is temporarily busy. No paid service is being used.
>
> Please wait briefly before running the workflow again.

This is particularly important because your hackathon judges should see a **professional application**, not a raw exception.

---

# 6. Our final free architecture

The architecture will now be:

```text
                USER
                  │
                  ▼
        ┌───────────────────┐
        │ Incident Input    │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ LOCAL TF-IDF RAG  │
        │ No API Cost       │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Agent 1           │
        │ Incident Triage   │
        │ 1 Gemini Call     │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Agent 2           │
        │ SOP Compliance    │
        │ 1 Gemini Call     │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Agent 3           │
        │ Department Route  │
        │ 1 Gemini Call     │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ Agent 4           │
        │ Executive Brief   │
        │ 1 Gemini Call     │
        └─────────┬─────────┘
                  │
                  ▼
        ┌───────────────────┐
        │ HUMAN APPROVAL    │
        │                   │
        │ Approve / Reject  │
        └───────────────────┘
```

**Maximum intended Gemini calls: ~4 per workflow.**

That preserves the actual **multi-agent** story rather than replacing it with a single Gemini call.

### One caution

Your current free quota is only **5 requests/minute**, so even four agent calls leave very little room for extra requests. The new configuration reduces unnecessary calls, but it cannot make Google's free quota unlimited. Google explicitly describes the free tier as rate-limited, while paid tiers provide higher rate limits.

**Next, I would modify `app.py` itself** to add the 429 protection, correct the visible HTML formatting, update the Gemini 3.8 label, and make the workflow execute safely within this four-call free-tier design. For that exact change, I need the **current `app.py`** you are running, because the complete latest version isn't preserved in the conversation context.

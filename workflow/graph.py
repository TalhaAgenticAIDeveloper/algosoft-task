"""
Multi-Agent LangGraph Workflow for Assessment Generation

Architecture:
  Manager Agent → routes to specialised agents via conditional edges.

Agents:
  1. Manager              – intent detection & orchestration
  2. Greeting             – handles greetings / casual messages
  3. Topic Extractor      – extracts topics from free-text
  4. LO Retriever         – semantic search over LO embeddings
  5. Chunk Retriever      – fetches content for selected LOs
  6. LO Browser           – shows all LOs when no topic is given
  7. Assessment Generator – creates the final assessment
  8. Rejection            – handles rejection / re-explain
  9. Regenerate           – improves a previously generated assessment

All nodes read / write the shared AssessmentState.
FastAPI contract is unchanged: build_graph() → compiled StateGraph.
"""

import re
import json
from typing import Dict, List

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from state import AssessmentState
from vectorstore import search_los

load_dotenv()

# ──────────────────────────────────────────────
# LLM
# ──────────────────────────────────────────────

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

# ──────────────────────────────────────────────
# Data files
# ──────────────────────────────────────────────

with open("lo_with_chunks.json", "r", encoding="utf-8") as _f:
    LO_DATA: List[dict] = json.load(_f)

with open("chunks_clean.json", "r", encoding="utf-8") as _f:
    CHUNK_DATA: List[dict] = json.load(_f)

CHUNK_MAP: Dict[str, str] = {c["chunk_id"]: c["content"] for c in CHUNK_DATA}

# Quick lookup: lo_id → lo record
LO_LOOKUP: Dict[str, dict] = {lo["lo_id"]: lo for lo in LO_DATA}


# ═══════════════════════════════════════════════
#  AGENT 0 — MANAGER (intent detection + routing)
# ═══════════════════════════════════════════════

def manager_agent(state: AssessmentState) -> dict:
    """
    Classifies the teacher's latest message into an intent.
    Stage-aware: honours stages that expect specific follow-ups
    so we don't re-classify mid-conversation.
    """
    teacher_input = state.get("teacher_input", "")
    current_stage = state.get("stage", "start")

    # ── Stage-based shortcuts (re-entry guards) ─────────────

    if current_stage == "waiting_lo_selection":
        # We are waiting for the teacher to provide LO IDs
        lo_ids_found = re.findall(r"\d+\.\d+\.\d+\.\d+\.\d+", teacher_input)
        if lo_ids_found:
            return {**state, "intent": "lo_selection"}
        # If no LO pattern found, treat as rejection / re-explain
        return {**state, "intent": "rejection"}

    if current_stage == "waiting_content_confirmation":
        lower = teacher_input.strip().lower()
        if any(w in lower for w in [
            "yes", "accept", "proceed", "sure", "ok",
            "generate", "go ahead", "yeah", "yep", "do it"
        ]):
            return {**state, "intent": "accept_content"}
        if any(w in lower for w in [
            "no", "reject", "not correct", "wrong",
            "change", "different", "nope"
        ]):
            return {**state, "intent": "rejection"}
        # Ambiguous — ask the LLM
        prompt = (
            "The teacher was asked whether to generate an assessment from shown content.\n"
            f"Teacher reply: \"{teacher_input}\"\n"
            "Classify as exactly one label: accept_content | rejection\n"
            "Return only the label."
        )
        label = llm.invoke(prompt).content.strip().lower()
        if label not in ("accept_content", "rejection"):
            label = "accept_content"
        return {**state, "intent": label}

    if current_stage == "waiting_regeneration":
        # Teacher asked to regenerate / improve the last assessment
        return {**state, "intent": "accept_content"}

    # ── General intent classification via LLM ───────────────

    prompt = f"""You are an intent classifier for an Assessment Creation Agent used by teachers.

Classify the teacher message into exactly ONE of these labels:
- greeting
- topic_assessment_request
- broad_assessment_request
- weak_student_topic
- regenerate_assessment
- lo_selection
- rejection

Rules:
- "greeting" = hello, hi, how are you, etc.
- "topic_assessment_request" = teacher mentions specific topics for assessment
- "broad_assessment_request" = teacher says "create assessment" but gives NO specific topic
- "weak_student_topic" = teacher mentions students are weak in specific topics (treat as topic request)
- "regenerate_assessment" = teacher is unhappy with the last assessment or wants to redo it
- "lo_selection" = message contains LO IDs like 6.5.3.1.1
- "rejection" = teacher rejects or says no

Return ONLY the label. No explanation.

Teacher message:
\"{teacher_input}\"
"""
    intent = llm.invoke(prompt).content.strip().lower()

    allowed = {
        "greeting",
        "topic_assessment_request",
        "broad_assessment_request",
        "weak_student_topic",
        "regenerate_assessment",
        "lo_selection",
        "rejection",
    }
    if intent not in allowed:
        intent = "topic_assessment_request"

    # weak_student_topic is handled identically to topic_assessment_request
    if intent == "weak_student_topic":
        intent = "topic_assessment_request"

    return {**state, "intent": intent}


# ═══════════════════════════════════════════════
#  AGENT 1 — GREETING
# ═══════════════════════════════════════════════

def greeting_agent(state: AssessmentState) -> dict:
    """Responds to greetings / casual messages."""
    prompt = (
        "You are a helpful assistant for teachers who want to create assessments.\n"
        "Respond warmly to the teacher's greeting and briefly explain what you can do:\n"
        "- Help create assessments based on topics or Learning Outcomes.\n"
        "- Browse available curriculum Learning Outcomes.\n"
        "- Generate aligned assessments with MCQs, short & long questions.\n\n"
        f"Teacher said: \"{state['teacher_input']}\""
    )
    response = llm.invoke(prompt).content

    msgs = list(state.get("messages", []))
    msgs.append(AIMessage(content=response))

    return {**state, "messages": msgs, "stage": "greeting_done"}


# ═══════════════════════════════════════════════
#  AGENT 2 — TOPIC EXTRACTOR
# ═══════════════════════════════════════════════

def topic_extractor_agent(state: AssessmentState) -> dict:
    """Extracts discrete learning topics from free-text input."""
    prompt = f"""Extract the main learning / assessment topics from the teacher's message.
Return ONLY a comma-separated list of short topic phrases. No numbering, no explanation.

Teacher message:
\"{state['teacher_input']}\"
"""
    raw = llm.invoke(prompt).content.strip()
    topics = [
        t.strip().strip('"').strip("'")
        for t in raw.split(",")
        if len(t.strip()) > 2
    ]

    return {**state, "extracted_topics": topics, "stage": "topics_extracted"}


# ═══════════════════════════════════════════════
#  AGENT 3 — LO RETRIEVER (embedding search)
# ═══════════════════════════════════════════════

def lo_retriever_agent(state: AssessmentState) -> dict:
    """
    For each extracted topic, run semantic search via vectorstore.search_los()
    and compile the results into a user-friendly message.
    """
    topics = state.get("extracted_topics", [])
    topic_matches = []
    seen_lo_ids: set = set()

    for topic in topics:
        results = search_los(topic, k=5)
        unique = []
        for r in results:
            if r["lo_id"] not in seen_lo_ids:
                seen_lo_ids.add(r["lo_id"])
                unique.append(r)
        topic_matches.append({"topic": topic, "matched_los": unique})

    # Build human-readable message
    lines = ["Here are the relevant Learning Outcomes:\n"]
    counter = 1
    for tm in topic_matches:
        for lo in tm["matched_los"]:
            lines.append(
                f"{counter}. **LO_ID:** {lo['lo_id']}\n"
                f"   **Domain:** {lo['domain']}\n"
                f"   **Subdomain:** {lo['subdomain']}\n"
                f"   **Description:** {lo['description']}\n"
            )
            counter += 1
    lines.append(
        "\nPlease mention the LO IDs (comma separated) that you want to include in the assessment."
    )
    text = "\n".join(lines)

    msgs = list(state.get("messages", []))
    msgs.append(AIMessage(content=text))

    return {
        **state,
        "messages": msgs,
        "topic_matches": topic_matches,
        "stage": "waiting_lo_selection",
    }


# ═══════════════════════════════════════════════
#  AGENT 4 — CHUNK RETRIEVER
# ═══════════════════════════════════════════════

def chunk_retriever_agent(state: AssessmentState) -> dict:
    """
    Given selected LO IDs from teacher input:
      1. Find chunk IDs from lo_with_chunks.json
      2. Fetch actual content from chunks_clean.json
      3. Present content and ask for confirmation.
    """
    teacher_input = state.get("teacher_input", "")
    selected = re.findall(r"\d+\.\d+\.\d+\.\d+\.\d+", teacher_input)

    if not selected:
        msgs = list(state.get("messages", []))
        msgs.append(AIMessage(
            content="I couldn't find valid LO IDs in your message. "
                    "Please provide them in the format like 6.5.3.1.1, 6.5.3.1.2"
        ))
        return {**state, "messages": msgs, "stage": "waiting_lo_selection"}

    # Collect chunk IDs for every selected LO
    chunk_ids: List[str] = []
    valid_selected: List[str] = []

    for lo_id in selected:
        lo = LO_LOOKUP.get(lo_id)
        if lo:
            valid_selected.append(lo_id)
            for cid in lo.get("chunks", []):
                if cid not in chunk_ids:
                    chunk_ids.append(cid)

    # Fetch content
    chunk_contents = [CHUNK_MAP[cid] for cid in chunk_ids if cid in CHUNK_MAP]

    # Build display message — show a preview
    preview = "\n\n---\n\n".join(chunk_contents[:5])
    if len(chunk_contents) > 5:
        preview += f"\n\n... and {len(chunk_contents) - 5} more sections."

    text = (
        f"Here is the extracted learning content for LOs: {', '.join(valid_selected)}\n\n"
        f"{preview}\n\n"
        "Do you want to create an assessment based on this content?"
    )

    msgs = list(state.get("messages", []))
    msgs.append(AIMessage(content=text))

    return {
        **state,
        "messages": msgs,
        "selected_los": valid_selected,
        "chunk_ids": chunk_ids,
        "chunk_contents": chunk_contents,
        "stage": "waiting_content_confirmation",
    }


# ═══════════════════════════════════════════════
#  AGENT 5 — LO BROWSER (broad request, no topic)
# ═══════════════════════════════════════════════

def lo_browser_agent(state: AssessmentState) -> dict:
    """
    Displays ALL available LOs grouped by Domain → Subdomain.
    Asks the teacher to select LO IDs.
    """
    grouped: Dict[str, Dict[str, list]] = {}
    for lo in LO_DATA:
        d = lo["domain"]
        s = lo["subdomain"]
        grouped.setdefault(d, {}).setdefault(s, []).append(lo)

    lines = ["Here are all available Learning Outcomes:\n"]
    counter = 1
    for domain in sorted(grouped):
        lines.append(f"### {domain}")
        for subdomain in sorted(grouped[domain]):
            lines.append(f"  **{subdomain}**")
            for lo in grouped[domain][subdomain]:
                lines.append(
                    f"    {counter}. {lo['lo_id']} — {lo['description']}"
                )
                counter += 1
        lines.append("")

    lines.append(
        "\nPlease mention the LO IDs (comma separated) that you want to "
        "include in the assessment."
    )
    text = "\n".join(lines)

    msgs = list(state.get("messages", []))
    msgs.append(AIMessage(content=text))

    return {
        **state,
        "messages": msgs,
        "grouped_los": grouped,
        "stage": "waiting_lo_selection",
    }


# ═══════════════════════════════════════════════
#  AGENT 6 — ASSESSMENT GENERATOR
# ═══════════════════════════════════════════════

def assessment_generator_agent(state: AssessmentState) -> dict:
    """
    Generates a complete assessment using the retrieved chunk content
    aligned with the selected LOs.
    """
    content = "\n\n".join(state.get("chunk_contents", []))
    selected = state.get("selected_los", [])

    lo_descriptions = ""
    for lo_id in selected:
        lo = LO_LOOKUP.get(lo_id)
        if lo:
            lo_descriptions += f"- {lo_id}: {lo['description']}\n"

    prompt = f"""You are an expert assessment creator for teachers.

Using the learning content and Learning Outcomes below, create a comprehensive assessment.

**Selected Learning Outcomes:**
{lo_descriptions}

**Learning Content:**
{content}

**Assessment Requirements:**
Include the following sections:
1. Multiple Choice Questions (MCQs) — at least 5
2. Short Answer Questions — at least 4
3. Long Answer Questions — at least 2
4. Application-Based Questions — at least 2

Each question must be clearly aligned with one or more of the selected Learning Outcomes.
Label each question with the relevant LO ID(s).
Provide an answer key at the end.
"""
    response = llm.invoke(prompt).content

    msgs = list(state.get("messages", []))
    msgs.append(AIMessage(content=response))

    return {
        **state,
        "messages": msgs,
        "generated_assessment": response,
        "last_assessment": response,
        "stage": "assessment_done",
    }


# ═══════════════════════════════════════════════
#  AGENT 7 — REJECTION / RE-EXPLAIN HANDLER
# ═══════════════════════════════════════════════

def rejection_agent(state: AssessmentState) -> dict:
    """
    The teacher rejected content or wants to start over.
    Clears transient state and asks the teacher to re-explain.
    """
    msgs = list(state.get("messages", []))
    msgs.append(
        AIMessage(
            content=(
                "No problem! Let's start fresh.\n\n"
                "Could you please re-explain the topic or requirement you'd like "
                "the assessment to cover? You can also just say "
                "\"create assessment\" to browse all available Learning Outcomes."
            )
        )
    )
    return {
        **state,
        "messages": msgs,
        "intent": "",
        "extracted_topics": [],
        "topic_matches": [],
        "selected_los": [],
        "chunk_ids": [],
        "chunk_contents": [],
        "stage": "start",
    }


# ═══════════════════════════════════════════════
#  AGENT 8 — REGENERATE / IMPROVE ASSESSMENT
# ═══════════════════════════════════════════════

def regenerate_assessment_agent(state: AssessmentState) -> dict:
    """
    Teacher is unhappy with the last assessment.
    Uses last_assessment + teacher feedback to produce an improved version.
    """
    last = state.get("last_assessment", "")
    feedback = state.get("teacher_input", "")

    if not last:
        msgs = list(state.get("messages", []))
        msgs.append(
            AIMessage(
                content=(
                    "I don't have a previous assessment on record. "
                    "Would you like to create a new one? "
                    "Please provide the topics or say \"create assessment\"."
                )
            )
        )
        return {**state, "messages": msgs, "stage": "start"}

    prompt = f"""You are an expert assessment creator.

The teacher was not satisfied with the previous assessment.

**Previous Assessment:**
{last}

**Teacher Feedback:**
{feedback}

Please regenerate an improved assessment addressing the feedback.
Keep the same structure (MCQs, Short Questions, Long Questions, Application-Based Questions)
but improve quality, clarity, and alignment with curriculum.
Provide an answer key at the end.
"""
    response = llm.invoke(prompt).content

    msgs = list(state.get("messages", []))
    msgs.append(AIMessage(content=response))

    return {
        **state,
        "messages": msgs,
        "generated_assessment": response,
        "last_assessment": response,
        "stage": "assessment_done",
    }


# ═══════════════════════════════════════════════
#  ROUTING FUNCTION
# ═══════════════════════════════════════════════

def route_from_manager(state: AssessmentState) -> str:
    """Conditional edge after manager_agent — routes to the correct agent."""
    intent = state.get("intent", "")

    routing_map = {
        "greeting":                 "greeting_agent",
        "topic_assessment_request": "topic_extractor_agent",
        "broad_assessment_request": "lo_browser_agent",
        "lo_selection":             "chunk_retriever_agent",
        "accept_content":           "assessment_generator_agent",
        "rejection":                "rejection_agent",
        "regenerate_assessment":    "regenerate_assessment_agent",
    }
    return routing_map.get(intent, "greeting_agent")


# ═══════════════════════════════════════════════
#  BUILD GRAPH
# ═══════════════════════════════════════════════

def build_graph():
    """
    Constructs and compiles the LangGraph workflow.

    Graph topology:

        ┌──────────────┐
        │ manager_agent │  (entry point — every user message starts here)
        └──────┬───────┘
               │  conditional routing based on intent
               ├──→ greeting_agent ──→ END
               ├──→ topic_extractor_agent ──→ lo_retriever_agent ──→ END
               ├──→ lo_browser_agent ──→ END
               ├──→ chunk_retriever_agent ──→ END
               ├──→ assessment_generator_agent ──→ END
               ├──→ rejection_agent ──→ END
               └──→ regenerate_assessment_agent ──→ END

    Each END returns control to FastAPI, which waits for the next user
    message and re-invokes the graph from manager_agent.
    """
    workflow = StateGraph(AssessmentState)

    # ── Register nodes ──────────────────────────
    workflow.add_node("manager_agent",               manager_agent)
    workflow.add_node("greeting_agent",              greeting_agent)
    workflow.add_node("topic_extractor_agent",       topic_extractor_agent)
    workflow.add_node("lo_retriever_agent",          lo_retriever_agent)
    workflow.add_node("chunk_retriever_agent",       chunk_retriever_agent)
    workflow.add_node("lo_browser_agent",            lo_browser_agent)
    workflow.add_node("assessment_generator_agent",  assessment_generator_agent)
    workflow.add_node("rejection_agent",             rejection_agent)
    workflow.add_node("regenerate_assessment_agent", regenerate_assessment_agent)

    # ── Entry point ─────────────────────────────
    workflow.set_entry_point("manager_agent")

    # ── Conditional routing from manager ────────
    workflow.add_conditional_edges(
        "manager_agent",
        route_from_manager,
        {
            "greeting_agent":              "greeting_agent",
            "topic_extractor_agent":       "topic_extractor_agent",
            "lo_browser_agent":            "lo_browser_agent",
            "chunk_retriever_agent":       "chunk_retriever_agent",
            "assessment_generator_agent":  "assessment_generator_agent",
            "rejection_agent":             "rejection_agent",
            "regenerate_assessment_agent": "regenerate_assessment_agent",
        },
    )

    # ── Topic extraction → LO retrieval (subgraph chain) ──
    workflow.add_edge("topic_extractor_agent", "lo_retriever_agent")

    # ── Terminal edges (return to user / FastAPI) ──
    workflow.add_edge("greeting_agent",              END)
    workflow.add_edge("lo_retriever_agent",          END)
    workflow.add_edge("chunk_retriever_agent",       END)
    workflow.add_edge("lo_browser_agent",            END)
    workflow.add_edge("assessment_generator_agent",  END)
    workflow.add_edge("rejection_agent",             END)
    workflow.add_edge("regenerate_assessment_agent", END)

    return workflow.compile()

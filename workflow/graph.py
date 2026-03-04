"""
LangGraph workflow for Part One
Handles:
- Intent detection
- Topic extraction
- LO retrieval
- Grouping
- Presentation
"""

import re
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from state import AssessmentState
from vectorstore import search_los


# -------------------------
# NODE 1: Intent Detection
# -------------------------

def detect_intent(state: AssessmentState):
    """
    Determines teacher intent
    """

    text = state["teacher_input"].lower()

    if any(greet in text for greet in ["hi", "hello", "assalam", "hey"]):
        state["intent"] = "greeting"
        state["stage"] = "greeting"
    else:
        state["intent"] = "assessment_request"
        state["stage"] = "topic_extraction"

    return state


# -------------------------
# NODE 2: Greeting
# -------------------------

def greeting_node(state: AssessmentState):
    """
    Respond to greeting
    """

    response = (
        "Hello 👋\n\n"
        "I can help you create an assessment.\n"
        "Please tell me the topics or areas you want to assess."
    )

    state["messages"].append(AIMessage(content=response))
    state["stage"] = "completed"

    return state


# -------------------------
# NODE 3: Topic Extraction
# -------------------------

def extract_topics(state: AssessmentState):
    """
    Extract multiple topics intelligently
    Splits by comma, 'and', '&'
    """

    text = state["teacher_input"]

    topics = re.split(r",|and|&", text)
    cleaned_topics = [t.strip() for t in topics if len(t.strip()) > 2]

    state["extracted_topics"] = cleaned_topics
    state["stage"] = "lo_retrieval"

    return state


# -------------------------
# NODE 4: LO Retrieval
# -------------------------

def retrieve_los(state: AssessmentState):
    """
    For each extracted topic:
    - Perform semantic search
    """

    topic_matches = []

    for topic in state["extracted_topics"]:
        results = search_los(topic, k=5)

        topic_matches.append({
            "topic": topic,
            "matched_los": results
        })

    state["topic_matches"] = topic_matches
    state["stage"] = "grouping"

    return state


# -------------------------
# NODE 5: Group by Domain/Subdomain
# -------------------------

def group_los(state: AssessmentState):
    """
    Merge all matched LOs and group by:
    Domain → Subdomain
    """

    grouped = {}

    for topic_data in state["topic_matches"]:
        for lo in topic_data["matched_los"]:

            domain = lo["domain"]
            subdomain = lo["subdomain"]

            if domain not in grouped:
                grouped[domain] = {}

            if subdomain not in grouped[domain]:
                grouped[domain][subdomain] = []

            # Avoid duplicate LO entries
            if lo not in grouped[domain][subdomain]:
                grouped[domain][subdomain].append(lo)

    state["grouped_los"] = grouped
    state["stage"] = "presentation"

    return state


# -------------------------
# NODE 6: Presentation
# -------------------------

def present_los(state: AssessmentState):
    """
    Present structured LO grouping
    """

    response = "Based on your input, the following areas are relevant:\n\n"

    for domain, subdomains in state["grouped_los"].items():
        response += f"📘 Domain: {domain}\n"

        for subdomain, los in subdomains.items():
            response += f"  📂 Subdomain: {subdomain}\n"

            for lo in los:
                response += (
                    f"    • {lo['lo_id']} – "
                    f"{lo['description']}\n"
                )

        response += "\n"

    response += "Please tell me which Learning Outcomes you want to include."

    state["messages"].append(AIMessage(content=response))
    state["stage"] = "waiting_selection"

    return state


# -------------------------
# BUILD GRAPH
# -------------------------

def build_graph():

    workflow = StateGraph(AssessmentState)

    workflow.add_node("intent", detect_intent)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("extract_topics", extract_topics)
    workflow.add_node("retrieve_los", retrieve_los)
    workflow.add_node("group_los", group_los)
    workflow.add_node("present", present_los)

    workflow.set_entry_point("intent")

    # Conditional routing
    workflow.add_conditional_edges(
        "intent",
        lambda state: state["intent"],
        {
            "greeting": "greeting",
            "assessment_request": "extract_topics"
        }
    )

    workflow.add_edge("extract_topics", "retrieve_los")
    workflow.add_edge("retrieve_los", "group_los")
    workflow.add_edge("group_los", "present")
    workflow.add_edge("present", END)

    return workflow.compile()
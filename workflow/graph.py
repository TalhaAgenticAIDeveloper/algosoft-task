# """
# LangGraph workflow for Part One
# Handles:
# - Intent detection
# - Topic extraction
# - LO retrieval
# - Grouping
# - Presentation
# """

# import re
# from langgraph.graph import StateGraph, END
# from langchain_core.messages import AIMessage
# from state import AssessmentState
# from vectorstore import search_los


# # -------------------------
# # NODE 1: Intent Detection
# # -------------------------

# def detect_intent(state: AssessmentState):
#     """
#     Determines teacher intent
#     """

#     text = state["teacher_input"].lower()

#     if any(greet in text for greet in ["hi", "hello", "assalam", "hey"]):
#         state["intent"] = "greeting"
#         state["stage"] = "greeting"
#     else:
#         state["intent"] = "assessment_request"
#         state["stage"] = "topic_extraction"

#     return state


# # -------------------------
# # NODE 2: Greeting
# # -------------------------

# def greeting_node(state: AssessmentState):
#     """
#     Respond to greeting
#     """

#     response = (
#         "Hello 👋\n\n"
#         "I can help you create an assessment.\n"
#         "Please tell me the topics or areas you want to assess."
#     )

#     state["messages"].append(AIMessage(content=response))
#     state["stage"] = "completed"

#     return state


# # -------------------------
# # NODE 3: Topic Extraction
# # -------------------------

# def extract_topics(state: AssessmentState):
#     """
#     Extract multiple topics intelligently
#     Splits by comma, 'and', '&'
#     """

#     text = state["teacher_input"]

#     topics = re.split(r",|and|&", text)
#     cleaned_topics = [t.strip() for t in topics if len(t.strip()) > 2]

#     state["extracted_topics"] = cleaned_topics
#     state["stage"] = "lo_retrieval"

#     return state


# # -------------------------
# # NODE 4: LO Retrieval
# # -------------------------

# def retrieve_los(state: AssessmentState):
#     """
#     For each extracted topic:
#     - Perform semantic search
#     """

#     topic_matches = []

#     for topic in state["extracted_topics"]:
#         results = search_los(topic, k=5)

#         topic_matches.append({
#             "topic": topic,
#             "matched_los": results
#         })

#     state["topic_matches"] = topic_matches
#     state["stage"] = "grouping"

#     return state


# # -------------------------
# # NODE 5: Group by Domain/Subdomain
# # -------------------------

# def group_los(state: AssessmentState):
#     """
#     Merge all matched LOs and group by:
#     Domain → Subdomain
#     """

#     grouped = {}

#     for topic_data in state["topic_matches"]:
#         for lo in topic_data["matched_los"]:

#             domain = lo["domain"]
#             subdomain = lo["subdomain"]

#             if domain not in grouped:
#                 grouped[domain] = {}

#             if subdomain not in grouped[domain]:
#                 grouped[domain][subdomain] = []

#             # Avoid duplicate LO entries
#             if lo not in grouped[domain][subdomain]:
#                 grouped[domain][subdomain].append(lo)

#     state["grouped_los"] = grouped
#     state["stage"] = "presentation"

#     return state


# # -------------------------
# # NODE 6: Presentation
# # -------------------------

# def present_los(state: AssessmentState):
#     """
#     Present structured LO grouping
#     """

#     response = "Based on your input, the following areas are relevant:\n\n"

#     for domain, subdomains in state["grouped_los"].items():
#         response += f"📘 Domain: {domain}\n"

#         for subdomain, los in subdomains.items():
#             response += f"  📂 Subdomain: {subdomain}\n"

#             for lo in los:
#                 response += (
#                     f"    • {lo['lo_id']} – "
#                     f"{lo['description']}\n"
#                 )

#         response += "\n"

#     response += "Please tell me which Learning Outcomes you want to include."

#     state["messages"].append(AIMessage(content=response))
#     state["stage"] = "waiting_selection"

#     return state


# # -------------------------
# # BUILD GRAPH
# # -------------------------

# def build_graph():

#     workflow = StateGraph(AssessmentState)

#     workflow.add_node("intent", detect_intent)
#     workflow.add_node("greeting", greeting_node)
#     workflow.add_node("extract_topics", extract_topics)
#     workflow.add_node("retrieve_los", retrieve_los)
#     workflow.add_node("group_los", group_los)
#     workflow.add_node("present", present_los)

#     workflow.set_entry_point("intent")

#     # Conditional routing
#     workflow.add_conditional_edges(
#         "intent",
#         lambda state: state["intent"],
#         {
#             "greeting": "greeting",
#             "assessment_request": "extract_topics"
#         }
#     )

#     workflow.add_edge("extract_topics", "retrieve_los")
#     workflow.add_edge("retrieve_los", "group_los")
#     workflow.add_edge("group_los", "present")
#     workflow.add_edge("present", END)

#     return workflow.compile()






"""
Advanced LangGraph Workflow for Assessment Agent (Part 1 Enhanced)

Features:
- LLM-based intent detection (Groq)
- Subgraph routing
- Human-in-the-loop refinement
- Stage-aware flow control
"""

import re
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage
from state import AssessmentState
from vectorstore import search_los
from dotenv import load_dotenv

load_dotenv()


# GROQ LLM
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0
)

# =========================================
# NODE 1: INTENT ROUTER (LLM BASED)
# =========================================

def detect_intent(state: AssessmentState):

    # If we are waiting for selection → don't re-detect
    if state["stage"] == "waiting_selection":
        state["intent"] = "selection"
        return state

    prompt = f"""
    You are an intent classifier for a teacher assessment system.

    Classify the teacher message into one of these:
    - greeting
    - broad_assessment_request (wants to create assessment but no topic)
    - topic_assessment_request (mentions topics)
    - refinement_request (rejecting or modifying previous suggestions)

    Only return the label.

    Teacher message:
    {state["teacher_input"]}
    """

    response = llm.invoke(prompt).content.strip().lower()

    state["intent"] = response
    return state


# =========================================
# SUBGRAPH 1: GREETING
# =========================================

def greeting_node(state: AssessmentState):

    response = llm.invoke(
        f"Respond politely to this greeting and ask how you can help:\n\n{state['teacher_input']}"
    ).content

    state["messages"].append(AIMessage(content=response))
    state["stage"] = "completed"

    return state


# =========================================
# SUBGRAPH 2: BROAD ASSESSMENT FLOW
# =========================================

def suggest_domains(state: AssessmentState):

    response = llm.invoke(
        """
        A teacher wants to create an assessment but did not specify topics.
        Suggest the major Domains and explain briefly what each includes.
        Keep it structured and professional.
        """
    ).content

    state["messages"].append(AIMessage(content=response))
    state["stage"] = "waiting_topic_input"

    return state


# =========================================
# SUBGRAPH 3: TOPIC EXTRACTION (LLM BASED)
# =========================================

def extract_topics(state: AssessmentState):

    prompt = f"""
    Extract key assessment topics from this teacher message.
    Return them as a comma separated list.

    Teacher message:
    {state["teacher_input"]}
    """

    topics = llm.invoke(prompt).content
    topic_list = [t.strip() for t in topics.split(",") if len(t.strip()) > 2]

    state["extracted_topics"] = topic_list
    state["stage"] = "lo_retrieval"

    return state


# =========================================
# LO RETRIEVAL
# =========================================

def retrieve_los(state: AssessmentState):

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


# =========================================
# GROUPING
# =========================================

def group_los(state: AssessmentState):

    grouped = {}

    for topic_data in state["topic_matches"]:
        for lo in topic_data["matched_los"]:

            domain = lo["domain"]
            subdomain = lo["subdomain"]

            grouped.setdefault(domain, {})
            grouped[domain].setdefault(subdomain, [])

            if lo not in grouped[domain][subdomain]:
                grouped[domain][subdomain].append(lo)

    state["grouped_los"] = grouped
    state["stage"] = "presentation"

    return state


# =========================================
# PRESENTATION (LLM ENHANCED)
# =========================================

def present_los(state: AssessmentState):

    structured_text = ""

    for domain, subdomains in state["grouped_los"].items():
        structured_text += f"\nDomain: {domain}\n"
        for subdomain, los in subdomains.items():
            structured_text += f"  Subdomain: {subdomain}\n"
            for lo in los:
                structured_text += f"    - {lo['lo_id']}: {lo['description']}\n"

    prompt = f"""
    Present the following curriculum structure in a clear structured format.
    Ask teacher to select LOs or request refinement.

    {structured_text}
    """

    response = llm.invoke(prompt).content

    state["messages"].append(AIMessage(content=response))
    state["stage"] = "waiting_selection"

    return state


# =========================================
# HUMAN IN THE LOOP: SELECTION
# =========================================

def handle_selection(state: AssessmentState):

    selected = re.findall(r"\d+\.\d+\.\d+\.\d+\.\d+", state["teacher_input"])

    state["selected_los"] = selected

    response = f"You selected: {', '.join(selected)}.\nWe will proceed with these."

    state["messages"].append(AIMessage(content=response))
    state["stage"] = "selection_done"

    return state


# =========================================
# HUMAN IN THE LOOP: REFINEMENT
# =========================================

def handle_refinement(state: AssessmentState):

    prompt = f"""
    The teacher is not satisfied and wants refinement.
    Analyze this feedback and suggest improved Learning Outcomes:

    Feedback:
    {state["teacher_input"]}
    """

    refined_topics = llm.invoke(prompt).content

    state["teacher_input"] = refined_topics
    state["stage"] = "topic_extraction"

    return state


# =========================================
# BUILD GRAPH
# =========================================

def build_graph():

    workflow = StateGraph(AssessmentState)

    # Nodes
    workflow.add_node("intent", detect_intent)
    workflow.add_node("greeting", greeting_node)
    workflow.add_node("suggest_domains", suggest_domains)
    workflow.add_node("extract_topics", extract_topics)
    workflow.add_node("retrieve_los", retrieve_los)
    workflow.add_node("group_los", group_los)
    workflow.add_node("present", present_los)
    workflow.add_node("selection", handle_selection)
    workflow.add_node("refinement", handle_refinement)

    workflow.set_entry_point("intent")

    # Routing
    workflow.add_conditional_edges(
        "intent",
        lambda state: state["intent"],
        {
            "greeting": "greeting",
            "broad_assessment_request": "suggest_domains",
            "topic_assessment_request": "extract_topics",
            "selection": "selection",
            "refinement_request": "refinement"
        }
    )

    # Flow edges
    workflow.add_edge("extract_topics", "retrieve_los")
    workflow.add_edge("retrieve_los", "group_los")
    workflow.add_edge("group_los", "present")
    workflow.add_edge("present", END)
    workflow.add_edge("selection", END)
    workflow.add_edge("greeting", END)
    workflow.add_edge("suggest_domains", END)

    return workflow.compile()
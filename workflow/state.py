"""
State definition for the Multi-Agent Assessment Workflow.

Stores all shared context across agents:
- conversation history
- extracted topics
- retrieved LOs
- selected LO IDs
- retrieved chunks
- generated assessments
- last_assessment
"""

from typing import List, Dict, TypedDict, Optional
from langchain_core.messages import BaseMessage


class TopicMatch(TypedDict):
    topic: str
    matched_los: List[dict]


class AssessmentState(TypedDict):
    # Session identifier
    session_id: str

    # Full conversation memory
    messages: List[BaseMessage]

    # Latest teacher input text
    teacher_input: str

    # Classified intent of the current message
    intent: str

    # Current workflow stage (used for re-entry routing)
    stage: str

    # Topics extracted from user message
    extracted_topics: List[str]

    # LO matches per topic (from embedding search)
    topic_matches: List[TopicMatch]

    # Grouped LOs by domain → subdomain
    grouped_los: Dict[str, Dict[str, List[dict]]]

    # LO IDs selected by the teacher
    selected_los: List[str]

    # Chunk IDs associated with selected LOs
    chunk_ids: List[str]

    # Actual text content of retrieved chunks
    chunk_contents: List[str]

    # The most recently generated assessment text
    generated_assessment: str

    # Persisted last assessment (for regeneration requests)
    last_assessment: str
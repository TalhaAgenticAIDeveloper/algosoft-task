"""
State definition for Part One of Assessment Agent
Handles:
- multi-topic extraction
- LO retrieval
- grouping
- session stage control
"""

from typing import List, Dict, TypedDict
from langchain_core.messages import BaseMessage


class TopicMatch(TypedDict):
    topic: str
    matched_los: List[dict]


class AssessmentState(TypedDict):
    # Unique session identifier
    session_id: str

    # Full conversation memory
    messages: List[BaseMessage]

    # Latest teacher input
    teacher_input: str

    # Intent classification
    intent: str

    # Extracted topics from teacher input
    extracted_topics: List[str]

    # LO matches per topic
    topic_matches: List[TopicMatch]

    # Final grouped LOs
    grouped_los: Dict[str, Dict[str, List[dict]]]

    # Current workflow stage
    stage: str
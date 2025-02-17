from typing import Annotated, List, Dict, TypedDict, Any
from pydantic import BaseModel
import operator
from langchain_core.messages import AnyMessage

class DepositionSection(BaseModel):
    """Structure for a deposition section"""
    name: str
    description: str
    content: str  # Contains relevant document quotes and their explanations

class ReportStateInput(TypedDict, total=False):
    user_provided_topic: str  # Deposition topic
    feedback_on_plan: str  # Feedback on the report structure
    accept_plan: bool  # Whether to accept the report plan

class ReportStateOutput(TypedDict):
    final_report: str  # Final markdown report
    status: Annotated[str, lambda a, b: b]  # Current workflow status

class ReportState(TypedDict, total=False):
    user_provided_topic: str  # Original deposition topic from user
    deposition_summary: str  # AI-generated summary of goals and strategy
    feedback_on_plan: str  # Feedback on the report structure from review
    accept_plan: bool  # Whether to accept or reject the report plan
    deposition_plan_prompt: List[AnyMessage]
    raw_sections: List[str]  # Raw sections before processing
    processed_sections: Annotated[List[DepositionSection], operator.add]  # Results from parallel section processing
    markdown_document: str  # markdown report
    status: Annotated[str, lambda a, b: b]  # Current workflow status

# Section processing sub-graph
class SectionState(TypedDict):
    raw_section: str
    processed_section: DepositionSection



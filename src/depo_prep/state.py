from typing import Annotated, List, Dict, TypedDict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import operator


class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Query for web search.")

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )

class SearchResult(BaseModel):
    """Structure for search results"""
    query: str
    section_name: str
    description: str
    results: List[Dict[str, str]]  # Note: all values must be strings

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
    raw_sections: List[str]  # Raw sections before processing
    processed_section: Annotated[List[Dict], operator.add]  # Results from parallel section processing
    markdown_document: str  # markdown report
    status: Annotated[str, lambda a, b: b]  # Current workflow status
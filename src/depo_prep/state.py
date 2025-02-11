from typing import Annotated, List, Dict, TypedDict, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum


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
    queries: List[str]
    investigation: bool = True

class ReportStateInput(TypedDict):
    topic: str  # Deposition topic
    feedback_on_report_plan: str  # Feedback on the report structure
    accept_report_plan: bool  # Whether to accept the report plan
    
class ReportStateOutput(TypedDict):
    final_report: str  # Final markdown report
    status: Annotated[str, lambda a, b: b]  # Current workflow status

class ReportState(TypedDict):
    topic: str  # Deposition topic    
    feedback_on_report_plan: str  # Feedback on the report structure from review
    accept_report_plan: bool  # Whether to accept or reject the report plan
    sections: List[Dict]  # List of structured sections
    search_results: List[Dict]  # Vector search results
    complaint_text: str  # Full text of the complaint
    final_report: str  # Final markdown report
    status: Annotated[str, lambda a, b: b]  # Current workflow status
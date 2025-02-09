from langchain_anthropic import ChatAnthropic 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from typing import Annotated, List, Dict, Any
import operator
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolExecutor
import json
import asyncio
from langchain_core.language_models.chat_models import BaseChatModel as ChatModel
import re

from src.report_maistro.state import ReportStateInput, ReportStateOutput, Sections, ReportState, SectionState, SectionOutputState, Queries, Section
from src.report_maistro.prompts import deposition_planner_query_writer_instructions, deposition_planner_instructions, query_writer_instructions, topic_writer_instructions, final_topic_writer_instructions
from src.report_maistro.configuration import Configuration
from src.report_maistro.utils import deduplicate_and_format_sources, format_sections, vector_db_search_async

# Initialize configuration and models
config = Configuration()
planner_model = ChatGoogleGenerativeAI(
    model=config.planner_model,
    temperature=0,
    convert_system_message_to_human=True
)
writer_model = ChatAnthropic(
    model=config.writer_model,
    temperature=0
)
json_formatter_model = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    temperature=0
)

# Constants
DEFAULT_NUM_QUERIES = 1

# Schema definitions
QUERY_SCHEMA = """{
    "queries": [
        {
            "query": "string",  // A simple phrase or sentence describing what to find - no boolean operators
            "rationale": "string",  // Why this query is useful
            "expected_findings": "string"  // What kind of documents we expect to find
        }
    ]
}"""

SECTIONS_SCHEMA = """{
    "sections": [
        {
            "name": "string - Name for this line of questioning",
            "description": "string - Brief overview of what you want to establish",
            "investigation": "boolean - Whether to search case documents for this topic",
            "content": null
        }
    ]
}"""

# Update query writer instructions to focus on semantic search
query_writer_instructions = """Generate {number_of_queries} search queries to find relevant documents about this topic.

Your response must be a valid JSON object with this structure:
{{
    "queries": [
        {{
            "query": "string",  // A simple phrase or sentence describing what to find - no boolean operators
            "rationale": "string",  // Why this query is useful
            "expected_findings": "string"  // What kind of documents we expect to find
        }}
    ]
}}

Example good queries:
- "Luther's experience with autonomous vehicles"
- "sensor system maintenance records"
- "accident investigation findings"

Do NOT use boolean operators (AND, OR). Just use natural language phrases that capture the key concepts.

Topic: {topic}"""

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text that may contain markdown or other formatting."""
    # First try direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON between ```json markers
    json_pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try to find anything that looks like a JSON object
    json_pattern = r"\{[\s\S]*\}"
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    print("\nFailed to extract JSON from text. Response content:")
    print(text)
    print("\nPlease ensure the model returns a valid JSON object.")
    raise ValueError("No valid JSON found in text")

# Nodes
async def generate_report_plan(state: ReportState) -> Dict[str, Any]:
    """Generate a focused plan for the deposition."""
    
    # Get topic from state
    topic = state["topic"]
    
    # Generate deposition plan using planner model with focused instructions
    messages = [
        HumanMessage(content=deposition_planner_instructions.format(
            topic=topic,
            deposition_organization=config.deposition_organization
        ))
    ]
    
    response = await planner_model.ainvoke(messages)
    
    # Format the response as JSON
    sections = await format_as_json(response.content, SECTIONS_SCHEMA)
    
    # For each section that needs investigation, generate search queries
    query_tasks = []
    for section in sections["sections"]:
        if section["investigation"]:
            messages = [
                HumanMessage(content=query_writer_instructions.format(
                    topic=section["description"],
                    number_of_queries=DEFAULT_NUM_QUERIES
                ))
            ]
            query_tasks.append(planner_model.ainvoke(messages))
    
    # Get all search queries concurrently
    query_responses = await asyncio.gather(*query_tasks)
    
    # Extract queries and perform searches
    all_queries = []
    for response in query_responses:
        try:
            queries = await format_as_json(response.content, QUERY_SCHEMA)
            all_queries.extend([query["query"] for query in queries["queries"]])
        except Exception as e:
            print(f"\nFailed to process queries: {e}")
            print("Response content:")
            print(response.content)
            raise
    
    # Perform vector DB searches
    search_docs = await vector_db_search_async(all_queries)
    
    # Format and deduplicate search results
    source_str = ""
    for doc in search_docs:
        source_str += deduplicate_and_format_sources(
            doc,
            config.max_tokens_per_source
        )
    
    # Update state with sections and research
    state["sections"] = sections["sections"]
    state["report_sections_from_research"] = source_str
    
    return state

async def format_as_json(content: str, schema_description: str) -> dict:
    """Format content as JSON using a dedicated model.
    
    Args:
        content: The content to format as JSON
        schema_description: Description of the required JSON schema
    
    Returns:
        dict: The formatted JSON object
    """
    # First try to parse directly if it's already valid JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    prompt = f"""You are a JSON formatting expert. Your task is to take the following content and format it as a valid JSON object.

Required JSON structure:
{schema_description}

Content to format:
{content}

Return ONLY the JSON object, no other text or explanations. The response must be directly parseable by json.loads()."""

    messages = [
        HumanMessage(content=prompt)
    ]
    
    response = await json_formatter_model.ainvoke(messages)
    
    try:
        return json.loads(response.content)
    except json.JSONDecodeError:
        # Try to extract JSON from the response if needed
        return extract_json_from_text(response.content)

async def generate_section_queries(state: SectionState) -> Dict[str, Any]:
    """Generate search queries for each section that requires investigation."""
    # First ensure we have the section in our state
    if "section" not in state:
        print("Warning: No section in state")
        return {"completed_sections": []}
        
    section = state["section"]
    if not section:
        print("Warning: Section is None")
        return {"completed_sections": []}

    print(f"\nGenerating queries for section: {section['name']}")
    
    # Generate queries for this section
    messages = [
        HumanMessage(content=query_writer_instructions.format(
            topic=section["description"],
            number_of_queries=DEFAULT_NUM_QUERIES
        ))
    ]
    
    response = await writer_model.ainvoke(messages)
    print("Raw response content:")
    print(response.content)
    
    try:
        # Parse queries
        query_data = await format_as_json(response.content, QUERY_SCHEMA)
        queries = query_data.get("queries", [])
        
        if not queries:
            print(f"Warning: No queries generated for section {section['name']}")
            return {"completed_sections": []}
            
        # Add section info to queries
        for query in queries:
            query["section"] = section["name"]
            
        # Perform vector DB searches
        search_queries = [query["query"] for query in queries]
        search_docs = await vector_db_search_async(search_queries)
        
        # Format and deduplicate search results
        source_str = ""
        for doc in search_docs:
            source_str += deduplicate_and_format_sources(
                doc,
                config.max_tokens_per_source
            )
        
        return {
            "queries": queries,
            "source_str": source_str,
            "section": section
        }
        
    except Exception as e:
        print(f"Failed to process queries for section {section['name']}: {str(e)}")
        raise

async def write_section(state: SectionState) -> Dict[str, Any]:
    """Write questions for a deposition topic."""
    # Get state 
    section = state["section"]
    source_str = state["source_str"]

    if not section:
        print("Warning: No section to write")
        return {"completed_sections": []}

    # Format system instructions
    system_instructions = topic_writer_instructions.format(
        topic=section["name"],
        context=source_str
    )

    # Generate questions  
    messages = [
        HumanMessage(content=system_instructions),
        HumanMessage(content="Generate deposition questions based on the provided case documents.")
    ]
    
    section_content = await writer_model.ainvoke(messages)
    
    # Write content to the section object  
    section["content"] = section_content.content

    print(f"\nGenerated content for section: {section['name']}")
    print("Preview:", section_content.content[:200], "...")

    # Write the updated section to completed sections
    return {"completed_sections": [section]}

async def write_final_sections(state: SectionState) -> Dict[str, Any]:
    """Write final sections of the deposition outline."""
    # Get state 
    section = state["section"]
    completed_sections = state["report_sections_from_research"]
    
    if not section:
        print("Warning: No section to write")
        return {"completed_sections": []}
    
    # Format system instructions
    system_instructions = final_topic_writer_instructions.format(
        topic=section["name"],
        context=completed_sections
    )

    # Generate section  
    messages = [
        HumanMessage(content=system_instructions),
        HumanMessage(content="Generate the deposition section based on the completed topics.")
    ]
    
    try:
        section_content = await writer_model.ainvoke(messages)
        
        # Create a new section object to avoid modifying the input
        completed_section = section.copy()
        completed_section["content"] = section_content.content

        print(f"\nGenerated content for final section: {completed_section['name']}")
        print("Preview:", section_content.content[:200], "...")

        # Return the completed section
        return {"completed_sections": [completed_section]}
        
    except Exception as e:
        print(f"Error generating content for final section {section['name']}: {str(e)}")
        return {"completed_sections": []}

def human_feedback(state: ReportState):
    """ No-op node that should be interrupted on """
    pass

def gather_completed_sections(state: ReportState):
    """ Gather completed sections from research and format them as context for writing the final sections """    

    # List of completed sections
    completed_sections = state["completed_sections"]

    # Convert dictionary sections to Section objects
    section_objects = []
    for section in completed_sections:
        section_objects.append(Section(
            name=section["name"],
            description=section["description"],
            research=section["investigation"],  # map investigation to research
            content=section.get("content", "")
        ))

    # Format completed section to str to use as context for final sections
    completed_report_sections = format_sections(section_objects)

    return {"report_sections_from_research": completed_report_sections}

def initiate_section_writing(state: ReportState):
    """This is the "map" step when we kick off research for some sections of the deposition."""
    # Get feedback
    feedback = state.get("feedback_on_report_plan", None)

    # If feedback exists and plan not accepted, regenerate plan
    if not state.get("accept_report_plan") and feedback:
        return "generate_report_plan"
    
    # Kick off section writing in parallel via Send() API for any sections that require investigation
    else:
        sends = []
        for section in state["sections"]:
            if section["investigation"]:
                sends.append(
                    Send("build_section_with_web_research", {
                        "section": section,
                        "sections": state["sections"]  # Pass full sections list
                    })
                )
        return sends

def initiate_final_section_writing(state: ReportState):
    """ Write any final sections using the Send API to parallelize the process """    

    # Kick off section writing in parallel via Send() API for any sections that do not require investigation
    return [
        Send("write_final_sections", {"section": s, "report_sections_from_research": state["report_sections_from_research"]}) 
        for s in state["sections"] 
        if not s["investigation"]
    ]

def compile_final_report(state: ReportState):
    """ Compile the final deposition outline """    

    # Get sections and completed sections from state
    sections = state["sections"]
    completed_sections = state.get("completed_sections", [])
    
    print("\nCompiling final report...")
    print(f"Found {len(sections)} total sections")
    print(f"Found {len(completed_sections)} completed sections")
    
    # Create a mapping of section names to their content
    section_content_map = {s["name"]: s.get("content", "") for s in completed_sections}
    print(f"Section names with content: {list(section_content_map.keys())}")
    
    # Update sections with completed content while maintaining original order
    final_sections = []
    for section in sections:
        section_name = section["name"]
        if section_name in section_content_map:
            section["content"] = section_content_map[section_name]
            final_sections.append(section)
        else:
            print(f"Warning: No content found for section {section_name}")
    
    # Compile final report
    if not final_sections:
        print("Warning: No sections with content found")
        return {"final_report": ""}
        
    all_sections = "\n\n".join([s["content"] for s in final_sections if s.get("content")])
    
    if not all_sections:
        print("Warning: No content found in any sections")
        return {"final_report": ""}
        
    print(f"\nSuccessfully compiled final report with {len(final_sections)} sections")
    return {"final_report": all_sections}

# Report section sub-graph -- 

# Add nodes 
section_builder = StateGraph(SectionState, output=SectionOutputState)
section_builder.add_node("generate_queries", generate_section_queries)
section_builder.add_node("write_section", write_section)

# Add edges
section_builder.add_edge(START, "generate_queries")
section_builder.add_edge("generate_queries", "write_section")
section_builder.add_edge("write_section", END)

# Outer graph -- 

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_vector_search", section_builder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)
builder.add_node("compile_final_report", compile_final_report)

# Add edges
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_conditional_edges("human_feedback", initiate_section_writing, ["build_section_with_web_research", "generate_report_plan"])
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])
builder.add_edge("write_final_sections", "compile_final_report")
builder.add_edge("compile_final_report", END)

graph = builder.compile(interrupt_before=['human_feedback'])
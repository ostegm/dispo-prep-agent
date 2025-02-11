import asyncio
import json
import logging
import re
from langchain_anthropic import ChatAnthropic 
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel as ChatModel
from langgraph.graph import START, END, StateGraph
from src.depo_prep.state import ReportStateInput, ReportStateOutput, ReportState, DepositionSection, SearchResult
from src.depo_prep import prompts
from src.depo_prep.configuration import Configuration
from src.depo_prep import utils

# Setup logging
logger = logging.getLogger(__name__)

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

# Nodes
async def generate_deposition_plan(state: ReportState, config: RunnableConfig) -> ReportState:
    """Generate initial deposition plan."""
    configurable = Configuration.from_runnable_config(config)
    # Fetch complaint text
    complaint_text = await utils.get_full_document_text(configurable.chroma_collection_name, "complaint.pdf")
    
    # Prepare feedback context if it exists
    feedback_context = ""
    if state.get("feedback_on_plan"):
        feedback_context = f"Based on previous feedback: {state['feedback_on_plan']}"
    
    # Format system instructions
    system_instructions = prompts.deposition_planner_instructions.format(
        complaint_context=complaint_text,
        topic=state["topic"],
        feedback_context=feedback_context,
        max_sections=configurable.max_sections,
        default_num_queries_per_section=configurable.default_num_queries_per_section
    )
    
    # Generate outline
    messages = [
        SystemMessage(content=system_instructions),
        HumanMessage(content="Generate a deposition outline based on the complaint and topic.")
    ]
    
    response = await planner_model.ainvoke(messages)
    raw_outline = response.content
    
    # Extract sections from the outline using HTML tags
    section_pattern = r"<section>(.*?)</section>"
    raw_sections = re.findall(section_pattern, raw_outline, re.DOTALL)
    
    if not raw_sections:
        raise ValueError("No sections found in model response")
    
    # Convert each section to structured format using structured output
    structured_planner = writer_model.with_structured_output(DepositionSection)
    
    async def process_section(raw_section: str):
        section_messages = [
            SystemMessage(content="""Convert this deposition section into a structured format with:
- A name field as a brief title
- A description field explaining the section
- A queries field as a list of search queries (not a string)

Format the output as a JSON object with these exact fields."""),
            HumanMessage(content=raw_section)
        ]
        section_data = await structured_planner.ainvoke(section_messages)
        return section_data.model_dump()
    
    # Process all sections in parallel
    structured_sections = await asyncio.gather(
        *[process_section(section) for section in raw_sections]
    )
    
    # Update state with structured sections
    state["sections"] = structured_sections
    state["complaint_text"] = complaint_text
    state["status"] = "plan_generated"
    
    return state

def human_feedback(state: ReportState):
    """ No-op node that should be interrupted on """
    logger.info("Entering human feedback node")
    logger.info(f"Current state keys: {state.keys()}")
    logger.info(f"Full state: {state}")
    
    state["status"] = "awaiting_feedback"
    return state


def maybe_regenerate_report_plan(state: ReportState):
    """Conditional routing based on feedback and plan acceptance status"""    
        
    feedback = state.get("feedback_on_plan", None)
    plan_accepted = state.get("accept_plan", False)
    
    logger.info("Checking feedback status:")
    logger.info(f"Feedback received: {feedback}")
    logger.info(f"Plan acceptance status: {plan_accepted}")

    if not plan_accepted:
        if not feedback:
            raise ValueError("Plan not accepted, but no feedback received")
        return "generate_deposition_plan"
    
    return "gather_search_results"

def compile_final_report(state: ReportState):
    """ Compile the final deposition outline """    
    state["status"] = "compiling_report"

    sections = state["sections"]
    completed_sections = state.get("completed_sections", [])
    
    logger.info("\nCompiling final report...")
    logger.info(f"Found {len(sections)} total sections")
    logger.info(f"Found {len(completed_sections)} completed sections")
    
    section_content_map = {s["name"]: s.get("content", "") for s in completed_sections}
    logger.info(f"Section names with content: {list(section_content_map.keys())}")
    
    final_sections = []
    for section in sections:
        section_name = section["name"]
        if section_name in section_content_map:
            section["content"] = section_content_map[section_name]
            final_sections.append(section)
        else:
            logger.warning(f"No content found for section {section_name}")
    
    if not final_sections:
        logger.warning("No sections with content found")
        state["status"] = "error"
        return {"final_report": "", "status": "error"}
        
    all_sections = "\n\n".join([s["content"] for s in final_sections if s.get("content")])
    
    if not all_sections:
        logger.warning("No content found in any sections")
        state["status"] = "error"
        return {"final_report": "", "status": "error"}
        
    logger.info(f"\nSuccessfully compiled final report with {len(final_sections)} sections")
    state["status"] = "completed"
    return {"final_report": all_sections, "status": "completed"}


async def gather_search_results(
    state: ReportState,
    config: RunnableConfig
) -> ReportState:
    """Gather all vector search results into structured format."""
    configurable = Configuration.from_runnable_config(config)
    
    # Gather all searches in parallel
    search_tasks = []
    for section in state["sections"]:
        for query in section["queries"]:
            search_tasks.append({
                "section_name": section["name"],
                "query": query,
                "description": section["description"],
                "task": utils.vector_db_search_async(configurable.chroma_collection_name, query)
            })
    
    # Execute all searches concurrently
    search_results = []
    if search_tasks:
        results = await asyncio.gather(*(task["task"] for task in search_tasks))
        
        # Format results into structured data
        for task, result in zip(search_tasks, results):
            search_results.append(SearchResult(
                query=task["query"],
                section_name=task["section_name"],
                description=task["description"],
                results=result
            ).model_dump())
    
    # Store in state
    state["sections"] = search_results
    state["status"] = "searches_completed"
    
    return state

async def compile_markdown_report(state: ReportState) -> ReportState:
    """Convert search results and sections into formatted markdown."""
    
    system_prompt = prompts.markdown_compiler_prompt

    # Prepare data for the writer
    data = {
        "topic": state["topic"],
        "complaint_text": state["complaint_text"],
        "sections": state["sections"],
    }
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Please create the markdown report using this data: {json.dumps(data, indent=2)}")
    ]
    
    # Generate markdown
    response = await writer_model.ainvoke(messages)
    
    # Update state
    state["final_report"] = response.content
    state["status"] = "markdown_compiled"
    
    return state

async def generate_deposition_questions(
    state: ReportState,
) -> ReportState:
    """Generate potential deposition questions based on the markdown report."""
    
    system_prompt = prompts.deposition_questions_prompt

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Given this deposition plan, generate appropriate questions:\n\n{state['final_report']}")
    ]
    
    # Generate questions
    response = await planner_model.ainvoke(messages)
    
    # Update state with enhanced markdown
    state["final_report"] = response.content
    state["status"] = "questions_added"
    
    return state

# Add nodes
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)

# Core nodes
builder.add_node("generate_deposition_plan", generate_deposition_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("gather_search_results", gather_search_results)
builder.add_node("compile_markdown_report", compile_markdown_report)
builder.add_node("generate_deposition_questions", generate_deposition_questions)

# Add edges for main flow
builder.add_edge(START, "generate_deposition_plan")
builder.add_edge("generate_deposition_plan", "human_feedback")

# Conditional edge for human feedback loop
builder.add_conditional_edges("human_feedback", maybe_regenerate_report_plan,
                              ["generate_deposition_plan", "gather_search_results"])

# Continue with approved plan
builder.add_edge("gather_search_results", "compile_markdown_report")
builder.add_edge("compile_markdown_report", "generate_deposition_questions")
builder.add_edge("generate_deposition_questions", END)

# Compile graph with interruption points
graph = builder.compile(interrupt_before=['human_feedback'])
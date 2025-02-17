import asyncio
import json
import logging
import re
from langchain_anthropic import ChatAnthropic 
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START, END, StateGraph
from langgraph.constants import Send
from langgraph.types import Command
from src.depo_prep.state import ReportStateInput, ReportStateOutput, ReportState, DepositionSection, SectionState
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


async def process_single_section(state: SectionState):
    """Process a single section with retry logic."""
    retry_count = 0
    max_retries = 3
    
    structured_writer = writer_model.with_structured_output(DepositionSection)
    
    while retry_count <= max_retries:
        try:
            section_messages = [
                SystemMessage(content=prompts.section_processor_prompt),
                HumanMessage(content=state["raw_section"])
            ]
            section_data = await structured_writer.ainvoke(section_messages)
            return {"processed_sections": [section_data]}
        except Exception as e:
            if "overloaded" in str(e).lower() and retry_count < max_retries:
                # Exponential backoff
                wait_time = 2 ** retry_count
                logger.warning(f"API overloaded, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                retry_count += 1
    logging.error("Returning empty processed section....")
    return {"processed_sections": []}

def initiate_section_processing(state: ReportState):
    """Fan out to process sections in parallel using Send API."""
    to_send = []
    for raw_section in state["raw_sections"]:
        to_send.append(Send("process_section", {"raw_section": raw_section}))
    return to_send

async def get_documents_for_context(configurable: Configuration):
    """Fetch all documents from the vector database."""
    logger.info("Fetching all documents from the vector database...")
    all_documents = await utils.get_all_documents_text(configurable.chroma_collection_name)
    
    # Format all documents into a context string
    context_parts = []
    for filename, content in all_documents.items():
        context_parts.append(f"### {filename}\n{content}\n")
    all_context = "\n".join(context_parts)
    return all_context


async def generate_deposition_plan(state: ReportState, config: RunnableConfig) -> ReportState:
    """Generate initial deposition plan."""
    configurable = Configuration.from_runnable_config(config)

    # Check if we are rebuilding the plan based on feedback from a previous turn.
    if state.get("feedback_on_plan"):
        logger.info("Feedback on plan found, adding feedback to the existing prompt.")
        existing_prompt_messages = state.get("deposition_plan_prompt", [])
        existing_prompt_messages.extend([
            AIMessage(content=state["deposition_summary"]),
            HumanMessage(content=state["feedback_on_plan"] + " dont ask any followup questions, just follow the original instructions on formatting.")
        ])
        messages = existing_prompt_messages
    else:
        # Build the prompt from user input + documents.
        system_instructions = prompts.deposition_planner_instructions.format(
            document_context=await get_documents_for_context(configurable),
            topic=state["user_provided_topic"],
            max_sections=configurable.max_sections,
            default_num_queries_per_section=configurable.default_num_queries_per_section
        )
        
        messages = [HumanMessage(content=system_instructions)]

    # Store the prompt in the state in case we need to regenerate the plan based on more feedback.
    state["deposition_plan_prompt"] = messages

    response = await planner_model.ainvoke(messages)
    raw_outline = response.content
    
    # Extract summary from the outline
    summary_pattern = r"<summary>(.*?)</summary>"
    summary_match = re.search(summary_pattern, raw_outline, re.DOTALL)
    if not summary_match:
        raise ValueError("No summary found in model response")
    
    # Store the summary in state
    state["deposition_summary"] = summary_match.group(1).strip()
    
    # Extract sections from the outline using HTML tags
    section_pattern = r"<section>(.*?)</section>"
    raw_sections = re.findall(section_pattern, raw_outline, re.DOTALL)
    
    if not raw_sections:
        raise ValueError("No sections found in model response")
    
    # Store raw sections for parallel processing
    state["raw_sections"] = raw_sections
    state["status"] = "sections_extracted"
    
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
        
    feedback = state.get("feedback_on_plan", "")
    plan_accepted = state.get("accept_plan", False)
    
    logger.info("Checking feedback status:")
    logger.info(f"Feedback received: {feedback}")
    logger.info(f"Plan acceptance status: {plan_accepted}")

    if plan_accepted:
        state["status"] = "plan_accepted"
        return "convert_sections_to_markdown"
    
    # If not accepted, we must have feedback
    if not plan_accepted and not feedback:
        raise ValueError("Plan not accepted, but no feedback received")
    
    # Clear any sections from previous turns.
    state["processed_sections"].clear()
    state["raw_sections"].clear()
    logging.info("Cleared sections from previous turns.")
    state["status"] = "regenerating_plan"
    return "generate_deposition_plan"


async def convert_sections_to_markdown(state: ReportState) -> ReportState:
    """Convert search results and sections into formatted markdown."""
    
    # Prepare data for the writer
    data = {
        "topic": state["deposition_summary"],
        "sections": [s.model_dump() for s in state["processed_sections"]],
    }
    
    messages = [
        SystemMessage(content=prompts.markdown_compiler_prompt),
        HumanMessage(content=f"Please create the markdown report using this data: {json.dumps(data, indent=2)}")
    ]
    
    # Generate markdown
    response = writer_model.invoke(messages)

    # Update state
    state["status"] = "markdown_compiled"
    state["markdown_document"] = response.content
    # Try to avoid duplicating processed sections.
    state["processed_sections"] = []
    
    return state

async def add_deposition_questions(
    state: ReportState,
) -> ReportState:
    """Generate potential deposition questions based on the markdown report."""
    
    system_prompt = prompts.deposition_questions_prompt
    human_message = """Given this deposition plan, generate appropriate questions:
        {markdown_document}""".format(markdown_document=state["markdown_document"])

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
    ]
    
    # Generate questions
    response = await planner_model.ainvoke(messages)
    
    # Update state with enhanced markdown
    state["markdown_document"] = response.content
    state["status"] = "questions_added"
    
    return state



# Main graph
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)

# Core nodes
builder.add_node("generate_deposition_plan", generate_deposition_plan)
builder.add_node("process_section", process_single_section)
builder.add_node("human_feedback", human_feedback)
builder.add_node("convert_sections_to_markdown", convert_sections_to_markdown)
builder.add_node("add_deposition_questions", add_deposition_questions)

# Add edges for main flow
builder.add_edge(START, "generate_deposition_plan")

# Add conditional edges for parallel processing
builder.add_conditional_edges(
    "generate_deposition_plan",
    initiate_section_processing,
    ["process_section"] 
)
builder.add_edge("process_section", "human_feedback")

# Conditional edge for human feedback loop
builder.add_conditional_edges(
    "human_feedback",
    maybe_regenerate_report_plan,
    ["generate_deposition_plan", "convert_sections_to_markdown"]
)

# Continue with approved plan
builder.add_edge("convert_sections_to_markdown", "add_deposition_questions")
builder.add_edge("add_deposition_questions", END)

# Compile graph with interruption points
graph = builder.compile(interrupt_before=['human_feedback'])
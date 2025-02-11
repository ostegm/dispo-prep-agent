import asyncio
import logging

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph

from src.depo_prep.configuration import Configuration
from src.depo_prep import prompts, utils
from src.depo_prep.state import ReportState, ReportStateInput, ReportStateOutput

logger = logging.getLogger(__name__)

async def one_shot_deposition_plan(state: ReportState, config) -> ReportState:
    """
    Generate a complete deposition plan in one shot using all available document context.
    
    This node:
      - Retrieves all documents from the vector database
      - Formats the deposition prompt using all available context
      - Calls the planner model to generate the deposition plan
      - Updates the state with the final report and status
    """
    # Initialize configuration from runnable config
    configurable = Configuration.from_runnable_config(config)
    
    logger.info("Fetching all documents from the vector database...")
    all_documents = await utils.get_all_documents_text(configurable.chroma_collection_name)
    
    # Format all documents into a context string
    context_parts = []
    for filename, content in all_documents.items():
        context_parts.append(f"### {filename}\n{content}\n")
    all_context = "\n".join(context_parts)
    
    # Build the prompt for the deposition plan generation using the new one-shot prompt with questions
    prompt = prompts.deposition_one_shot_instructions.format(
        complaint_context=all_context,
        topic=state["topic"],
        max_sections=configurable.max_sections,
        default_num_queries_per_section=configurable.default_num_queries_per_section
    )
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content="Generate the complete deposition plan with detailed questions in one shot.")
    ]
    
    logger.info("Generating deposition plan in one shot...")
    planner_model = ChatGoogleGenerativeAI(
        model=configurable.planner_model,
        temperature=0,
        convert_system_message_to_human=True
    )
    response = await planner_model.ainvoke(messages)
    final_plan = response.content
    
    # Update state with all document context and generated final report
    state["all_documents"] = all_documents
    state["final_report"] = final_plan
    state["status"] = "completed"
    
    logger.info("One-shot deposition plan generation completed.")
    return state

# Build the one-shot graph using StateGraph from LangGraph
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("one_shot_deposition_plan", one_shot_deposition_plan)
builder.add_edge(START, "one_shot_deposition_plan")
builder.add_edge("one_shot_deposition_plan", END)

graph = builder.compile() 
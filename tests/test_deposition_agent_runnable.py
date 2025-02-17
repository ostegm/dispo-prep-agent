"""Test the deposition preparation agent workflow using direct graph execution."""

import asyncio
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import logging

from langgraph.types import Command, Interrupt
from langgraph.checkpoint.memory import MemorySaver
from src.depo_prep.configuration import Configuration
from src.depo_prep.graph import builder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def handle_human_feedback(plan_text: str) -> bool:
    """Handle human feedback for the plan.
    
    Args:
        plan_text: The plan text to review
        
    Returns:
        True to accept plan, False to provide feedback
    """
    print("\nPlease review the following report plan.")
    print("=" * 80)
    print(plan_text)
    print("=" * 80)
    
    while True:
        response = input("\nPass 'True' to approve the report plan or provide feedback to regenerate the report plan: ").strip()
        
        if response.lower() == 'true':
            return True
        elif response:
            return response
        else:
            print("Please either enter 'True' or provide feedback")

async def test_deposition_workflow():
    """Run the deposition workflow using direct graph execution."""
    
    # Initialize configuration
    config = Configuration()
    
    # Create reports directory in project root
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Default deposition topic
    default_topic = "(TERMINAL) As the plaintiff's attorney, prepare for a deposition of the eyewitness to the accident. The goal is to explore factors that may affect the witness's reliability, including their vantage point, lighting conditions, distractions, potential biases, etc."
    
    # Ask user for topic, use default if blank
    print("\nDefault deposition topic:")
    print(default_topic)
    deposition_topic = input("\nEnter deposition topic (or press Enter to use default): ").strip()
    
    if not deposition_topic:
        deposition_topic = default_topic
        print("\nUsing default topic")
    
    print(f"\nProceeding with topic:\n{deposition_topic}")
    
    # Create thread config
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            "search_api": "tavily",
            "planner_provider": "openai",
            "max_search_depth": 1,
            "planner_model": "o3-mini"
        }
    }
    
    # Initialize memory saver
    memory = MemorySaver()
    
    # Compile the graph with memory
    graph = builder.compile(checkpointer=memory, interrupt_before=['human_feedback'])
    
    # Create initial input
    input_data = {
        "user_provided_topic": deposition_topic,
    }
    
    try:
        # Run the graph and handle interrupts
        print("\nStarting graph execution...")
        
        # Initial run
        async for event in graph.astream(input_data, thread, stream_mode="updates"):
            if "__interrupt__" in event:
                interrupt = event["__interrupt__"]
                if isinstance(interrupt, (dict, Interrupt)):
                    value = interrupt.get("value") if isinstance(interrupt, dict) else interrupt.value
                    if isinstance(value, str):
                        feedback = await handle_human_feedback(value)
                        if feedback is True:
                            # User approved the plan
                            async for resume_event in graph.astream(Command(resume=True), thread, stream_mode="updates"):
                                logger.debug(f"Resume event: {resume_event}")
                        else:
                            # User provided feedback
                            async for resume_event in graph.astream(Command(resume=feedback), thread, stream_mode="updates"):
                                logger.debug(f"Resume event: {resume_event}")
                        break
        
        # Get final state
        final_state = graph.get_state(thread)
        
        if final_state and "markdown_document" in final_state.values:
            print("\nFinal Report:")
            print("=" * 80)
            print(final_state.values["markdown_document"])
            print("=" * 80)
            
            # Save report to reports directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"report_{timestamp}.md"
            report_file.write_text(final_state.values["markdown_document"])
            print(f"\nReport saved to: {report_file}")
            
            # Save memory state
            memory_file = reports_dir / f"memory_{timestamp}.json"
            memory.save(memory_file)
            print(f"\nMemory state saved to: {memory_file}")
        else:
            print("\nNo final report in results")
            print("Final state:", final_state)
            
    except Exception as e:
        logger.error(f"Error during graph execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(test_deposition_workflow()) 
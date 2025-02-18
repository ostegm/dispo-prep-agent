"""Test the deposition preparation agent workflow using direct graph execution."""

import asyncio
import uuid
from pathlib import Path
import logging

from langgraph.checkpoint.memory import MemorySaver
from src.depo_prep.configuration import Configuration
from src.depo_prep.graph import builder
from src.depo_prep import utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_deposition_workflow():
    """Run the deposition workflow using direct graph execution."""
    
    # Initialize configuration
    config = Configuration()
    
    # Create reports directory in project root
    reports_dir = Path(__file__).parent.parent / "reports"
    
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
    
    # Initialize graph input
    initial_input = await utils.initialize_graph_input(deposition_topic)
    
    # Create thread config
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    
    # Initialize memory saver and compile graph
    memory = MemorySaver()
    graph = builder.compile(checkpointer=memory, interrupt_before=['human_feedback'])
    
    try:
        # Run the workflow
        markdown_doc = await utils.run_deposition_workflow(graph, initial_input, thread)
        
        if markdown_doc:
            print("\nFinal Report:")
            print("=" * 80)
            print(markdown_doc)
            print("=" * 80)
            
            # Save the report
            report_file = utils.save_deposition_report(markdown_doc, str(reports_dir))
            print(f"\nReport saved to: {report_file}")
            
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_deposition_workflow()) 
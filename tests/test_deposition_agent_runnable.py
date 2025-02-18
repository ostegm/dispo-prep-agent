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
from src.depo_prep.state import DepositionSection
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
    
    # Fetch document context
    print("\nFetching document context...")
    all_documents = await utils.get_all_documents_text(config.chroma_collection_name)
    context_parts = []
    for filename, content in all_documents.items():
        context_parts.append(f"### {filename}\n{content}\n")
    document_context = "\n".join(context_parts)
    
    # Create thread config
    thread = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    
    # Initialize memory saver
    memory = MemorySaver()
    
    # Compile the graph with memory
    graph = builder.compile(checkpointer=memory, interrupt_before=['human_feedback'])
    
    try:
        # Run the graph and handle interrupts
        print("\nStarting graph execution...")
        
        # Initial input with topic and document context
        initial_input = {
            "user_provided_topic": deposition_topic,
            "document_context": document_context,
        }
        
        # Start the graph execution
        graph_stream = graph.astream(initial_input, thread, stream_mode="values")
        
        # Process events until we need human feedback
        async for event in graph_stream:
            # Get current values and status
            status = event.get("status", "unknown")
            print(f"\nStatus: {status}")
                
            # Show deposition summary when it exists
            deposition_summary = event.get("deposition_summary")
            processed_sections = event.get("processed_sections")
            if deposition_summary and processed_sections:
                print("\nDeposition Summary:")
                print("=" * 80)
                print(deposition_summary)
                print("=" * 80)
                
                # Get user feedback on the plan
                feedback = input("\nEnter feedback on the plan (press Enter to accept without feedback): ").strip()
                accept_plan = not feedback  # Accept if no feedback given
                
                print(f"\n{'Accepting' if accept_plan else 'Rejecting'} plan{' with feedback' if feedback else ''}...")
                # Resume execution with feedback
                graph.update_state(
                    values={
                        "feedback_on_plan": feedback or None,
                        "accept_plan": accept_plan,
                    },
                    config=thread,
                    as_node="human_feedback"
                )
                break
        
        print("Resuming execution...")
        resume_command = Command(
            resume={
                "feedback_on_plan": feedback or None,
                "accept_plan": accept_plan
            }
        )
        # Continue streaming until we get the final state
        final_state = None
        async for event in graph.astream(resume_command, thread, stream_mode="values"):
            print(f"Event: {event.keys()}")
            status = event.get("status", "unknown")
            print(f"\nStatus: {status}")
            
            # Keep track of the latest state
            if event:
                final_state = event
                    
        if not final_state:
            raise ValueError("No final state received from graph execution")
            
        markdown_doc = final_state.get("markdown_document")
        if markdown_doc:
            print("\nFinal Report:")
            print("=" * 80)
            print(markdown_doc)
            print("=" * 80)
            
            # Save report to reports directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = reports_dir / f"report_{timestamp}.md"
            report_file.write_text(markdown_doc)
            print(f"\nReport saved to: {report_file}")
            return  # Exit the function when we have the final report


    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_deposition_workflow()) 
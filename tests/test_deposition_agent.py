"""Test the deposition preparation agent workflow using LangGraph server API."""

import asyncio
import httpx
import json
from pathlib import Path
from typing import Dict, Any
import uuid

from report_maistro.configuration import Configuration
from report_maistro.state import ReportStateInput

LANGGRAPH_API_URL = "http://localhost:2024"

async def get_or_create_assistant() -> str:
    """Get or create an assistant for the graph."""
    async with httpx.AsyncClient() as client:
        # First try to search for existing assistants
        response = await client.post(
            f"{LANGGRAPH_API_URL}/assistants/search",
            json={"graph_id": "report_masitro"}
        )
        response.raise_for_status()
        assistants = response.json()
        
        if assistants:
            return assistants[0]["assistant_id"]
        
        # If no assistant exists, create one
        response = await client.post(
            f"{LANGGRAPH_API_URL}/assistants",
            json={
                "graph_id": "report_masitro",
                "assistant_id": str(uuid.uuid4()),
                "name": "Report Maistro Assistant"
            }
        )
        response.raise_for_status()
        return response.json()["assistant_id"]

async def create_thread() -> str:
    """Create a new thread."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{LANGGRAPH_API_URL}/threads",
            json={
                "thread_id": str(uuid.uuid4()),
                "metadata": {"purpose": "deposition_preparation"}
            }
        )
        response.raise_for_status()
        return response.json()["thread_id"]

async def get_human_feedback(sections: list) -> tuple[bool, str]:
    """Get human feedback on the deposition plan."""
    print("\nPlease review the deposition plan above.")
    while True:
        response = input("\nDo you want to proceed with this plan? (Y/N): ").strip().upper()
        if response in ['Y', 'N']:
            if response == 'Y':
                return True, None
            feedback = input("\nPlease provide feedback on what to change in the plan: ").strip()
            return False, feedback
        print("Invalid input. Please enter Y or N.")

async def test_deposition_workflow():
    """Test the full deposition preparation workflow with a real case scenario using LangGraph server."""
    
    # Initialize configuration
    config = Configuration()
    
    # Ensure we have case documents indexed
    case_docs_dir = Path(__file__).parent.parent / "case_documents"
    assert case_docs_dir.exists(), "case_documents directory not found"
    assert any(case_docs_dir.glob("*.pdf")), "No PDF files found in case_documents directory"
    
    # Test case: Prepare for a deposition about semi-autonomous vehicle accident
    deposition_topic = "Prepare for a deposition of the defendant's engineer about the semi-autonomous vehicle's sensor system failure that led to the accident"
    
    # Initialize the input state as a dictionary
    input_state = {
        "topic": deposition_topic,
        "feedback_on_report_plan": None,
        "accept_report_plan": True
    }
    
    # Create config dict for the graph
    config_dict = {"configurable": config.model_dump()}
    
    print("\nStarting deposition preparation workflow...")
    print(f"Topic: {deposition_topic}")
    
    try:
        # Get or create an assistant
        assistant_id = await get_or_create_assistant()
        print(f"Using assistant: {assistant_id}")
        
        # Create a thread
        thread_id = await create_thread()
        print(f"Created thread: {thread_id}")
        
        # Track if we need to regenerate the plan
        regenerate_plan = True
        current_sections = None
        
        while regenerate_plan:
            # Create a run and stream the results
            async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
                response = await client.post(
                    f"{LANGGRAPH_API_URL}/threads/{thread_id}/runs/stream",
                    json={
                        "assistant_id": assistant_id,
                        "input": input_state,
                        "config": config_dict,
                        "stream_mode": ["values", "events"]
                    }
                )
                response.raise_for_status()
                
                # Process the server-sent events stream
                final_output = None
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])  # Skip "data: " prefix
                            
                            # Store the last output as final
                            if isinstance(data, dict):
                                final_output = data
                                if "sections" in data:
                                    current_sections = data["sections"]
                            
                            # Process intermediate outputs
                            process_output(data)
                            
                        except json.JSONDecodeError:
                            continue
                
                # If we have sections, get human feedback
                if current_sections:
                    accept_plan, feedback = await get_human_feedback(current_sections)
                    if accept_plan:
                        regenerate_plan = False
                    else:
                        # Update input state with feedback
                        input_state = {
                            "topic": deposition_topic,
                            "feedback_on_report_plan": feedback,
                            "accept_report_plan": False
                        }
                        print("\nRegenerating plan with feedback...")
                        continue
                
                # Process final output if we have one
                if final_output and "final_report" in final_output:
                    print("\nFinal Deposition Report:")
                    print("=" * 80)
                    print(final_output["final_report"])
                    print("=" * 80)
                            
            print("\nDeposition preparation workflow completed successfully!")
        
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        raise

def process_output(output: Dict[str, Any]):
    """Process and validate the graph output."""
    if isinstance(output, str):
        print(f"\nEvent: {output}")
        return
        
    if "sections" in output:
        print("\nDeposition Topics Generated:")
        for section in output["sections"]:
            print(f"\n- Topic: {section['name']}")
            print(f"  Description: {section['description']}")
            print(f"  Requires Investigation: {section['investigation']}")
            if section.get('content'):
                print(f"\n  Questions Preview: {section['content'][:200]}...")
    
    if "report_sections_from_research" in output:
        print("\nCase Document Search Results:")
        print(f"Found {len(output['report_sections_from_research'].split('Source'))-1} relevant document sections")
    
    if "final_report" in output:
        print("\nFinal Deposition Outline:")
        print("-" * 80)
        print(output["final_report"])
        print("-" * 80)

if __name__ == "__main__":
    asyncio.run(test_deposition_workflow()) 
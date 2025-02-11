"""Test the deposition preparation agent workflow using LangGraph server API."""

import asyncio
import httpx
import json
import uuid
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from depo_prep.configuration import Configuration

LANGGRAPH_API_URL = "http://localhost:2024"

def get_available_graphs() -> Dict[str, str]:
    """Get available graphs from langgraph.json."""
    langgraph_path = Path(__file__).parent.parent / "langgraph.json"
    with open(langgraph_path) as f:
        config = json.load(f)
    return config.get("graphs", {})

async def get_or_create_assistant(graph_id: str) -> str:
    """Get or create an assistant for the graph."""
    async with httpx.AsyncClient() as client:
        # First try to search for existing assistants
        response = await client.post(
            f"{LANGGRAPH_API_URL}/assistants/search",
            json={"graph_id": graph_id}
        )
        response.raise_for_status()
        assistants = response.json()
        
        if assistants:
            return assistants[0]["assistant_id"]
        
        # If no assistant exists, create one
        response = await client.post(
            f"{LANGGRAPH_API_URL}/assistants",
            json={
                "graph_id": graph_id,
                "assistant_id": str(uuid.uuid4()),
                "name": f"{graph_id.title()} Assistant"
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

async def resume_after_human_feedback(client: httpx.AsyncClient, thread_id: str, assistant_id: str) -> None:
    """Resume the workflow after human feedback."""
    # Get user input for feedback
    feedback = input("\nEnter feedback on the plan (press Enter to accept without feedback): ").strip()
    accept_plan = not feedback  # Accept if no feedback given
    
    print(f"\n{'Accepting' if accept_plan else 'Rejecting'} plan{' with feedback' if feedback else ''}...")
    print(f"Accept plan: {accept_plan}")
    print(f"Feedback: {feedback}")
    try:
        response = await client.post(
            f"{LANGGRAPH_API_URL}/threads/{thread_id}/runs",
            json={
                "assistant_id": assistant_id,
                "checkpoint": {
                    "thread_id": thread_id,
                },
                "command": {
                    "update": {
                        "feedback_on_plan": feedback or None,
                        "accept_plan": accept_plan
                    },
                }
            }
        )
        response.raise_for_status()
        print("Successfully resumed workflow")
    except httpx.HTTPStatusError as e:
        print(f"\nError resuming workflow: {e.response.status_code} {e.response.text}")
        print(f"Request URL: {e.request.url}")
        print(f"Request headers: {dict(e.request.headers)}")
        print(f"Request body: {e.request.content.decode()}")
        raise
    except Exception as e:
        print(f"\nUnexpected error resuming workflow: {e}")
        raise

def format_plan(sections: list[dict]) -> str:
    """Format the sections into a readable plan."""
    plan = []
    for i, section in enumerate(sections, 1):
        plan.append(f"{i}. {section['name']}")
        plan.append(f"   Description: {section['description']}")
        plan.append(f"   Research needed: {'Yes' if section.get('investigation') else 'No'}")
        plan.append("")
    return "\n".join(plan)

async def poll_thread_state(client: httpx.AsyncClient, thread_id: str, assistant_id: str) -> Dict[str, Any]:
    """Poll thread state until completion."""
    while True:
        response = await client.get(f"{LANGGRAPH_API_URL}/threads/{thread_id}/state")
        data = response.json()
        
        # Get current values and status
        values = data.get("values", {})
        if values:
            status = values.get("status", "unknown")
            print(f"\nStatus: {status}")
            
            # If we have sections and we're at plan review stage, display the plan
            if status == "plan_generated" and values.get("sections"):
                print("\nProposed Plan:")
                print("=" * 80)
                plan = format_plan(values["sections"])
                print(plan)
                print("=" * 80)
            
            # Check if we have a final report
            if values.get("final_report"):
                return values
                
        # Show what nodes we're waiting on
        next_nodes = data.get("next", [])
        if next_nodes:
            print(f"Waiting at: {next_nodes}")
            
            if "human_feedback" in next_nodes:
                await resume_after_human_feedback(client, thread_id, assistant_id)
            
        await asyncio.sleep(2)

async def test_deposition_workflow():
    """Run the deposition workflow and poll for results."""
    
    # Initialize configuration
    config = Configuration()
    
    # Get available graphs from langgraph.json
    available_graphs = get_available_graphs()
    
    if not available_graphs:
        raise ValueError("No graphs found in langgraph.json")
    
    # Display available graphs
    print("\nAvailable graphs:")
    for i, (graph_id, path) in enumerate(available_graphs.items(), 1):
        description = "(Multi-step with human feedback)" if graph_id == "depo_prep" else "(Single-step generation)"
        print(f"{i}. {graph_id} {description}")
        print(f"   Path: {path}")
    
    # Get user selection
    while True:
        graph_choice = input(f"\nSelect graph (1-{len(available_graphs)}): ").strip()
        if graph_choice.isdigit() and 1 <= int(graph_choice) <= len(available_graphs):
            break
        print(f"Invalid choice. Please enter a number between 1 and {len(available_graphs)}")
    
    # Get selected graph_id
    graph_id = list(available_graphs.keys())[int(graph_choice) - 1]
    print(f"\nUsing graph: {graph_id}")
    
    # Create reports directory in project root
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Default deposition topic
    default_topic = "Prepare for a deposition of the defendant Luther about the human factors and decisions that led to the crash, focusing on their state of mind, awareness of risks, and actions taken before and during the incident"
    
    # Ask user for topic, use default if blank
    print("\nDefault deposition topic:")
    print(default_topic)
    deposition_topic = input("\nEnter deposition topic (or press Enter to use default): ").strip()
    
    if not deposition_topic:
        deposition_topic = default_topic
        print("\nUsing default topic")
    
    print(f"\nProceeding with topic:\n{deposition_topic}")
    
    # Create input state
    input_state = {
        "topic": deposition_topic,
        "feedback_on_report_plan": None,
        "accept_report_plan": False
    }
    
    # Get or create assistant
    assistant_id = await get_or_create_assistant(graph_id)
    print(f"\nUsing assistant: {assistant_id}")
    
    # Create thread
    thread_id = await create_thread()
    print(f"\nCreated thread: {thread_id}")
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
        # Start the workflow
        run_response = await client.post(
            f"{LANGGRAPH_API_URL}/threads/{thread_id}/runs",
            json={
                "assistant_id": assistant_id,
                "input": input_state,
                "config": {"configurable": config.model_dump()}
            }
        )
        run_response.raise_for_status()
        print("\nStarted workflow, polling for results...")
        
        # Poll until completion
        try:
            final_result = await poll_thread_state(client, thread_id, assistant_id)
            if final_result.get("final_report"):
                print("\nFinal Report:")
                print("=" * 80)
                print(final_result["final_report"])
                print("=" * 80)
                
                # Save report to reports directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                report_file = reports_dir / f"report_{timestamp}.md"
                report_file.write_text(final_result["final_report"])
                print(f"\nReport saved to: {report_file}")
            else:
                print("\nNo final report in results")
                
        except Exception as e:
            print(f"Error: {e}")
            raise

if __name__ == "__main__":
    asyncio.run(test_deposition_workflow()) 
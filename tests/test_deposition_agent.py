"""Test the deposition preparation agent workflow."""

import asyncio
from pathlib import Path
from typing import Dict, Any

from src.report_maistro.configuration import Configuration
from src.report_maistro.graph import graph
from src.report_maistro.state import ReportStateInput

async def test_deposition_workflow():
    """Test the full deposition preparation workflow with a real case scenario."""
    
    # Initialize configuration
    config = Configuration()
    
    # Ensure we have case documents indexed
    case_docs_dir = Path(__file__).parent.parent / "case_documents"
    assert case_docs_dir.exists(), "case_documents directory not found"
    assert any(case_docs_dir.glob("*.pdf")), "No PDF files found in case_documents directory"
    
    # Test case: Prepare for a deposition about semi-autonomous vehicle accident
    deposition_topic = "Prepare for a deposition of the defendant's engineer about the semi-autonomous vehicle's sensor system failure that led to the accident"
    
    # Initialize the input state
    input_state = ReportStateInput(
        topic=deposition_topic,
        feedback_on_report_plan=None,
        accept_report_plan=True
    )
    
    # Create config dict for the graph
    config_dict = {"configurable": config.model_dump()}
    
    # Run the graph
    print("\nStarting deposition preparation workflow...")
    print(f"Topic: {deposition_topic}")
    
    def process_output(output: Dict[str, Any]):
        """Process and validate the graph output."""
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
            print(output["final_report"][:500] + "...")
            print("-" * 80)
    
    # Run the graph
    try:
        async for output in graph.astream(
            input_state,
            config=config_dict
        ):
            process_output(output)
            
        # Get final result
        result = output
        
        # Validate the final output
        assert "final_report" in result, "Final deposition outline not generated"
        assert len(result["sections"]) > 0, "No deposition topics generated"
        
        # Check content of sections
        for section in result["sections"]:
            assert section["content"], f"No content generated for topic: {section['name']}"
            assert len(section["content"]) > 100, f"Insufficient content for topic: {section['name']}"
        
        print("\nDeposition preparation workflow completed successfully!")
        
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_deposition_workflow()) 
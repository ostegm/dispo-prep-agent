import asyncio
from typing import List, Dict, Optional, Union, Any
import chromadb
from chromadb.config import Settings
from openai import AsyncOpenAI
from src.depo_prep.configuration import Configuration
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize configuration and Chroma client
config = Configuration()
chroma_client = chromadb.Client(Settings(
    persist_directory=config.chroma_persist_dir,
    is_persistent=True
))

async def get_embedding(query: str) -> List[float]:
    """Get embedding for a text using OpenAI's API."""
    async_client = AsyncOpenAI()
    response = await async_client.embeddings.create(
        model="text-embedding-ada-002",
        input=query
    )
    return response.data[0].embedding

async def vector_db_search_async(chroma_collection_name: str, search_queries: Union[str, List[str]]) -> List[Dict]:
    """
    Performs concurrent vector database searches using Chroma.
    
    Args:
        search_queries: Either a single search query string or a list of query strings
        
    Returns:
        List[dict]: List of search results from the vector database, one per query
    """
    if isinstance(search_queries, str):
        search_queries = [search_queries]
        
    async_client = AsyncOpenAI()
    
    async def get_embedding(query: str):
        response = await async_client.embeddings.create(
            model="text-embedding-ada-002",
            input=query
        )
        return response.data[0].embedding
    collection = chroma_client.get_collection(chroma_collection_name)
    async def search_single_query(query: str):
        embedding = await get_embedding(query)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=5
        )
        
        # Format results to match SearchResult model expectations
        formatted_results = []
        for i, (text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            result = {
                'source': metadata.get('source', 'unknown'),
                'content': text,
                'score': str(results['distances'][0][i])
            }
            formatted_results.append(result)
            
        logger.info(f"Found {len(formatted_results)} results for query: {query}")
        return formatted_results
    
    # Run searches concurrently
    tasks = [search_single_query(query) for query in search_queries]
    results = await asyncio.gather(*tasks)
    
    return results[0] if len(results) == 1 else results



async def get_full_document_text(chroma_collection_name: str, filename: str) -> str:
    """
    Fetches and combines all chunks of a document from the vector database.
    
    Args:
        filename: Base name of the document file to retrieve (e.g. "complaint.pdf")
        
    Returns:
        str: The full text of the document
    """
    # Get all documents with this filename
    collection = chroma_client.get_collection(chroma_collection_name)
    results = collection.get(
        where={
            "source": filename  # Simple exact match
        },
        include=['documents', 'metadatas']
    )
    
    if not results or not results.get('documents'):
        raise ValueError(f"Could not find {filename} in the vector database")

    # Sort chunks by index
    chunks = list(zip(results['documents'], results['metadatas']))
    sorted_chunks = sorted(chunks, key=lambda x: x[1].get('chunk_index', 0))
    
    # Join all text chunks with spaces
    full_text = ' '.join(chunk[0] for chunk in sorted_chunks)
    
    return full_text

async def get_all_documents_text(chroma_collection_name: str) -> Dict[str, str]:
    """
    Fetches and combines all documents from the vector database, organized by source file.
    
    Returns:
        Dict[str, str]: Dictionary mapping filenames to their full document text
    """
    try:
        collection = chroma_client.get_collection(chroma_collection_name)
        
        # Get all documents
        results = collection.get(
            include=['documents', 'metadatas']
        )
        
        if not results or not results.get('documents'):
            logger.warning("No documents found in the vector database")
            return {}

        # Group chunks by source file
        documents = {}
        chunks = list(zip(results['documents'], results['metadatas']))
        
        # Sort and combine chunks for each source file
        for text, metadata in chunks:
            source = metadata.get('source', 'unknown')
            if source not in documents:
                documents[source] = []
            documents[source].append((text, metadata.get('chunk_index', 0)))
        
        # Sort chunks by index and join text for each document
        full_documents = {}
        for source, chunks in documents.items():
            sorted_chunks = sorted(chunks, key=lambda x: x[1])
            full_documents[source] = ' '.join(chunk[0] for chunk in sorted_chunks)
            logger.info(f"Retrieved {len(chunks)} chunks for {source}")
        
        return full_documents
        
    except Exception as e:
        logger.error(f"Error retrieving documents: {str(e)}")
        raise

async def initialize_graph_input(deposition_topic: str) -> Dict[str, str]:
    """
    Initialize the input for the graph execution with topic and document context.
    
    Args:
        deposition_topic: The topic for the deposition preparation
        
    Returns:
        Dict containing the initial input for the graph
    """
    # Fetch document context
    all_documents = await get_all_documents_text(config.chroma_collection_name)
    context_parts = []
    for filename, content in all_documents.items():
        context_parts.append(f"### {filename}\n{content}\n")
    document_context = "\n".join(context_parts)
    
    return {
        "user_provided_topic": deposition_topic,
        "document_context": document_context,
    }

def save_deposition_report(markdown_doc: str, reports_dir: str) -> str:
    """
    Save the deposition report to a file.
    
    Args:
        markdown_doc: The markdown content to save
        reports_dir: Directory to save the report in
        
    Returns:
        str: Path to the saved report file
    """
    from datetime import datetime
    from pathlib import Path
    
    # Create reports directory if it doesn't exist
    reports_path = Path(reports_dir)
    reports_path.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = reports_path / f"report_{timestamp}.md"
    
    # Save the report
    report_file.write_text(markdown_doc)
    return str(report_file)

async def run_deposition_workflow(graph, initial_input: Dict[str, Any], thread: Dict[str, Any]) -> Optional[str]:
    """
    Run the deposition workflow graph with automatic plan acceptance.
    
    Args:
        graph: The compiled workflow graph
        initial_input: Initial input for the graph
        thread: Thread configuration
        
    Returns:
        Optional[str]: The final markdown document if successful
    """
    from langgraph.types import Command
    
    try:
        # Start the graph execution
        graph_stream = graph.astream(initial_input, thread, stream_mode="values")
        
        # Process until we need human feedback
        async for event in graph_stream:
            status = event.get("status", "unknown")
            logger.info(f"Status: {status}")
            
            if event.get("deposition_summary") and event.get("processed_sections"):
                # Automatically accept the plan without feedback
                graph.update_state(
                    values={
                        "feedback_on_plan": None,
                        "accept_plan": True,
                    },
                    config=thread,
                    as_node="human_feedback"
                )
                break
        
        # Resume execution
        resume_command = Command(
            resume={
                "feedback_on_plan": None,
                "accept_plan": True
            }
        )
        
        # Continue streaming until we get the final state
        final_state = None
        async for event in graph.astream(resume_command, thread, stream_mode="values"):
            logger.info(f"Status: {event.get('status', 'unknown')}")
            if event:
                final_state = event
        
        if not final_state:
            raise ValueError("No final state received from graph execution")
        
        return final_state.get("markdown_document")
        
    except Exception as e:
        logger.error(f"Error during graph execution: {e}")
        raise

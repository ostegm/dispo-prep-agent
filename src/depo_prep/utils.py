import asyncio
from typing import List, Dict, Optional, Union
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
collection = chroma_client.get_collection(config.chroma_collection_name)

async def vector_db_search_async(search_queries: Union[str, List[str]]) -> List[Dict]:
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



async def get_full_document_text(filename: str) -> str:
    """
    Fetches and combines all chunks of a document from the vector database.
    
    Args:
        filename: Base name of the document file to retrieve (e.g. "complaint.pdf")
        
    Returns:
        str: The full text of the document
    """
    # Get all documents with this filename
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

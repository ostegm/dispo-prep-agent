import asyncio
from typing import List, Dict, Optional, Union
import chromadb
from chromadb.config import Settings
import openai
from openai import AsyncOpenAI
from src.report_maistro.state import Section
import json
import re

# Initialize Chroma client
chroma_client = chromadb.Client(Settings(persist_directory="./chroma_db"))
collection = chroma_client.get_or_create_collection("case_documents")

def deduplicate_and_format_sources(search_response, max_tokens_per_source, include_raw_content=True):
    """
    Takes either a single search response or list of responses from Tavily API and formats them.
    Limits the raw_content to approximately max_tokens_per_source.
    include_raw_content specifies whether to include the raw_content from Tavily in the formatted string.
    
    Args:
        search_response: Either:
            - A dict with a 'results' key containing a list of search results
            - A list of dicts, each containing search results
            
    Returns:
        str: Formatted string with deduplicated sources
    """
    # Convert input to list of results
    if isinstance(search_response, dict):
        sources_list = search_response['results']
    elif isinstance(search_response, list):
        sources_list = []
        for response in search_response:
            if isinstance(response, dict) and 'results' in response:
                sources_list.extend(response['results'])
            else:
                sources_list.extend(response)
    else:
        raise ValueError("Input must be either a dict with 'results' or a list of search results")
    
    # Deduplicate by URL
    unique_sources = {}
    for source in sources_list:
        if source['url'] not in unique_sources:
            unique_sources[source['url']] = source
    
    # Format output
    formatted_text = "Sources:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source['content']}\n===\n"
        if include_raw_content:
            # Using rough estimate of 4 characters per token
            char_limit = max_tokens_per_source * 4
            # Handle None raw_content
            raw_content = source.get('raw_content', '')
            if raw_content is None:
                raw_content = ''
                print(f"Warning: No raw_content found for source {source['url']}")
            if len(raw_content) > char_limit:
                raw_content = raw_content[:char_limit] + "... [truncated]"
            formatted_text += f"Full source content limited to {max_tokens_per_source} tokens: {raw_content}\n\n"
                
    return formatted_text.strip()

def format_sections(sections: list[Section]) -> str:
    """ Format a list of sections into a string """
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
{'='*60}
Section {idx}: {section.name}
{'='*60}
Description:
{section.description}
Requires Research: 
{section.research}

Content:
{section.content if section.content else '[Not yet written]'}

"""
    return formatted_str

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
        
        # Format results to match existing pipeline expectations
        formatted_results = []
        for i, (text, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            formatted_results.append({
                'title': f"Result {i+1} from {metadata.get('source', 'case documents')}",
                'content': text,
                'score': results['distances'][0][i]
            })
        return formatted_results
    
    # Run searches concurrently
    tasks = [search_single_query(query) for query in search_queries]
    results = await asyncio.gather(*tasks)
    
    return results

def extract_json_from_text(text: str) -> dict:
    """Extract JSON from text that may contain markdown or other formatting."""
    # First try direct JSON parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON between ```json markers
    json_pattern = r"```json\s*(.*?)\s*```"
    matches = re.findall(json_pattern, text, re.DOTALL)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    # Try to find anything that looks like a JSON object
    json_pattern = r"\{[\s\S]*\}"
    matches = re.findall(json_pattern, text)
    
    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue
    
    print("\nFailed to extract JSON from text. Response content:")
    print(text)
    print("\nPlease ensure the model returns a valid JSON object.")
    raise ValueError("No valid JSON found in text")

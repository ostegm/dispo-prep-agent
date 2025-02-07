"""Test the indexed data in Chroma."""

import chromadb
from chromadb.config import Settings
from src.report_maistro.configuration import Configuration
from pathlib import Path
import asyncio
from openai import AsyncOpenAI

async def test_indexed_data():
    """Verify the data indexed in Chroma."""
    
    # Initialize Chroma client
    config = Configuration()
    persist_dir = config.chroma_persist_dir
    print(f"\nChroma persist directory: {persist_dir}")
    print(f"Directory exists: {Path(persist_dir).exists()}")
    
    client = chromadb.Client(Settings(
        persist_directory=persist_dir,
        is_persistent=True
    ))
    
    print("\nCollections:")
    collections = client.list_collections()
    print(f"Found {len(collections)} collections")
    for collection in collections:
        print(f"  - {collection}")
    
    # Get our collection
    collection = client.get_collection("case_documents")
    print("\nAccessing collection 'case_documents'...")
    
    try:
        # Get all documents
        results = collection.get()
        
        print("\nChroma Index Summary:")
        print("-" * 80)
        print(f"Total chunks indexed: {len(results['ids'])}")
        
        if len(results['ids']) > 0:
            # Group chunks by source document
            chunks_by_doc = {}
            for doc, metadata in zip(results['documents'], results['metadatas']):
                source = metadata['source']
                if source not in chunks_by_doc:
                    chunks_by_doc[source] = []
                chunks_by_doc[source].append({
                    'chunk_index': metadata['chunk_index'],
                    'total_chunks': metadata['total_chunks'],
                    'word_count': len(doc.split())
                })
            
            # Print summary for each document
            for source, chunks in sorted(chunks_by_doc.items()):
                print(f"\nDocument: {source}")
                print(f"Number of chunks: {len(chunks)}")
                total_words = sum(chunk['word_count'] for chunk in chunks)
                print(f"Total words: {total_words}")
                print(f"Average words per chunk: {total_words/len(chunks):.1f}")
                print("\nChunk sizes:")
                for chunk in sorted(chunks, key=lambda x: x['chunk_index']):
                    print(f"  Chunk {chunk['chunk_index']+1}/{chunk['total_chunks']}: {chunk['word_count']} words")
                print("\nChunk contents:")
                for chunk in sorted(chunks, key=lambda x: x['chunk_index']):
                    print(f"\nChunk {chunk['chunk_index']+1}:")
                    print("-" * 40)
                    print(results['documents'][chunk['chunk_index']])
                    print("-" * 40)
            
            # Test search functionality
            print("\nTesting Search:")
            print("-" * 80)
            print("Searching for documents containing 'semi'...")
            
            # Get embedding for search query
            openai_client = AsyncOpenAI()
            response = await openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input="semi"
            )
            query_embedding = response.data[0].embedding
            
            # Search using the embedding
            search_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=5
            )
            
            if len(search_results['documents'][0]) > 0:
                print(f"\nFound {len(search_results['documents'][0])} matching chunks:")
                for i, (doc, metadata) in enumerate(zip(search_results['documents'][0], search_results['metadatas'][0])):
                    print(f"\nMatch {i+1} from {metadata['source']} (chunk {metadata['chunk_index']+1}/{metadata['total_chunks']}):")
                    print("-" * 40)
                    print(doc)
                    print("-" * 40)
            else:
                print("No matches found")
    except Exception as e:
        print(f"Error accessing collection: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_indexed_data()) 
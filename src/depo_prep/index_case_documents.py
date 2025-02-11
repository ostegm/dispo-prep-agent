"""Script for processing and indexing case documents into a vector database."""

import os
import base64
from io import BytesIO
from pathlib import Path
from typing import List, Optional, Tuple
import asyncio
from pdf2image import convert_from_path
from PIL import Image
from openai import AsyncOpenAI
import chromadb
from chromadb.config import Settings
from langchain_google_genai import ChatGoogleGenerativeAI
import re
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()

from src.depo_prep.configuration import Configuration

OCR_PROMPT = """\
OCR the following page into Markdown. Ignore line numbers, unnecessary spaces and line breaks.
Preserve the semantic structure and formatting of the document.
Do not surround your output with triple backticks and do not include the word "markdown" in your output.
"""

CHUNKING_PROMPT = """\
Split the following legal document into semantic chunks of roughly 250-1000 words each.
Each chunk should represent a coherent section or topic from the document.
Preserve any markdown formatting within the chunks.

Surround each chunk with <chunk> </chunk> html tags.
Do not surround your output with triple backticks and do not include the word "markdown" in your output.

Document to chunk:
"""

class DocumentProcessor:
    """Processes PDF documents and indexes them in a vector database."""
    
    def __init__(self, config: Optional[Configuration] = None):
        """Initialize the document processor.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or Configuration()
        self.openai_client = AsyncOpenAI()
        self.gemini = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-lite-preview-02-05",
            convert_system_message_to_human=False,
            temperature=0.0
        )
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=self.config.chroma_persist_dir,
            is_persistent=True
        ))
        self.collection = self.chroma_client.get_or_create_collection(
            self.config.chroma_collection_name
        )
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF pages to images.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            List[Image.Image]: List of PIL Image objects, one per page.
        """
        return convert_from_path(pdf_path)
    
    async def ocr_page(self, image: Image.Image, page_num: int) -> str:
        """OCR a single page using Gemini.
        
        Args:
            image: PIL Image object of the page.
            page_num: Page number for logging.
            
        Returns:
            str: OCR'd text in markdown format.
        """
        print(f"OCRing page {page_num}...")
        
        # Convert image to bytes and base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_bytes = buffered.getvalue()
        image_b64 = base64.b64encode(image_bytes).decode()
        
        # Process with Gemini
        response = await self.gemini.ainvoke([
            {"role": "system", "content": OCR_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
            ]}
        ])
        
        return response.content
    
    async def chunk_text(self, text: str) -> List[str]:
        """Split text into semantic chunks using Gemini.
        
        Args:
            text: Text to split into chunks.
            
        Returns:
            List[str]: List of text chunks.
        """
        print("Chunking document...")
        
        # Process with Gemini
        response = await self.gemini.ainvoke([
            {"role": "system", "content": CHUNKING_PROMPT},
            {"role": "user", "content": text}
        ])
        
        # Extract chunks
        chunks = re.findall(r'<chunk>(.*?)</chunk>', response.content, re.DOTALL)
        chunks = [chunk.strip() for chunk in chunks]
        print(f"Found {len(chunks)} chunks")
        return chunks
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts using OpenAI's API.
        
        Args:
            texts: List of text chunks to get embeddings for.
            
        Returns:
            List[List[float]]: List of embeddings.
        """
        embeddings = []
        # Process in batches of 100 to avoid rate limits
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = await self.openai_client.embeddings.create(
                model=self.config.embedding_model,
                input=batch
            )
            embeddings.extend([e.embedding for e in response.data])
        return embeddings
    
    async def process_document(self, pdf_path: str, max_pages: int = None) -> None:
        """Process a single PDF document.
        
        Args:
            pdf_path: Path to the PDF file.
            max_pages: Maximum number of pages to process. If None, process all pages.
        """
        pdf_name = os.path.basename(pdf_path)
        
        # Check if document has already been processed
        existing_docs = self.collection.get(
            where={"source": pdf_name},
            limit=1
        )
        
        if existing_docs and existing_docs['ids']:
            print(f"\nSkipping {pdf_path} - already processed")
            return
            
        print(f"\nProcessing {pdf_path}...")
        
        # Convert PDF to images
        print("Converting PDF to images...")
        images = self.convert_pdf_to_images(pdf_path)
        if max_pages:
            images = images[:max_pages]
        print(f"Found {len(images)} pages")
        
        # OCR each page concurrently
        print("OCRing pages in parallel...")
        page_texts = await asyncio.gather(*[
            self.ocr_page(image, i+1)
            for i, image in enumerate(images)
        ])
        print("OCR completed")
        
        # Combine all pages with page breaks
        print("Combining pages...")
        full_text = "\n\n---\n\n".join(page_texts)
        print(f"Combined text length: {len(full_text)} characters")
        
        # Chunk the full document
        print("Chunking document...")
        chunks = await self.chunk_text(full_text)
        
        if not chunks:
            print(f"Warning: No chunks found in {pdf_path}")
            return
        
        print(f"Found {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  Chunk {i+1}: {len(chunk.split())} words")
            print("  Preview:", chunk[:100].replace('\n', ' '), "...")
        
        # Get embeddings
        print("Generating embeddings...")
        embeddings = await self.get_embeddings(chunks)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Create metadata for each chunk
        metadatas = [{
            "source": os.path.basename(pdf_path),
            "chunk_index": i,
            "total_chunks": len(chunks)
        } for i in range(len(chunks))]
        
        # Add to Chroma
        print("Adding to Chroma...")
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=[f"{os.path.basename(pdf_path)}_{i}" for i in range(len(chunks))]
            )
            print("Successfully added to Chroma")
        except Exception as e:
            print(f"Error adding to Chroma: {e}")
            raise
        
        print(f"Finished processing {pdf_path} - indexed {len(chunks)} chunks")

async def main():
    """Main function to process all PDFs in the case_documents directory."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process and index case documents.')
    parser.add_argument('--clear', action='store_true', 
                       help='Clear existing collection before processing documents')
    args = parser.parse_args()
    
    # Get the case_documents directory relative to the workspace root
    workspace_root = Path(__file__).parent.parent.parent
    case_docs_dir = workspace_root / "case_documents"
    
    # Create directory if it doesn't exist
    case_docs_dir.mkdir(exist_ok=True)
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Clear collection if requested
    if args.clear:
        print("\nClearing existing collection...")
        # Delete all documents by matching any source
        processor.collection.delete(where={"source": {"$ne": ""}})
        print("Collection cleared")
    
    # Process all PDFs
    pdf_files = list(case_docs_dir.glob("*.pdf"))
    if not pdf_files:
        print("No PDF files found in case_documents directory")
        return
    
    # Process all PDFs
    for pdf_file in pdf_files:
        await processor.process_document(str(pdf_file))
    print("Successfully processed all documents")

if __name__ == "__main__":
    asyncio.run(main()) 
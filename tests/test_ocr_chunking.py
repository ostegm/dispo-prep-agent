"""Test OCR and chunking functionality using Gemini."""

import asyncio
from pathlib import Path
import time
from src.report_maistro.index_case_documents import DocumentProcessor, OCR_PROMPT, CHUNKING_PROMPT
from src.report_maistro.configuration import Configuration

async def test_document_processing():
    """Process a PDF document and verify OCR and chunking."""
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Get test PDF path
    workspace_root = Path(__file__).parent.parent
    case_docs_dir = workspace_root / "case_documents"
    pdf_files = list(case_docs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in case_documents directory")
        return
        
    # Use the first PDF found
    test_pdf = pdf_files[0]
    print(f"\nTesting with file: {test_pdf}\n")
    
    # Convert PDF to images
    print("Converting PDF to images...")
    start_time = time.time()
    images = processor.convert_pdf_to_images(str(test_pdf))
    if not images:
        print("Failed to convert PDF to images")
        return
    print(f"Found {len(images)} pages")
    print(f"Conversion took {time.time() - start_time:.2f} seconds")
    
    # Test OCR
    print("\nTesting OCR:")
    print("-" * 80)
    print(f"Using OCR prompt:\n{OCR_PROMPT}\n")
    
    # OCR all pages concurrently
    print(f"Processing {len(images)} pages in parallel...")
    start_time = time.time()
    page_texts = await asyncio.gather(*[
        processor.ocr_page(image, i+1)
        for i, image in enumerate(images)
    ])
    ocr_time = time.time() - start_time
    print(f"\nOCR completed in {ocr_time:.2f} seconds")
    print(f"Average {ocr_time/len(images):.2f} seconds per page")
        
    # Test chunking
    print("\nTesting Chunking:")
    print("-" * 80)
    print(f"Using chunking prompt:\n{CHUNKING_PROMPT}\n")
    
    # Combine pages and chunk
    print("Combining pages and chunking...")
    start_time = time.time()
    full_text = "\n\n---\n\n".join(page_texts)
    
    print("\nFull text to be chunked:")
    print("=" * 80)
    print(full_text)
    print("=" * 80)
    
    chunks = await processor.chunk_text(full_text)
    chunk_time = time.time() - start_time
    print(f"Chunking completed in {chunk_time:.2f} seconds")
    
    print(f"\nFound {len(chunks)} chunks:")
    total_words = 0
    for i, chunk in enumerate(chunks, 1):
        word_count = len(chunk.split())
        total_words += word_count
        print(f"\nChunk {i}:")
        print("=" * 80)
        print(f"Word count: {word_count}")
        print("-" * 40)
        print(chunk)
        print("=" * 80)
    
    print(f"\nProcessing Summary:")
    print("-" * 80)
    print(f"Total pages: {len(images)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total words: {total_words}")
    print(f"Average words per chunk: {total_words/len(chunks):.1f}")
    print(f"OCR time: {ocr_time:.2f} seconds")
    print(f"Chunking time: {chunk_time:.2f} seconds")
    print(f"Total processing time: {ocr_time + chunk_time:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(test_document_processing()) 
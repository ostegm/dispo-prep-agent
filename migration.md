# Migration Plan for Deposition Preparation

## 1. Overview of the New Use Case

### Domain Shift
Transform the tool from a report-generation assistant into a deposition preparation assistant for trial attorneys.

- **Input:** A deposition topic provided by the user
- **Planning:** The planning model generates a deposition plan (a set of key deposition topics or lines of questioning)
- **Context Retrieval:** Each deposition topic is used to search a vector database that contains chunked text from case documents (e.g., PDF deposition transcripts or briefs)
- **Content Generation:** A downstream model (using Gpt4o-mini) generates deposition preparation content based on the retrieved context

### New Data Pipeline
- A new folder `case_documents` contains several PDF files
- A separate utility script (e.g., `index_case_documents.py`) will process PDFs:
  - Send these PDFs to Gpt4o-mini for text extraction
  - Chunk the extracted text appropriately
  - Compute OpenAI embeddings for each chunk
  - Index and persist them using Chroma with local storage

## 2. Changes to the Search Workflow

### Remove Web-Based Search
- Remove all references to Tavily API and web search functions

### Introduce Vector DB Search
Create a new asynchronous function (e.g., `vector_db_search_async`) that:
- Accepts a search query or list of queries
- Computes query embeddings using OpenAI embeddings
- Uses these embeddings to retrieve the top relevant document chunks from the locally persisted Chroma vector database
- Returns the search results formatted so they integrate with the rest of the pipeline (including using current deduplication/formatting functions if applicable)

### Graph Node Updates
- In the planning and deposition topics generation node, remove the call to `tavily_search_async` and replace it with `vector_db_search_async`
- Update any variable or configuration names that refer to web search (e.g., remove `tavily_topic` and `tavily_days`) to avoid confusion

## 3. Utility Script for Processing Case Documents

### Purpose
Create a separate utility script (e.g., `index_case_documents.py`) that:
- Scans the `case_documents` folder for PDF files
- For each PDF:
  - Uses Gpt4o-mini to extract text (this will involve sending the PDF or its contents to the extraction API)
  - Chunks the extracted text (using your preferred chunking method or LangChain's text splitters)
  - Computes OpenAI embeddings for each chunk
  - Indexes the chunks into a Chroma vector database set up for local persistent storage
- This script should be idempotent, so re-running it updates the index as new or modified PDFs appear

### Dependencies
- Chroma client (with local persistence)
- OpenAI embeddings package, possibly using LangChain's wrappers
- A PDF processing library (e.g., PyMuPDF, PDFPlumber, or similar)
- Gpt4o-mini integration details (assuming it provides an API for text extraction)

## 4. Prompt and Model Adjustments

### Prompt Updates

#### Planning Prompts
Update the instructions in `src/report_maistro/prompts.py` so that the planning model creates a "deposition plan" rather than a generic report plan:
- Change keywords such as "report" and "section" to "deposition" and "deposition topic" where applicable
- Maintain the structured output approach (e.g., a list of deposition topics, with fields such as name, description, plan, investigation, and content)

#### Writing Prompts
Similarly, update prompts used in the writing nodes to generate deposition preparation content:
- Adjust language to refer to deposition preparation, potential lines of questioning, examination topics, etc.

## 5. Graph Workflow Adjustments

### Node Renaming / Minimal Changes
- For the initial migration, **keep the existing node and state names** (e.g., `generate_report_plan`, `write_section`) to keep structural changes minimal
- Internally, update the calls so they use the new vector search function and updated prompts
- Future cleanup can introduce renaming (e.g., `generate_deposition_plan`, `write_deposition_topic`) for clarity

### Configuration Changes
- In `src/report_maistro/configuration.py`, remove or repurpose fields related to web search (`tavily_topic`, `tavily_days`) if they become unnecessary
- Optionally add new configuration fields for Chroma (e.g., local storage path), if needed
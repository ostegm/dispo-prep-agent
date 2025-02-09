# Report mAIstro

Report mAIstro is an open-source research assistant that helps prepare for legal depositions by analyzing case documents and generating comprehensive deposition outlines. It combines document analysis, semantic search, and structured writing with human oversight.

Key features:
- Uses Google's Gemini model for OCR and document chunking
- Leverages Chroma vector database for semantic search of case documents
- Enables human review and iteration of the deposition plan
- Parallelizes research across multiple deposition topics
- Produces well-formatted markdown deposition outlines
- Supports customizable models, prompts, and deposition structure

## 🚀 Quickstart

1. Clone the repository:
```bash
git clone https://github.com/langchain-ai/report_maistro.git
cd report_maistro
```

2. Set up your environment:
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv pip install -e .
```

3. Set up your API keys:
```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:
```bash
export OPENAI_API_KEY=<your_openai_api_key>  # For embeddings
export ANTHROPIC_API_KEY=<your_anthropic_api_key>  # For content generation
export GOOGLE_API_KEY=<your_google_api_key>  # For OCR and chunking
```

## 📚 Indexing Case Documents

1. Place your case documents (PDFs) in the `case_documents/` directory.

2. Run the indexing script:
```bash
python -m src.report_maistro.index_case_documents
```

This will:
- Use Gemini to OCR and extract text from PDFs
- Split documents into semantic chunks
- Compute embeddings using OpenAI's embedding model
- Store everything in a local Chroma vector database

You can verify the indexed documents by running:
```bash
python -m tests.test_chroma_index
```

## 🏃‍♂️ Running the Code

1. Start the LangGraph server:
```bash
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```

2. Access the interfaces:
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs

3. Run the test workflow:
```bash
python -m tests.test_deposition_agent
```

## 📖 How it Works

1. **Document Processing**
   - PDFs are converted to images and OCR'd using Gemini
   - Text is split into semantic chunks (250-1000 words)
   - Chunks are embedded and stored in Chroma

2. **Deposition Planning**
   - System analyzes the deposition topic and generates structured sections
   - Each section is marked for investigation if it requires document research
   - Human review/feedback can be provided on the plan

3. **Document Research**
   - System generates semantic search queries for each section
   - Queries are run against the vector database
   - Relevant document chunks are retrieved and deduplicated

4. **Content Generation**
   - System writes questions for each section using retrieved documents
   - Background sections are written with standard questions
   - Final sections combine insights from research

## 🔧 Customization

The system can be customized through:
- `src/report_maistro/configuration.py`: Model selection, chunk sizes, etc.
- `src/report_maistro/prompts.py`: Instructions for each step
- `DEFAULT_DEPOSITION_STRUCTURE`: Organization of the deposition

## 🧪 Testing

Run individual test files:
```bash
# Test document processing
python -m tests.test_ocr_chunking

# Test vector database
python -m tests.test_chroma_index

# Test full workflow
python -m tests.test_deposition_agent
```



## Next Steps

- Fix the local testing script - it works when I use langgraph.
- Investigate chunking vs sending all documents with the initial planning step.
- Move all prompt related stuff to prompts.py
- Remove print statements
- Figure out if we really need to fan out to sections or can we just do it as one step?
- Convert my back medical record summary to a injury related to this case and index. 
- Checkout email from Nick https://mail.google.com/mail/u/0/#inbox/FMfcgzQZTCjdhFBklTDhxZQBJFsvFBjt

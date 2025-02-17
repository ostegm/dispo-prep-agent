# Depo Prep Agent

Depo Prep Agent is an agent that helps prepare for legal depositions by analyzing case documents and generating comprehensive deposition outlines. It combines document analysis, semantic search, and structured writing with human oversight.

Key features:
- Semantic search of case documents using Chroma vector database
- Human review and iteration of deposition plans
- Parallel processing of search queries
- Well-formatted markdown deposition outlines
- Customizable models, prompts, and deposition structure

## 🚀 Quickstart

1. Clone the repository:

2. Set up your environment:
```bash
# Create and activate a virtual environment
uv venv 
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install uv package manager
brew install uv

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
python -m src.depo_prep.index_case_documents
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
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev --host 0.0.0.0 --port 2024
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
   - Gemini analyzes the topic and generates structured sections
   - Each section includes targeted search queries
   - Human review/feedback loop available

3. **Document Research**
   - System runs parallel semantic searches for each query
   - Results are structured and organized by section
   - Relevant document chunks are retrieved

4. **Content Generation**
   - Claude compiles search results into a structured report
   - Gemini generates specific deposition questions
   - Final output combines document insights with expert questioning

## 🔧 Customization

The system can be customized through:
- `src/depo_prep/configuration.py`: Model selection and search parameters
- `src/depo_prep/prompts.py`: Instructions for each step
- `src/depo_prep/state.py`: Data structures and state management

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
- Investigate hosting options ([self host](https://langchain-ai.github.io/langgraph/how-tos/deploy-self-hosted) in docker)
- Write a script to run the CLI tool using graph.stream directly: https://github.com/langchain-ai/open_deep_research/blob/main/src/open_deep_research/graph.ipynb
- Determine if REST API vs graph.invoke is best serving approach. 
- Run the dodona server localy (comment out some stuff?)
- Map out how to add a single graph to the dodona backened.
   - Add a new route in the backend for triggering the grpah
   - Add a page in the frontend which triggers the graph, allows chat interaction and then shows the Markdown output in the UI (editable)
   - Export to PDF
# Report mAIstro

Report mAIstro is an open-source research assistant that generates comprehensive reports on any topic, following a workflow similar to Google's [Gemini Deep Research](https://blog.google/products/gemini/google-gemini-deep-research/). It combines planning, parallel web research, and structured writing with human oversight.

Key features:
- Uses OpenAI o-series reasoning model (default) for intelligent report planning
- Enables human review and iteration of the research plan
- Parallelizes web research across multiple report sections, using Claude-3.5-Sonnet for report writing
- Produces well-formatted markdown reports
- Supports customizable models, prompts, and report structure

## 🚀 Quickstart

Clone the repository:
```bash
git clone https://github.com/langchain-ai/report_maistro.git
cd report_maistro
```

Set API keys for Anthropic (default writer), OpenAI (default planner), and [Tavily](https://tavily.com) for free web search up to 1000 requests):

```bash
cp .env.example .env
```

Edit the `.env` file with your API keys:

```bash
export TAVILY_API_KEY=<your_tavily_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>
export OPENAI_API_KEY=<your_openai_api_key>
```

Launch the assistant with the LangGraph server, which will open in your browser:

#### Mac

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and start the LangGraph server
uvx --refresh --from "langgraph-cli[inmem]" --with-editable . --python 3.11 langgraph dev
```


Use this to open the Studio UI:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

## 📖 Customizing the report

Optionally, provide a description of the report structure you want as a configuration. You can further tune this during the feedback phase. While a topic alone can generate reports, we found that providing a structure significantly improves quality. For example, business strategy reports might need case studies, while comparative analyses benefit from structured comparison tables. The natural language structure acts as a flexible template, guiding the AI to create more focused and relevant reports.

> See [some example report types here](report_examples/)!

## Motivation 

Automating research and report writing is a common need. [Deep Research](https://blog.google/products/gemini/google-gemini-deep-research/) from Google is a great example of this. This open source project mirror the flow of Deep Research, but allow you to customize the models, prompts, and research report structure.

## How it works
   
1. `Plan and Execute` - Report mAIstro follows a [plan-and-execute workflow](https://github.com/assafelovic/gpt-researcher) that separates planning from research, allowing for better resource management, human-in-the-loop approval, and significantly reducing overall report creation time:

   - **Planning Phase**: An LLM analyzes the user's `topic` and `structure` using a planning prompt to create the report sections first. 
   - **Research Phase**: The system parallelizes web research across all sections requiring external data:
     - Uses [Tavily API](https://tavily.com/) for targeted web searches
     - Processes multiple sections simultaneously for faster report generation
     - Synthesizes gathered information into coherent section content
   
2. `Sequential Writing` - The report generation follows a logical sequence:
   - First, completes all research-dependent sections in parallel
   - Then generates connecting sections like introductions and conclusions
   - Uses insights from research sections to create cohesive narratives
   - Maintains contextual awareness across all sections
   
   While this sequence can be customized via the `structure`, the default flow ensures that conclusions meaningfully incorporate research findings.

3. `Managing different types` - Report mAIstro is built on LangGraph, which has native support for configuration management [using assistants](https://langchain-ai.github.io/langgraph/concepts/assistants/). The report `structure` is a field in the graph configuration, which allows users to create different assistants for different types of reports. 

## UX

### Local deployment

Follow the [quickstart](#quickstart) to run the assistant locally.

### Hosted deployment
 
You can easily deploy to [LangGraph Platform ](https://langchain-ai.github.io/langgraph/concepts/#deployment-options). 

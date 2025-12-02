# VNPT AI RAG Pipeline

High-performance RAG (Retrieval-Augmented Generation) pipeline for answering Vietnamese multiple-choice questions using LangGraph and agentic workflows.

## Architecture

The pipeline uses a **LangGraph StateGraph** with intelligent routing:

- **Router Node**: Classifies questions into `knowledge`, `math`, or `toxic` categories
- **Knowledge RAG Node**: Retrieves relevant context from Qdrant vector store for history/culture questions
- **Logic Solver Node**: Uses Python Code Interpreter (local REPL) for math and reasoning questions
- **Safety Guard Node**: Handles sensitive/toxic content with refusal responses

## Quick Start

### Prerequisites

- Python ≥3.10
- [uv](https://github.com/astral-sh/uv) package manager
- Google AI API key (Gemini)

### Installation

```bash
# Clone repository
git clone https://github.com/duongtruongbinh/vnpt-ai
cd vnpt-ai

# Install dependencies
uv sync

# Edit .env and add your GOOGLE_API_KEY
```

### Usage

```bash
# Generate dummy test data
uv run python data/generate_dummy_data.py

# Run pipeline
uv run python main.py
```

Input: `data/public_test.csv` or `data/private_test.csv`  
Output: `data/pred.csv` or `/output/pred.csv` (in production)

## Project Structure

```
vnpt-ai/
├── src/
│   ├── graph.py          # LangGraph workflow definition
│   ├── config.py         # Configuration settings
│   ├── nodes/
│   │   ├── router.py     # Question classification
│   │   ├── rag.py        # Knowledge retrieval
│   │   └── logic.py      # Code agent for math
│   └── utils/
│       └── ingestion.py  # Vector store ingestion
├── data/                 # Test data and knowledge base
├── main.py               # Entry point
└── pyproject.toml        # Dependencies
```

## Technologies

- **LangGraph**: Workflow orchestration
- **LangChain**: LLM integration and tooling
- **Google Gemini**: LLM (gemini-2.5-flash-lite)
- **Qdrant**: Vector database
- **HuggingFace**: Vietnamese embeddings (bkai-foundation-models/vietnamese-bi-encoder)
- **Python REPL**: Local code execution for calculations

## License

MIT


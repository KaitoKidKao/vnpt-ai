# VNPT AI RAG Pipeline

High-performance Agentic RAG Pipeline designed for the VNPT AI Hackathon (Track 2).

This project implements a modular, model-agnostic workflow using **LangGraph** to intelligently route questions, execute Python code for complex reasoning, and retrieve knowledge from a persistent vector store. It is engineered for high accuracy, fault tolerance, and API quota efficiency.

## 🚀 Key Features

- **Agentic Workflow**: Utilizes a **Router Node** to classify questions into distinct domains (Math, Knowledge, or Toxic) and routes them to specialized solvers.
- **Program-Aided Language Models (PAL)**:
  - Solves math and logic problems by generating and executing Python code via a local REPL.
  - **Self-Correction Loop**: The logic solver iteratively executes code, captures output, and feeds it back to the LLM to correct errors or format the final answer (up to 5 retry steps).
- **Quota Optimization**:
  - **Tiered Modeling Architecture**: Supports using lightweight "Small" models for routing and "Large" models for deep reasoning/RAG.
  - **Smart Caching**: Implements local disk caching for **Qdrant** to prevent redundant re-embedding of the knowledge base.
- **Responsible AI**: Robust safety guardrails to detect and refuse toxic, dangerous, or politically sensitive content based on Vietnamese context.

## 🏗️ Architecture

The pipeline is orchestrated by a **LangGraph StateGraph**:

```mermaid
graph TD
    Start([Input Question]) --> RouterNode{Router Node<br/>Small Model}
    
    RouterNode -- "Math/Logic" --> LogicSolver[Logic Solver - Code Agent<br/>Large Model]
    RouterNode -- "History/Culture" --> KnowledgeRAG[Knowledge RAG - Retrieval<br/>Large Model]
    RouterNode -- "Toxic/Sensitive" --> SafetyGuard[Safety Guard - Refusal]
    
    subgraph "Knowledge Processing"
        KnowledgeRAG <--> VectorDB[(Qdrant Local Disk)]
        VectorDB <..- IngestionScript[Ingestion Logic<br/>Persistent Cache]
    end
    
    subgraph "Logic Processing"
        LogicSolver <--> PythonREPL[Python Interpreter<br/>Iterative Execution]
    end
    
    LogicSolver --> End([Final Answer])
    KnowledgeRAG --> End
    SafetyGuard --> End
````

### Components

1.  **Router Node**: A classifier using a small LLM to categorize inputs.
2.  **Logic Solver**: A Code Agent that extracts Python code from LLM responses, executes it locally, and parses the standard output to find the final answer. It includes error handling and retry logic.
3.  **Knowledge RAG**: Retrieves relevant context from the Qdrant vector store and generates answers using the large LLM.
4.  **Safety Guard**: A deterministic sink node that provides standard refusal responses for content classified as "Toxic".

## 🛠️ Tech Stack

| Component | Implementation |
| :--- | :--- |
| **Orchestration** | LangGraph, LangChain |
| **Package Manager** | uv |
| **Vector DB** | Qdrant (Local Persistence) |
| **Embedding** | BKAI Vietnamese Bi-encoder |
| **Code Execution** | LangChain Experimental PythonREPL |
| **Models** | Configurable via `.env` (Default: Qwen-4B) |

## ⚡ Quick Start

### Prerequisites

  - Python ≥3.10
  - [uv](https://github.com/astral-sh/uv) (Recommended for fast dependency management)
  - CUDA-capable GPU (Recommended for local inference)

### Installation

1.  **Clone the repository**

    ```bash
    git clone https://github.com/duongtruongbinh/vnpt-ai
    cd vnpt-ai
    ```

2.  **Install dependencies**

    ```bash
    uv sync
    ```

3.  **Configure Environment (Optional)**
    Create a `.env` file to point to your specific local model paths.

    ```env
    # Example .env
    LLM_MODEL_SMALL=/path/to/your/small/model
    LLM_MODEL_LARGE=/path/to/your/large/model
    EMBEDDING_MODEL=bkai-foundation-models/vietnamese-bi-encoder
    ```

### Usage

**1. Generate Dummy Data (Optional)**
If you don't have the official dataset yet, generate sample questions and a knowledge base:

```bash
uv run python scripts/generate_data.py
```

**2. Collect & Ingest Data (Optional)**
Expand your knowledge base by crawling websites or adding local documents.

* Crawl Data: Fetch content from websites using the crawler CLI.
  ```bash
  # Example: Crawl a website filtering by topic
  uv run python scripts/crawl.py --url https://example.com --mode links --topic "Vietnam History"
  ```

* Ingest Data: Load crawled JSON files or local documents (PDF, DOCX, TXT) into the Qdrant vector store.
  ```bash
  # Ingest crawled data (use --append to keep existing data)
  uv run python scripts/ingest.py data/crawled/*.json --append

  # Ingest a folder of documents
  uv run python scripts/ingest.py --dir data/documents --append
  ```

**3. Run the Pipeline**
The system automatically handles vector ingestion.

  * **First run:** Embeds `knowledge_base.txt` and saves to `data/qdrant_storage`.
  * **Subsequent runs:** Loads directly from disk (Instant startup).

```bash
uv run python main.py
```

  * **Input Priority:** Checks for `data/private_test.csv` first, then falls back to `data/public_test.csv`.
  * **Output:** Results are saved to `data/pred.csv`.

## 📂 Project Structure

```
vnpt-ai/
├── data/                 
│   ├── qdrant_storage/   # Persistent Vector DB (Git ignored)
│   ├── crawled/          # Crawled website data (JSON)
│   ├── knowledge_base.txt
│   └── public_test.csv
├── scripts/
│   ├── __init__.py
│   ├── crawl.py          # Web crawler CLI script
│   ├── ingest.py         # Data ingestion CLI script
│   └── generate_data.py # Generate dummy test data
├── src/
│   ├── graph.py          # LangGraph workflow definition
│   ├── config.py         # Configuration & Environment loading
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── router.py     # Classification Logic
│   │   ├── rag.py        # Retrieval & Safety Logic
│   │   └── logic.py      # Python Code Agent Logic
│   └── utils/
│       ├── llm.py        # HuggingFace Model Loading
│       ├── ingestion.py  # Qdrant Ingestion & Caching
│       └── web_crawler.py # Web crawler utilities
├── main.py               # Application Entry Point
└── pyproject.toml        # Dependencies & Project Metadata
```
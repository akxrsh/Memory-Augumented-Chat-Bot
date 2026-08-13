# Memory-Augmented Hybrid RAG & Knowledge Graph Chatbot System

An enterprise-grade, stateful AI Assistant built from scratch using **FastAPI**, **LangGraph**, **ChromaDB**, **MongoDB**, and **Neo4j**. 

This system features long-term user memory (fact extraction, importance scoring, and deduplication), a Hybrid RAG pipeline (dense vector embeddings + BM25 keyword search combined with Reciprocal Rank Fusion), a Knowledge Graph extraction/reasoning layer, dynamic tool execution, and an LLM-as-a-judge evaluation loop to prevent hallucinations.

---

## System Architecture

```mermaid
graph TD
    User([User]) <--> Frontend[React/Tailwind Client Dashboard]
    Frontend <--> FastAPI[FastAPI Backend]
    
    subgraph Agentic Brain (LangGraph)
        FastAPI <--> Agent[Stateful StateGraph]
        Agent --> MemoryRetrieve[Retrieve Memory Node]
        Agent --> Router[Intent Detection Router]
        
        Router -->|RAG| RAGNode[Hybrid RAG Retrieval Node]
        Router -->|Knowledge Graph| GraphNode[Knowledge Graph Node]
        Router -->|Tool Call| ToolNode[Tool Execution Node]
        Router -->|Direct Chat| GenNode[Generation Node]
        
        RAGNode --> GenNode
        GraphNode --> GenNode
        ToolNode --> GenNode
        
        GenNode --> EvalNode[Evaluation Judge Node]
        EvalNode -->|Fail Groundedness| GenNode
        EvalNode -->|Pass / Max Retry| EndNode([Return response])
    end

    subgraph Database Layers
        MemoryRetrieve <--> MongoDB[(MongoDB / SQLite Fallback)]
        RAGNode <--> Chroma[(ChromaDB)]
        GraphNode <--> Neo4j[(Neo4j / SQLite Fallback)]
    end

    subgraph Tool Sandbox
        ToolNode --> PythonREPL[Python Executor]
        ToolNode --> Wikipedia[Wikipedia API]
        ToolNode --> DuckDuckGo[DuckDuckGo Search]
        ToolNode --> ArXiv[ArXiv API]
    end
```

---

## Features

1. **Long-Term User Memory:** Dynamically extracts user statements (name, preference, job, goals) in the background. Scores fact importance (1-10) and merges new facts with existing ones using semantic deduplication to avoid redundancies and conflicts.
2. **Hybrid RAG Pipeline:** Indexes uploaded PDFs, CSVs, Markdown, and scraped web pages. Matches user query using dense vector similarity and sparse BM25 keyword matching, ranking the merged results using Reciprocal Rank Fusion (RRF).
3. **Knowledge Graph (Neo4j):** Extracts entities and relationship tuples from documents using LLMs and inserts them into Neo4j. Routes connection queries (e.g. "what frameworks are related to LangChain?") to graph lookups.
4. **Self-Correcting LangGraph Agent:** Implements a stateful workflow that routes queries, executes parallel tools, builds dynamic context prompts, and checks output faithfulness using a retry evaluation loop.
5. **Real-time Tools:** Secure Python REPL code sandbox, DuckDuckGo Search, Wikipedia query, ArXiv retrieval, Datetime helper, and Currency exchange conversions.
6. **Unified Dashboard:** Custom glassmorphic React dark-mode dashboard providing a streaming chat UI, document uploader/web-scraper, memory vault explorer, force-directed graph renderer, and Ragas quality analytics indicators.
7. **Offline DB Fallback:** Bootable out of the box! If MongoDB or Neo4j are not running, the backend seamlessly routes document operations and nodes to a local SQLite fallback database.

---

## File Structure

```
project/
├── backend/
│   ├── api/                 # Endpoint routers (auth, chat, memory, docs, graph, eval)
│   ├── services/            # Stateful Agent, Auth, and LLM Initializer
│   ├── memory/              # Memory manager & conflict resolver
│   ├── rag/                 # Text cleaner, Chroma client, and Hybrid search indexer
│   ├── graph/               # Graph database clients and entity extractor
│   ├── tools/               # Secure Python REPL, DDG, Wikipedia, and Arxiv wrappers
│   ├── prompts/             # System templates
│   ├── evaluation/          # LLM-as-a-judge scorers
│   ├── models/              # DAO database drivers
│   ├── schemas/             # Pydantic request/response validations
│   ├── database/            # Connection initializers
│   ├── utils/               # App logger
│   └── main.py              # FastAPI entrypoint
├── frontend/
│   └── index.html           # React glassmorphic dashboard
├── docker/
│   ├── backend.Dockerfile
│   └── docker-compose.yml
├── datasets/
│   └── langchain_docs.md    # Sample documents for RAG testing
├── .env                     # App environment configuration
└── requirements.txt         # Package dependencies
```

---

## Setup & Execution

### Prerequisites
- Python 3.11+
- [Docker & Docker Compose](https://www.docker.com/products/docker-desktop/) (Optional, but recommended)

### Quick Start (Local Development)

1. **Clone the Repository and install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables:**
   Rename `.env.example` to `.env` and fill in your OpenAI key:
   ```env
   OPENAI_API_KEY=sk-proj-...
   ```
   *Note: If no API key is specified, the system will initialize in mock-fallback mode for dry runs.*

3. **Start the FastAPI Backend:**
   ```bash
   python backend/main.py
   ```
   The backend will start on [http://localhost:8000](http://localhost:8000). The Swagger interactive docs are at `/docs`.

4. **Access the Client Dashboard:**
   Open your browser and navigate to:
   ```
   http://localhost:8000/frontend/
   ```

---

### Run with Docker Compose

To start MongoDB, Neo4j, and the FastAPI application in Docker containers:

1. **Navigate to the docker folder:**
   ```bash
   cd docker
   ```

2. **Run docker-compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access endpoints:**
   - Client Dashboard: [http://localhost:8000/frontend/](http://localhost:8000/frontend/)
   - Neo4j Browser Console: [http://localhost:7474](http://localhost:7474) (Username: `neo4j`, Password: `password`)

---

## API Documentation Summary

| Method | Endpoint | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| **POST** | `/api/v1/auth/register` | Register a new user | No |
| **POST** | `/api/v1/auth/login` | Login and acquire JWT Token | No |
| **POST** | `/api/v1/chat` | Send a query to the LangGraph agent | Yes |
| **GET** | `/api/v1/chat/sessions` | Fetch all conversation sessions for user | Yes |
| **GET** | `/api/v1/chat/sessions/{id}` | Load messages for a session | Yes |
| **GET** | `/api/v1/memory` | Retrieve compiled user memories | Yes |
| **POST** | `/api/v1/memory/add` | Manually insert a user fact | Yes |
| **DELETE**| `/api/v1/memory/{id}` | Delete a specific memory fact | Yes |
| **POST** | `/api/v1/documents/upload`| Upload and index file chunks (PDF, CSV, MD, TXT)| Yes |
| **POST** | `/api/v1/documents/scrape`| Scrape URL and index web page | Yes |
| **POST** | `/api/v1/graph/build` | Extract and build graph nodes from text | Yes |
| **GET** | `/api/v1/graph/neighborhood`| Fetch neighborhood nodes & edges for graph visualization | Yes |
| **GET** | `/api/v1/evaluate/metrics`| Retrieve aggregated Ragas metrics | Yes |
| **GET** | `/health` | Verify system health | No |

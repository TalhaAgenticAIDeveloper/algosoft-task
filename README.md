# AI-Powered Assessment Creation Agent

An end-to-end system that helps teachers generate curriculum-aligned assessments (MCQs, short-answer, and long-answer questions) through a conversational interface, powered by LLMs, LangGraph multi-agent orchestration, and semantic search.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE (offline)                     │
│                                                                 │
│  LO.json ──→ LO_cleaning.py ──→ clean_los.json                 │
│  chunks.json ──→ cleaning_chunks.py ──→ chunks_clean.json       │
│                          ↓                                      │
│         match_los_to_chunks.py (embedding + cosine similarity)  │
│                          ↓                                      │
│                   lo_with_chunks.json                            │
│              (each LO mapped to top-3 chunks)                   │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│                   RUNTIME APPLICATION                            │
│                                                                 │
│  vectorstore.py ─ FAISS index over LO descriptions              │
│  graph.py ─────── 9-agent LangGraph workflow (Groq LLM)        │
│  state.py ─────── Shared AssessmentState (TypedDict)            │
│  main.py ──────── FastAPI server (sessions + /answer/ endpoint) │
└─────────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────────┐
│               WEB UI (static HTML/JS/CSS)                       │
│                                                                 │
│  Chat interface → POST /answer/ → response + stage display      │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **LLM**: Groq (`openai/gpt-oss-120b`) via LangChain
- **Orchestration**: LangGraph (multi-agent state machine)
- **Vector Store**: FAISS with `all-MiniLM-L6-v2` embeddings
- **Backend**: FastAPI + Uvicorn
- **Frontend**: Vanilla HTML/CSS/JavaScript
- **Embeddings**: Sentence-Transformers, HuggingFace

## Project Structure

```
├── cleaning_chunks.py        # Cleans raw textbook chunks → chunks_clean.json
├── LO_cleaning.py            # Parses & cleans learning outcomes → clean_los.json
├── match_los_to_chunks.py    # Semantically matches LOs to textbook chunks → lo_with_chunks.json
├── chunks_clean.json         # Cleaned textbook chunks
├── clean_los.json            # Cleaned learning outcomes
├── requirements.txt          # Python dependencies
└── workflow/
    ├── main.py               # FastAPI server with session management
    ├── graph.py              # LangGraph multi-agent workflow (9 agents)
    ├── state.py              # AssessmentState definition
    ├── vectorstore.py        # FAISS vector store for LO semantic search
    ├── lo_with_chunks.json   # LOs mapped to their top-3 textbook chunks
    ├── chunks_clean.json     # Cleaned chunks (used at runtime)
    └── static/
        ├── index.html        # Chat UI
        ├── scripts.js        # Client-side chat logic
        └── styles.css        # Styling
```

## Agents

| Agent | Purpose |
|---|---|
| **Manager** | Entry point — classifies user intent and routes to the correct agent |
| **Greeting** | Responds to greetings and explains system capabilities |
| **Topic Extractor** | Extracts learning topics from free-text teacher input |
| **LO Retriever** | Runs semantic search per topic and shows paginated LO results |
| **LO Pager** | Shows next page of LOs when the teacher requests more |
| **LO Browser** | Displays all LOs grouped by Domain → Subdomain |
| **Chunk Retriever** | Fetches linked textbook chunks for selected LO IDs |
| **Exclusion** | Removes teacher-excluded LOs, then triggers assessment generation |
| **Assessment Generator** | Generates MCQs, short/long questions, and answer key from chunks & LOs |
| **Rejection** | Handles rejection — refines content via refined search or resets |
| **Regenerate Assessment** | Improves a previous assessment based on teacher feedback |

## Setup

### Prerequisites

- Python 3.10+
- A Groq API key

### Installation

1. **Clone the repository** and navigate into the project directory.

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file in the project root:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Usage

### Data Pipeline (run once)

If starting from raw data, run the preprocessing scripts in order:

```bash
python cleaning_chunks.py
python LO_cleaning.py
python match_los_to_chunks.py
```

This produces `chunks_clean.json`, `clean_los.json`, and `lo_with_chunks.json`.

### Running the Application

```bash
cd workflow
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser to access the chat interface.

### Conversation Flow

1. **Greet** the assistant to learn about its capabilities.
2. **Describe a topic** (e.g., "I need questions about photosynthesis").
3. **Browse LOs** — paginate through matching Learning Outcomes.
4. **Select LO IDs** to include in the assessment.
5. **Preview** linked textbook content.
6. **Generate** the assessment (MCQs + short/long answer + answer key).
7. **Refine** — exclude LOs, reject and re-search, or regenerate with feedback.

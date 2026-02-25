# Legal Research RAG Chatbot

A production-quality **Retrieval-Augmented Generation (RAG)** chatbot for India's new criminal laws — the **Bharatiya Nyaya Sanhita (BNS)** and **Bharatiya Nagarik Suraksha Sanhita (BNSS)** — with automatic IPC/CrPC cross-references. Built for the Senpiper engineering assignment.

---

## Demo

Ask natural questions like:

| Query | System understands |
|---|---|
| "What replaced IPC Section 302?" | Maps IPC 302 → BNS 103 via query expansion, retrieves murder law |
| "What is the punishment for rape?" | Retrieves BNS Section 64 with sentencing details |
| "How to file an FIR?" | Maps to BNSS Section 173, explains the procedure |
| "What is anticipatory bail?" | Retrieves BNSS Section 482 |
| "Is murder cognizable and non-bailable?" | Pulls from First Schedule (Table I) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  User Query (natural language)                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │   Streamlit  │  (app.py — chat UI)
                    │     UI       │
                    └──────┬──────┘
                           │  HTTP POST /query
                    ┌──────▼──────┐
                    │   FastAPI    │  (src/api/server.py)
                    │   Server     │
                    └──────┬──────┘
                           │
              ┌────────────▼─────────────┐
              │        RAG Pipeline       │  (src/rag/pipeline.py)
              │                          │
              │  1. Query Expansion       │  IPC/CrPC → BNS/BNSS hint
              │  2. Embed expanded query  │  all-MiniLM-L6-v2 (local)
              │  3. ChromaDB retrieval    │  Top-20 candidates
              │  4. Cross-encoder rerank  │  ms-marco-MiniLM-L-6-v2
              │  5. LLM generation        │  Gemma-3-27b-it (Google AI)
              └────────────┬─────────────┘
                           │
              ┌────────────▼─────────────┐
              │  ChromaDB (local)         │  ~1,500 chunks
              │  data/chroma_db/          │  persistent on disk
              └──────────────────────────┘
```

---

## Chunking Strategy

### Why section-aware chunking?

BNS and BNSS are structured statutes: each **Section** is the minimum meaningful legal unit. Splitting across sections would mix unrelated offences into one chunk, destroying retrieval precision. Splitting within a section would fragment the legal rule and lose context (e.g., punishment separated from the offence definition).

**Approach (`src/ingestion/parse_pdf.py`):**

1. **Section detection**: Regex `(?<!\d)(\d{1,3})\.(?:\s*\([a-z0-9]+\))*\s*[A-Z]` fires at the start of each numbered section. Each section becomes one chunk, regardless of length.

2. **Chapter headers**: Stripped from content but stored as metadata (`chapter_title`) to provide context without polluting the embedding.

3. **Table I (First Schedule)**: The BNSS First Schedule lists every BNS offence with cognizability, bail status, and trial court. These are parsed **row-by-row** into separate chunks with structured fields (`bns_section`, `offence`, `punishment`, `cognizable`, `bailable`, `court`). This allows targeted retrieval for bail/cognizability questions.

4. **PDF artifact repair (`_fix_tok`)**: pdfplumber returns words with no spaces (e.g., `"Whoevercommits"`). A wordninja-based splitter is applied token-by-token to restore word boundaries, with Unicode punctuation (curly quotes U+201C/U+201D, em-dashes U+2014) handled explicitly.

**Result:** 1,500+ chunks — 358 BNS sections, ~532 BNSS sections, ~438 Table I rows.

---

## Retrieval Pipeline

### Stage 1 — Query Expansion

`expand_query()` in `src/rag/pipeline.py` applies a **rule-based expansion map** before embedding:

- **IPC → BNS mappings**: "IPC Section 302" → appends "BNS Section 103 murder" so the vector search finds the right chunk even though the BNS text never says "this replaces IPC 302".
- **CrPC → BNSS mappings**: "CrPC 437 bail" → "BNSS Section 480 bail bailable offence"
- **Legal jargon**: "FIR" → "First Information Report", "zero FIR" → BNSS 173 hint, "challan" → charge sheet, "chowki" → police station
- **Abbreviations**: BNS → "Bharatiya Nyaya Sanhita", BNSS → "Bharatiya Nagarik Suraksha Sanhita"

The original query is **preserved and prepended** — the expansion is appended so ChromaDB sees the hint while the LLM still receives grammatically correct text.

### Stage 2 — Bi-encoder Retrieval

**Model**: `all-MiniLM-L6-v2` (sentence-transformers, runs locally — no API calls)

The expanded query is embedded and ChromaDB returns **top-20** candidates by cosine similarity.

### Stage 3 — Cross-encoder Re-ranking

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally)

The cross-encoder scores each of the 20 candidates against the **expanded query** using full attention (not just cosine distance). The top-5 are passed to the LLM. This step is critical for IPC→BNS queries: the expanded query contains the BNS section number, so the cross-encoder correctly surfaces BNS 103 over irrelevant BNSS procedural sections.

### Stage 4 — LLM Generation

**Model**: `gemma-3-27b-it` via Google AI API (free tier, 14,400 req/day)

The system prompt includes:
- Role definition (Indian law assistant, BNS + BNSS only, no other jurisdictions)
- Strict citation format: `[BNS Section X]` or `[BNSS Section Y]`
- **Authoritative IPC→BNS and CrPC→BNSS correspondence table** — prevents the LLM from hallucinating section numbers for "what replaced X?" questions

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| PDF Parsing | `pdfplumber` | Handles multi-column tables, preserves text order |
| Chunking | Custom section-aware (`parse_pdf.py`) | Statute structure demands section-level granularity |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Fast local inference, strong legal text performance |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 2-stage retrieval dramatically improves precision |
| Vector DB | ChromaDB (local, persistent) | No server setup, data stays on disk, simple API |
| LLM | Gemma-3-27b-it (Google AI API) | Open-weight, free tier, 27B parameters, strong reasoning |
| API server | FastAPI + Uvicorn | Decouples UI from pipeline, makes testing straightforward |
| UI | Streamlit | Rapid iteration, built-in chat components |

**Why open-weight LLM (Gemma) over GPT-4?** The assignment specifically values open-weight models. Gemma-3-27b-it offers near-GPT-4 quality via Google AI's free API and can be self-hosted if needed.

**Why local embeddings/re-ranking?** Zero latency after first download, no API cost, no data leaves the machine.

**Why BNSS in addition to BNS?** The assignment asked for BNS only, but users naturally ask procedural questions ("how to file FIR", "what is bail") that are answered in BNSS, not BNS. Including BNSS makes the chatbot actually useful for legal research.

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- A free Google AI Studio API key — get one at https://aistudio.google.com

### 1. Clone the repository
```bash
git clone https://github.com/99VICKY99/Legal-Research-RAG-Chatbot.git
cd Legal-Research-RAG-Chatbot
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
> First run downloads two local models (~90 MB total): `all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-6-v2`. Subsequent runs use the local cache.

### 4. Set up API key
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

### 5. Download source PDFs
```bash
python src/ingestion/download_data.py
```
Downloads BNS and BNSS official gazette PDFs into `data/raw/`.

### 6. Parse PDFs and build the vector index (run once)
```bash
# Parse PDFs → data/processed/chunks.json
python src/ingestion/parse_pdf.py

# Embed chunks → data/chroma_db/
python src/embeddings/embed_chunks.py
```

### 7. Start the API server
```bash
uvicorn src.api.server:app --port 8000
```

### 8. Launch the UI (in a second terminal)
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Running Tests

### Unit tests — query expansion (no dependencies needed)
```bash
pytest tests/test_query_expansion.py -v
```
64 tests covering all IPC→BNS, CrPC→BNSS expansions, FIR variants, case-insensitivity, ordering, and no-expansion cases.

### Data quality tests — chunks.json validation
```bash
pytest tests/test_chunks.py -v
```
27 tests verifying section counts (358 BNS, ~532 BNSS, ~438 Table I rows), key section content, and no PDF concatenation artifacts.

### Retrieval quality tests — ChromaDB (requires indexed data)
```bash
pytest tests/test_retrieval.py -v
```
26 tests confirming key sections appear in top-10 results and query expansion improves retrieval.

### All offline tests together
```bash
pytest tests/test_query_expansion.py tests/test_chunks.py -v
```
**91 tests, 91 passing.**

---

## End-to-End Evaluation

Requires the API server to be running (`uvicorn src.api.server:app --port 8000`):

```bash
python tests/eval_e2e.py
```

30 test cases covering:
- Basic punishment questions (BNS S.64, 80, 103, 303, 309, 310, 318, 356)
- Procedure questions (FIR, arrest, anticipatory bail, charge sheet)
- Cognizability/bail status questions
- IPC → BNS replacement queries (6 cases: IPC 302, 307, 376, 420, 498A, 304B)
- CrPC → BNSS replacement queries (CrPC 154, 437, 438)
- Edge cases (bare "IPC 302", all-caps, zero FIR, group murder)

**Options:**
```bash
python tests/eval_e2e.py --fast          # health check only (no LLM calls)
python tests/eval_e2e.py --model gemini-2.5-flash-lite  # override model
python tests/eval_e2e.py --delay 3       # 3s between calls (rate limit safety)
```

---

## Project Structure

```
├── app.py                          # Streamlit chat UI
├── data/
│   ├── raw/                        # Source PDFs (BNS, BNSS gazette)
│   ├── processed/
│   │   └── chunks.json             # Parsed chunks (~1,500 entries)
│   └── chroma_db/                  # Persistent ChromaDB vector store
├── src/
│   ├── api/
│   │   └── server.py               # FastAPI server (POST /query, GET /health)
│   ├── embeddings/
│   │   └── embed_chunks.py         # Embeds chunks.json → ChromaDB
│   ├── ingestion/
│   │   ├── parse_pdf.py            # Section-aware PDF parser
│   │   └── download_data.py        # Downloads source PDFs
│   ├── llm/
│   │   └── gemini_client.py        # Google AI API wrapper + system prompt
│   └── rag/
│       └── pipeline.py             # Full RAG pipeline (expand→retrieve→rerank→generate)
├── tests/
│   ├── conftest.py                 # pytest path setup
│   ├── test_query_expansion.py     # 64 unit tests for expand_query()
│   ├── test_chunks.py              # 27 data quality tests for chunks.json
│   ├── test_retrieval.py           # 26 ChromaDB retrieval quality tests
│   └── eval_e2e.py                 # 30 end-to-end eval cases (live API)
├── .env.example                    # API key template
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

**1. Two-stage retrieval (bi-encoder + cross-encoder)**

Bi-encoder retrieval is fast but approximate — cosine similarity in a 384-dim space misses many relevant chunks. The cross-encoder re-ranker reads each candidate in full context and produces a relevance score orders of magnitude more accurate, at the cost of running 20 forward passes. For a legal chatbot where correctness matters more than sub-100ms latency, this tradeoff is correct.

**2. Query expansion over query rewriting**

Query rewriting (asking an LLM to rephrase) adds a full LLM round-trip latency and can hallucinate. Rule-based expansion is deterministic, zero-latency, and covers all the IPC→BNS/CrPC→BNSS mappings that users commonly ask about. The expand→append design ensures the LLM always receives the original grammatically correct question.

**3. IPC→BNS correspondence table in the system prompt**

BNS law text never says "this replaces IPC Section X" — Section 358 is a generic omnibus repeal clause. RAG alone cannot answer "what replaced IPC 302?" without the table. Rather than allowing the LLM to use free knowledge (which led to hallucination of wrong section numbers in testing), the authoritative mapping is embedded directly in the system prompt as ground truth.

**4. Row-per-offence for Table I (First Schedule)**

The First Schedule lists 438 offences in a dense table. Embedding the whole table as one chunk makes bail/cognizability retrieval impossible — the embedding averages across 438 offences. Parsing each row as a separate chunk with structured metadata allows `WHERE source_pdf = 'BNSS'` filtering and precise retrieval for "is X a bailable offence?" questions.

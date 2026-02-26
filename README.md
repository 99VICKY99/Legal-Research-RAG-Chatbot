---
title: LegalAI — BNS & BNSS Research Chatbot
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Legal Research RAG Chatbot

A production-quality **Retrieval-Augmented Generation (RAG)** chatbot for India's new criminal laws — the **Bharatiya Nyaya Sanhita (BNS)** and **Bharatiya Nagarik Suraksha Sanhita (BNSS)** — with automatic IPC/CrPC cross-references, bail/cognizability lookup, and BNSS Second Schedule form retrieval.

---

## Demo Queries

| Query | What the system does |
|---|---|
| `"What replaced IPC Section 302?"` | Expands IPC 302 → BNS 103, retrieves murder law |
| `"What is the punishment for rape?"` | Retrieves BNS Section 64 with full sentencing |
| `"How to file an FIR?"` | Maps FIR → BNSS Section 173, explains procedure |
| `"What is anticipatory bail?"` | Retrieves BNSS Section 482 directly |
| `"Is murder cognizable and non-bailable?"` | Pulls First Schedule Table I row for BNS 103 |
| `"section 302"` | Recognises famous IPC number → injects BNS 103 (murder) |
| `"u/s 376"` | Expands u/s → "under section", finds BNS 64 (rape) |
| `"dhara 302 kya hai"` | Hindi "dhara" pattern → BNS Section 103 murder |
| `"302 of IPC"` | Suffix-format IPC reference → BNS 103 |
| `"What is Form 7?"` | Direct metadata injection for Second Schedule forms |
| `"default bail"` | Jargon maps to BNSS 187 (60/90 day custody limit) |
| `"mob lynching"` | Jargon maps to BNS 103(2) (five or more persons) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  User Query (natural language)                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                  ┌──────▼──────┐
                  │  Streamlit   │  app.py — chat UI
                  │     UI       │
                  └──────┬──────┘
                         │  HTTP POST /query
                  ┌──────▼──────┐
                  │   FastAPI    │  src/api/server.py
                  │   Server     │
                  └──────┬──────┘
                         │
            ┌────────────▼───────────────┐
            │        RAG Pipeline         │  src/rag/pipeline.py
            │                            │
            │  1. Query Expansion        │  rule-based IPC/CrPC→BNS/BNSS hints
            │  2. Injection (if exact)   │  direct metadata lookup — bypasses embedding
            │  3. Embed expanded query   │  all-MiniLM-L6-v2 (local, no API)
            │  4. ChromaDB retrieval     │  top-20 candidates
            │  5. Cross-encoder rerank   │  ms-marco-MiniLM-L-6-v2 (local)
            │  6. LLM generation         │  Gemma-3-27b-it (Google AI free tier)
            └────────────┬───────────────┘
                         │
            ┌────────────▼───────────────┐
            │  ChromaDB (local)           │  ~1,500 chunks, persistent on disk
            │  data/chroma_db/            │  data/processed/chunks.json
            └────────────────────────────┘
```

---

## Chunking Strategy

### Why section-aware chunking?

BNS and BNSS are structured statutes: each **Section** is the minimum meaningful legal unit. Splitting across sections mixes unrelated offences into one chunk, destroying retrieval precision. Splitting within a section loses the connection between an offence definition and its punishment.

**Implementation (`src/ingestion/parse_pdf.py`):**

1. **Section detection**: Regex `(?<!\d)(\d{1,3})\.(?:\s*\([a-z0-9]+\))*\s*[A-Z]` fires at numbered section starts.
2. **Chapter headers**: Stripped from content, stored as `chapter_title` metadata — provides context without polluting the embedding.
3. **Table I (First Schedule)**: BNSS First Schedule lists every BNS offence with cognizability, bail status, and trial court. Parsed **row-by-row** with structured metadata (`bns_section`, `offence`, `punishment`, `cognizable`, `bailable`, `court`). Enables precise "is X bailable?" retrieval.
4. **Table II (First Schedule)**: Bail schedule — 3 chunks with bail amounts for non-bailable offences.
5. **Second Schedule (Forms 1–58)**: Each BNSS court form (arrest warrants, bail bonds, summons, etc.) parsed as a separate chunk with `form_number` and `form_title` metadata.
6. **PDF artifact repair**: pdfplumber returns words with no spaces (e.g. `"Whoevercommits"`). A wordninja-based splitter restores word boundaries token-by-token, with Unicode punctuation (curly quotes U+201C/U+201D, em-dashes U+2014) handled explicitly.

**Result**: 1,500+ chunks across:
- 358 BNS sections
- ~532 BNSS sections
- ~438 Table I rows (First Schedule)
- 3 Table II rows
- 58 Second Schedule forms

---

## Retrieval Pipeline

### Stage 1 — Query Expansion

`expand_query()` applies a **rule-based expansion map** before embedding. The original query is preserved; matched patterns append keyword hints as a suffix. The LLM always receives the original unaltered question.

**IPC → BNS mappings** (27 sections):

| IPC | → | BNS | Offence |
|---|---|---|---|
| 302 | → | 103 | Murder |
| 376 | → | 64 | Rape |
| 420 | → | 318 | Cheating |
| 498A | → | 85 | Cruelty by husband |
| 304B | → | 80 | Dowry death |
| 307 | → | 109 | Attempt to murder |
| 124A | → | 152 | (former) Sedition |
| 323 | → | 115 | Voluntarily causing hurt |
| 363 | → | 137 | Kidnapping |
| 375 | → | 63 | Rape (definition) |
| 447 | → | 329 | Criminal trespass |
| 503/506 | → | 351 | Criminal intimidation |
| *(+15 more)* | | | |

**CrPC → BNSS mappings** (14 sections):

| CrPC | → | BNSS | Procedure |
|---|---|---|---|
| 154 | → | 173 | FIR / information to police |
| 437/438 | → | 480/482 | Bail / anticipatory bail |
| 167 | → | 187 | Remand / custody |
| 173 | → | 193 | Charge sheet / police report |
| 482 | → | 528 | High Court inherent powers |
| 125 | → | 144 | Maintenance |
| 144 | → | 163 | Orders to prevent assembly |
| 197 | → | 218 | Sanction for prosecution |
| *(+6 more)* | | | |

**Query format variants handled** — all of these correctly resolve to the same section:

```
"IPC Section 302"          "IPC 302"          "302 IPC"
"Section 302 of IPC"       "302 of IPC"       "u/s 302"
"under section 302"        "charged under 302" "dhara 302"
"CrPC 154"                 "154 of CrPC"      "Section 154 CrPC"
"sec 75"                   "S.75"             "BNS 103" (no "Section" word)
```

**Legal jargon** (absent from statute text):

| Term | Maps to |
|---|---|
| FIR, zero FIR, e-FIR | BNSS Section 173 |
| charge sheet, challan | BNSS Section 193 |
| remand, default bail | BNSS Section 187 |
| anticipatory bail | BNSS Section 482 |
| maintenance | BNSS Section 144 |
| mob lynching | BNS Section 103(2) |
| organised crime | BNS Section 111 |
| hit-and-run | BNS Section 106 |
| sedition | BNS Section 152 |
| stalking, voyeurism, acid attack, trafficking | BNS Sections 78, 77, 124, 143 |
| community service | BNS Section 4 |
| panchnama, absconder/proclaimed offender | BNSS Sections 194, 84 |
| curfew, thana, chowki | BNSS Section 163 / police station terms |

### Stage 2 — Direct Injection (bypasses vector search for exact queries)

For queries that name a specific section or form, the pipeline **fetches the chunk directly by metadata** and guarantees it appears in the final top-5 — even if the cross-encoder would have ranked it below 5. This solves a fundamental limitation of embedding models: "BNSS Section 173" has no semantic content, so cosine similarity cannot reliably distinguish it from other sections.

Four injection paths:

| Trigger | Example | Action |
|---|---|---|
| `"form N"` | "What is Form 7?" | Fetch by `form_number = 7` |
| `"BNS/BNSS Section N"` or `"BNS N"` | "What is BNSS 482?" | Fetch by `source_pdf + section_number` |
| Bare `"section N"` (famous IPC number) | "section 302" | Look up `_FAMOUS_IPC_TO_BNS` → inject BNS 103 (murder) |
| Bare `"section N"` (unknown source) | "section 75" | Try both BNS 75 and BNSS 75 |
| Bail/cognizability + section N | "Is murder bailable?" | Also inject all Table I rows for BNS 103 |

**`_FAMOUS_IPC_TO_BNS` dict** handles the case where users type a bare famous IPC number without context — `"section 302"` almost certainly means IPC 302 (murder → BNS 103), not BNS Section 302:

```python
_FAMOUS_IPC_TO_BNS = {
    302: 103,   # murder           376: 64,    # rape
    420: 318,   # cheating         498: 85,    # 498A cruelty
    307: 109,   # attempt murder   379: 303,   # theft
    395: 310,   # dacoity          # ... 19 entries total
}
```

### Stage 3 — Bi-encoder Retrieval

**Model**: `all-MiniLM-L6-v2` (sentence-transformers, runs locally, no API)

Expanded query embedded → ChromaDB returns top-20 candidates by cosine similarity. Injected chunks are prepended to the candidate list before re-ranking.

### Stage 4 — Cross-encoder Re-ranking

**Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (runs locally)

Scores all candidates as (query, chunk_text) pairs with full attention — far more accurate than cosine similarity. The top-5 are passed to the LLM. Injected chunks are force-included if the cross-encoder ranked them below 5.

### Stage 5 — LLM Generation

**Model**: `gemma-3-27b-it` via Google AI API (open-weight, free tier, 14,400 req/day)

System prompt includes:
- Role: Indian law assistant, BNS + BNSS only, no other jurisdictions
- Citation format: `[BNS Section X]` / `[BNSS Section Y]`
- Authoritative IPC→BNS and CrPC→BNSS correspondence table to prevent section number hallucination

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| PDF Parsing | `pdfplumber` | Handles multi-column tables, preserves text order |
| Chunking | Custom section-aware (`parse_pdf.py`) | Statute structure demands section-level granularity |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) | Fast local inference, no API cost |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 2-stage retrieval dramatically improves precision |
| Vector DB | ChromaDB (local, persistent) | No server setup, data stays on disk |
| LLM | `gemma-3-27b-it` (Google AI free tier) | Open-weight, strong reasoning, free API |
| API server | FastAPI + Uvicorn | Decouples UI from pipeline, makes testing clean |
| UI | Streamlit | Rapid iteration, built-in chat components |

**Why open-weight LLM?** The assignment values open-weight models. Gemma-3-27b-it is Google's open-weight model — weights are publicly released and it can be self-hosted. The Google AI free tier is used here for convenience; replacing it with a local Ollama instance requires changing one line in `gemini_client.py`.

**Why local embeddings/reranking?** Zero latency after first download, no API cost, no data leaves the machine.

**Why BNSS in addition to BNS?** The assignment asked for BNS only. In practice, users ask procedural questions ("how to file FIR?", "what is anticipatory bail?") that are answered in BNSS, not BNS. A BNS-only chatbot would fail half of realistic legal queries.

---

## Setup Instructions

### Prerequisites

- Python 3.10+
- A free Google AI Studio API key — get one at [https://aistudio.google.com](https://aistudio.google.com)

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
# Edit .env and set: GEMINI_API_KEY=your_key_here
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

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Running Tests

All tests below require `data/processed/chunks.json` and `data/chroma_db/` to be built (Step 6 above). End-to-end tests also require the API server (Step 7).

### Offline unit tests — no server needed

```bash
# Query expansion: 64 tests for IPC/CrPC/jargon patterns
pytest tests/test_query_expansion.py -v

# Data quality: 27 tests for section counts and content
pytest tests/test_chunks.py -v

# Retrieval quality: 26 tests for ChromaDB top-k results
pytest tests/test_retrieval.py -v

# Run all offline tests together (117 tests, all passing)
pytest tests/test_query_expansion.py tests/test_chunks.py tests/test_retrieval.py -v
```

### End-to-end LLM evaluation — requires API server running

```bash
# Test all 58 BNSS Second Schedule forms (Form 1–58)
python tests/eval_forms.py
# Result: 58/58 PASS  (proof in tests/results/eval_forms_result.txt)

# Test ~60 representative queries across all chunk types and query styles
python tests/eval_all_chunks.py

# Test 67 edge cases for every query pattern variant
python tests/eval_edge_cases.py

# Full coverage: every BNS section, every BNSS section, every Table I section,
# every definition sub-chunk, plus 70 topic/keyword/IPC/CrPC queries (~1,230 total)
python tests/eval_comprehensive.py
```

**Documented result**: `tests/results/eval_forms_result.txt` — 58/58 PASS, all BNSS forms correctly retrieved and answered by the LLM.

---

## Eval Coverage Summary

| Test suite | Cases | What it proves |
|---|---|---|
| `eval_forms.py` | 58 | Every BNSS Second Schedule form (Form 1–58) answered correctly |
| `eval_all_chunks.py` | 60 | Every chunk category reachable via at least one access path |
| `eval_edge_cases.py` | 67 | All query format variants handled (see below) |
| `eval_comprehensive.py` | ~1,230 | Full sweep: every BNS section, every BNSS section, every Table I entry |

### Edge case coverage (`eval_edge_cases.py`)

| Category | Examples tested |
|---|---|
| `"BNS N"` (no "Section" word) | `"What is BNS 103?"` → murder; `"BNSS 482"` → anticipatory bail |
| Famous IPC bare number | `"section 302"` → BNS 103 murder; `"section 376"` → BNS 64 rape |
| IPC suffix format | `"302 of IPC"`, `"Section 376 of IPC"`, `"498A of IPC"` |
| New IPC mappings | IPC 323 → BNS 115; IPC 363 → BNS 137; IPC 375 → BNS 63; IPC 447 → BNS 329 |
| CrPC suffix format | `"154 of CrPC"`, `"Section 438 of CrPC"` |
| New CrPC mappings | CrPC 125 → BNSS 144; CrPC 144 → BNSS 163; CrPC 197 → BNSS 218 |
| `u/s` abbreviation | `"u/s 302"` → murder; `"u/s 376"` → rape; `"u/s 420"` → cheating |
| Hindi `dhara N` | `"dhara 302 kya hai"` → BNS 103; `"dhara 376"` → BNS 64 |
| `sec N` abbreviation | `"BNS sec 103"`, `"BNSS sec 173"`, `"BNS sec 85"` |
| Alphanumeric sections | `"section 498A"` → BNS 85 (cruelty); `\d+[A-Za-z]?` regex fix |
| Legal jargon | mob lynching, organised crime, zero FIR, maintenance, challan, panchnama, default bail, curfew, sedition, community service, hit-and-run, stalking, acid attack, trafficking, criminal conspiracy |

---

## Project Structure

```
├── app.py                          # Streamlit chat UI
├── src/
│   ├── api/
│   │   └── server.py               # FastAPI server (POST /query, GET /health)
│   ├── embeddings/
│   │   └── embed_chunks.py         # Embeds chunks.json → ChromaDB
│   ├── ingestion/
│   │   ├── parse_pdf.py            # Section-aware PDF parser + artifact repair
│   │   └── download_data.py        # Downloads source PDFs
│   ├── llm/
│   │   └── gemini_client.py        # Google AI API wrapper + system prompt
│   └── rag/
│       └── pipeline.py             # Full RAG pipeline:
│                                   #   expand_query(), direct injection,
│                                   #   ChromaDB retrieval, cross-encoder rerank
├── data/
│   ├── raw/                        # Source PDFs (BNS, BNSS gazette)
│   ├── processed/
│   │   └── chunks.json             # Parsed chunks (~1,500 entries)
│   └── chroma_db/                  # Persistent ChromaDB vector store
├── tests/
│   ├── conftest.py                 # pytest path setup
│   ├── test_query_expansion.py     # 64 unit tests for expand_query()
│   ├── test_chunks.py              # 27 data quality tests for chunks.json
│   ├── test_retrieval.py           # 26 ChromaDB retrieval quality tests
│   ├── eval_forms.py               # E2E: all 58 BNSS Second Schedule forms
│   ├── eval_all_chunks.py          # E2E: 60 representative queries (all categories)
│   ├── eval_edge_cases.py          # E2E: 67 edge-case query patterns
│   ├── eval_comprehensive.py       # E2E: full coverage (~1,230 queries)
│   └── results/
│       └── eval_forms_result.txt   # Proof: 58/58 PASS
├── .env.example                    # API key template
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

**1. Direct injection over pure vector search for section numbers**

"BNSS Section 173" is semantically meaningless to an embedding model — all BNS/BNSS section chunks look nearly identical for a bare number query. Cosine distances cluster around 0.80–0.82 for all sections regardless of which one the user wants. Direct metadata lookup (`WHERE section_number = 173`) bypasses this entirely and guarantees the right chunk is returned.

**2. Two-stage retrieval (bi-encoder + cross-encoder)**

Bi-encoder retrieval is fast but approximate. The cross-encoder re-ranker reads each (query, chunk) pair with full attention and produces a relevance score orders of magnitude more accurate. For a legal chatbot where correctness matters more than sub-100ms latency, this tradeoff is correct.

**3. Query expansion over query rewriting**

Query rewriting (asking an LLM to rephrase) adds a full LLM round-trip and can hallucinate. Rule-based expansion is deterministic, zero-latency, and covers all IPC→BNS / CrPC→BNSS mappings users commonly ask about. The append design ensures the LLM always receives the original grammatically correct question.

**4. IPC→BNS correspondence table in the system prompt**

BNS statute text never says "this replaces IPC Section X" — Section 358 is a generic omnibus repeal clause. RAG alone cannot answer "what replaced IPC 302?" without a mapping. The authoritative mapping is embedded in the system prompt as ground truth, preventing the LLM from hallucinating wrong section numbers.

**5. Row-per-offence for Table I (First Schedule)**

The First Schedule lists 438 offences in a dense table. Embedding the whole table as one chunk makes bail/cognizability retrieval impossible — the embedding averages across 438 offences and loses specificity. Parsing each row as a separate chunk with structured metadata enables precise "is murder bailable?" queries.

**6. `_FAMOUS_IPC_TO_BNS` dict for bare section numbers**

Users frequently type "section 302" without any IPC/BNS prefix. In most Indian legal contexts, "302" means IPC 302 (murder → BNS 103), not BNS 302 (which is an unrelated provision). The dict pre-injects the BNS equivalent as the primary pinned chunk, with the literal BNS/BNSS N as a secondary candidate.

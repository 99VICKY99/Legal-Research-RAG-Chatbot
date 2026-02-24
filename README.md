# Legal Research RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot for querying the **Bharatiya Nyaya Sanhita (BNS)** — India's new criminal code.

## Architecture

```
User Query
    │
    ▼
[Streamlit UI]
    │
    ▼
[RAG Pipeline]
    ├── Retrieval: Query → Embeddings → ChromaDB → Top-K Chunks
    └── Generation: Chunks + Query → Groq (Llama 3.1 70B) → Answer
```

## Tech Stack

| Component        | Technology                          |
|-----------------|-------------------------------------|
| PDF Parsing     | pdfplumber                          |
| Chunking        | Section-aware (BNS structure)       |
| Embeddings      | sentence-transformers (local)       |
| Vector Database | ChromaDB (local, persistent)        |
| LLM             | Llama 3.1 70B via Groq API          |
| UI              | Streamlit                           |

## Setup Instructions

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

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your Groq API key (free at https://console.groq.com)
```

### 5. Download BNS documents
```bash
python src/ingestion/download_data.py
```

### 6. Build the vector store (run once)
```bash
python src/pipeline/build_index.py
```

### 7. Run the app
```bash
streamlit run ui/app.py
```

## Project Structure

```
├── data/
│   └── raw/              # BNS PDF documents
├── src/
│   ├── ingestion/        # PDF loading & chunking
│   ├── embeddings/       # Embedding model
│   ├── retrieval/        # ChromaDB retrieval
│   ├── generation/       # Groq LLM integration
│   └── pipeline/         # Full RAG pipeline
├── ui/
│   └── app.py            # Streamlit chat UI
├── tests/
│   └── evals.py          # Evaluation with known Q&A pairs
├── .env.example
├── requirements.txt
└── README.md
```

## Design Decisions

> Full write-up coming after all components are built.

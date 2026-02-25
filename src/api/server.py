"""
src/api/server.py

FastAPI backend for the Legal Research RAG Chatbot.

The model and ChromaDB are loaded ONCE at server startup and kept in memory.
Streamlit connects to this server via HTTP — making the UI instantly responsive.

Start with:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.rag.pipeline import query, _get_embedder, _get_collection, _get_reranker

app = FastAPI(title="LegalAI API", version="1.0")


# ── Pre-warm on startup ────────────────────────────────────────────────────────

@app.on_event("startup")
def _startup():
    """Load embedding model, reranker, and ChromaDB into memory at server start."""
    _get_embedder()
    _get_reranker()
    _get_collection()


# ── Request / Response models ──────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:      str
    source_filter: Optional[str] = None   # "BNS", "BNSS", or None
    fetch_k:       int            = 20     # candidates fetched from ChromaDB
    keep_k:        int            = 5      # top chunks kept after re-ranking
    model_name:    Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def run_query(req: QueryRequest):
    try:
        result = query(
            question      = req.question,
            source_filter = req.source_filter,
            fetch_k       = req.fetch_k,
            keep_k        = req.keep_k,
            model_name    = req.model_name,
        )
        # Remove chunks_used (large, not needed by UI)
        result.pop("chunks_used", None)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

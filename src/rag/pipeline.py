"""
src/rag/pipeline.py

Full RAG pipeline for the Legal Research Chatbot.

Flow:
  question → query expansion → ChromaDB retrieval → LLM generation → structured result

Public API:
  result = query(question, source_filter=None, n_results=7)

  result = {
      "answer":      str,           # LLM answer with inline citations
      "citations":   list[str],     # e.g. ["BNS Section 103", "BNSS Section 173"]
      "chunks_used": list[dict],    # raw retrieved chunks (for debug/UI)
      "model_used":  str,           # which LLM was used
      "query_used":  str,           # expanded query actually sent to ChromaDB
  }
"""

import re
import sys
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.gemini_client import ask, MODEL_NAME

sys.stdout.reconfigure(encoding="utf-8")

ROOT            = Path(__file__).resolve().parents[2]
CHROMA_DIR      = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "legal_india"
EMBED_MODEL     = "all-MiniLM-L6-v2"

# ── Query expansion map ────────────────────────────────────────────────────────
# Expands common abbreviations before embedding so retrieval works correctly.
# e.g. "FIR" alone won't match "First Information Report" in the text.

_EXPANSIONS = {
    r"\bFIR\b":        "First Information Report",
    r"\bIPC\b":        "Indian Penal Code",
    r"\bCrPC\b":       "Code of Criminal Procedure",
    r"\bS\.?\s*(\d+)\b": r"Section \1",   # "S.103" or "S 103" → "Section 103"
    # Old IPC section → BNS equivalent hint
    r"\bIPC\s+[Ss]ection\s+302\b":  "BNS Section 103 murder",
    r"\bIPC\s+[Ss]ection\s+307\b":  "BNS Section 109 attempt to murder",
    r"\bIPC\s+[Ss]ection\s+376\b":  "BNS Section 64 rape",
    r"\bIPC\s+[Ss]ection\s+420\b":  "BNS Section 318 cheating",
    r"\bIPC\s+[Ss]ection\s+498A\b": "BNS Section 85 cruelty by husband",
}


def expand_query(question: str) -> str:
    """Expand abbreviations in the question for better retrieval."""
    expanded = question
    for pattern, replacement in _EXPANSIONS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


# ── Lazy-loaded singletons ─────────────────────────────────────────────────────
# Loaded once on first call, reused for all subsequent queries (fast).

@lru_cache(maxsize=1)
def _get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


# ── Citation extractor ─────────────────────────────────────────────────────────

def _extract_citations(chunks: list[dict]) -> list[str]:
    """Build clean citation strings from chunk metadata."""
    citations = []
    seen = set()
    for chunk in chunks:
        meta  = chunk.get("metadata", {})
        src   = meta.get("source_pdf", "")
        ctype = meta.get("chunk_type", "")

        if ctype == "section":
            sec   = meta.get("section_number", "")
            title = meta.get("section_title", "")
            label = f"{src} Section {sec}"
            if title:
                label += f" — {title}"

        elif ctype == "table1":
            sec   = meta.get("bns_section", "")
            label = f"{src} First Schedule Table I — Section {sec}"

        elif ctype == "table2":
            label = f"{src} First Schedule Table II"

        elif ctype == "form":
            num   = meta.get("form_number", "")
            title = meta.get("form_title", "")
            label = f"{src} Second Schedule Form {num} — {title}"

        else:
            label = f"{src} chunk {meta.get('chunk_id', '')}"

        if label not in seen:
            seen.add(label)
            citations.append(label)

    return citations


# ── Main pipeline ──────────────────────────────────────────────────────────────

def query(
    question: str,
    source_filter: str | None = None,   # "BNS", "BNSS", or None (both)
    n_results: int = 7,
) -> dict:
    """
    Run the full RAG pipeline.

    Parameters
    ----------
    question      : User's legal question.
    source_filter : Restrict retrieval to "BNS" or "BNSS". None = search both.
    n_results     : Number of chunks to retrieve from ChromaDB.

    Returns
    -------
    dict with keys:
        answer      — LLM-generated answer string
        citations   — list of citation strings
        chunks_used — list of raw chunk dicts (for UI display)
        model_used  — model name used for generation
        query_used  — expanded query sent to ChromaDB
    """
    # 1. Expand query abbreviations
    expanded = expand_query(question)

    # 2. Embed the (expanded) query
    embedder   = _get_embedder()
    collection = _get_collection()
    q_emb      = embedder.encode([expanded]).tolist()

    # 3. Build optional metadata filter
    where = {"source_pdf": source_filter} if source_filter else None

    # 4. Retrieve top-K chunks from ChromaDB
    query_kwargs = dict(
        query_embeddings = q_emb,
        n_results        = n_results,
        include          = ["documents", "metadatas", "distances"],
    )
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    chunks = [
        {
            "document": doc,
            "metadata": meta,
            "distance": dist,
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ]

    # 5. Generate answer with LLM
    answer = ask(question, chunks)

    # 6. Extract citations from retrieved chunks
    citations = _extract_citations(chunks)

    return {
        "answer":      answer,
        "citations":   citations,
        "chunks_used": chunks,
        "model_used":  MODEL_NAME,
        "query_used":  expanded,
    }


# ── Quick test ─────────────────────────────────────────────────────────────────

def _test():
    import json

    test_cases = [
        # (label, question, source_filter)
        ("BASIC",        "What is the punishment for murder under BNS?",            None),
        ("EXPANSION",    "What is the procedure for filing an FIR?",                None),
        ("IPC MAPPING",  "What replaced IPC Section 302?",                          None),
        ("BNS FILTER",   "What are the offences related to assault?",               "BNS"),
        ("BNSS FILTER",  "What is the procedure when a person is arrested?",        "BNSS"),
        ("MULTI-SEC",    "What is the difference between robbery and dacoity?",     None),
        ("OUT OF SCOPE", "What is the punishment for income tax evasion?",          None),
    ]

    print("=" * 65)
    print("  Full RAG Pipeline — Test Suite")
    print("=" * 65)

    for label, question, src_filter in test_cases:
        print(f"\n[{label}]  filter={src_filter or 'ALL'}")
        print(f"Q: {question}")
        if src_filter != (src_filter or ''):
            print(f"   (expanded: {expand_query(question)})")
        print("-" * 65)

        result = query(question, source_filter=src_filter)

        # Show expanded query if it changed
        if result["query_used"] != question:
            print(f"Expanded query: {result['query_used']}")

        print(result["answer"])
        print(f"\nCitations ({len(result['citations'])}):")
        for c in result["citations"]:
            print(f"  • {c}")
        print(f"Model: {result['model_used']}")


if __name__ == "__main__":
    _test()

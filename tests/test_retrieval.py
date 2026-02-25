"""
tests/test_retrieval.py
Integration tests for ChromaDB retrieval quality — no LLM calls.

Requires: data/chroma_db/ to be populated (run embed_chunks.py first).
Run with: pytest tests/test_retrieval.py -v
"""
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Patch LLM import so pipeline can be imported without google-generativeai
with mock.patch.dict("sys.modules", {
    "google.generativeai": mock.MagicMock(),
}):
    from src.rag.pipeline import expand_query

import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_DIR      = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "legal_india"
EMBED_MODEL     = "all-MiniLM-L6-v2"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)

@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformer(EMBED_MODEL)


def _top(collection, embedder, query: str, n: int = 10) -> list[dict]:
    """Return top-n chunk metadata for a query (uses expanded query)."""
    expanded = expand_query(query)
    q_emb    = embedder.encode([expanded]).tolist()
    results  = collection.query(
        query_embeddings=q_emb,
        n_results=n,
        include=["metadatas", "distances"],
    )
    return results["metadatas"][0]

def _in_top(metas, source: str, section: str) -> bool:
    return any(
        str(m.get("source_pdf")) == source and str(m.get("section_number")) == section
        for m in metas
    )


# ══════════════════════════════════════════════════════════════════════════════
# Collection health
# ══════════════════════════════════════════════════════════════════════════════

def test_collection_populated(collection):
    assert collection.count() > 500, (
        f"Only {collection.count()} items in ChromaDB — may not be indexed"
    )

def test_no_duplicate_ids(collection):
    result = collection.get(include=[], limit=10000)
    ids = result["ids"]
    assert len(ids) == len(set(ids)), (
        f"{len(ids) - len(set(ids))} duplicate chunk IDs in ChromaDB"
    )


# ══════════════════════════════════════════════════════════════════════════════
# BNS section retrieval — expanded queries must surface correct section
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query,src,sec,description", [
    # Direct section queries (already include section number in query)
    ("BNS Section 103 murder punishment",         "BNS",  "103", "Murder"),
    ("BNS Section 64 rape punishment",            "BNS",  "64",  "Rape"),
    ("BNS Section 80 dowry death punishment",     "BNS",  "80",  "Dowry death"),
    ("BNS Section 85 cruelty husband wife",       "BNS",  "85",  "Cruelty by husband"),
    ("BNS Section 303 theft punishment",          "BNS",  "303", "Theft"),
    ("BNS Section 309 robbery punishment",        "BNS",  "309", "Robbery"),
    ("BNS Section 310 dacoity five persons",      "BNS",  "310", "Dacoity"),
    ("BNS Section 318 cheating punishment",       "BNS",  "318", "Cheating"),
    ("BNS Section 61 criminal conspiracy",        "BNS",  "61",  "Criminal conspiracy"),
    ("BNS Section 74 assault criminal force",     "BNS",  "74",  "Assault on woman"),
    ("BNS Section 108 abetment suicide",          "BNS",  "108", "Abetment of suicide"),
    ("BNS Section 109 attempt to murder",         "BNS",  "109", "Attempt to murder"),
    ("BNS Section 152 sovereignty India",         "BNS",  "152", "Endangering sovereignty"),
    ("BNS Section 356 defamation punishment",     "BNS",  "356", "Defamation"),
])
def test_bns_section_in_top10(collection, embedder, query, src, sec, description):
    metas = _top(collection, embedder, query, n=10)
    assert _in_top(metas, src, sec), (
        f"[{description}] {src} Section {sec} not in top-10 for: {query!r}\n"
        f"  Got: {[(m.get('source_pdf'), m.get('section_number')) for m in metas]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# BNSS section retrieval
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query,src,sec,description", [
    ("BNSS Section 173 information cognizable offence FIR",   "BNSS", "173", "FIR"),
    ("BNSS Section 35 arrest without warrant police",         "BNSS", "35",  "Arrest w/o warrant"),
    ("BNSS Section 187 remand custody detention",             "BNSS", "187", "Remand"),
    ("BNSS Section 193 report police investigation",          "BNSS", "193", "Charge sheet"),
    ("BNSS Section 480 bail bailable offence",                "BNSS", "480", "Bail"),
    ("BNSS Section 482 anticipatory bail",                    "BNSS", "482", "Anticipatory bail"),
    ("BNSS Section 528 inherent powers High Court",           "BNSS", "528", "Inherent powers HC"),
])
def test_bnss_section_in_top10(collection, embedder, query, src, sec, description):
    metas = _top(collection, embedder, query, n=10)
    assert _in_top(metas, src, sec), (
        f"[{description}] {src} Section {sec} not in top-10 for: {query!r}\n"
        f"  Got: {[(m.get('source_pdf'), m.get('section_number')) for m in metas]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# IPC→BNS expansion improves retrieval (expanded query must beat bare query)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bare_question,src,sec", [
    ("What replaced IPC Section 302?",  "BNS",  "103"),
    ("What replaced IPC Section 376?",  "BNS",  "64"),
    ("What replaced IPC Section 420?",  "BNS",  "318"),
    ("What replaced CrPC Section 154?", "BNSS", "173"),
    ("What replaced CrPC Section 438?", "BNSS", "482"),
])
def test_expansion_surfaces_correct_section(collection, embedder, bare_question, src, sec):
    """After query expansion the correct BNS/BNSS section must appear in top 10."""
    metas = _top(collection, embedder, bare_question, n=10)
    assert _in_top(metas, src, sec), (
        f"{src} Section {sec} not retrieved for: {bare_question!r}\n"
        f"  Expanded: {expand_query(bare_question)!r}\n"
        f"  Got: {[(m.get('source_pdf'), m.get('section_number')) for m in metas]}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# Source filter
# ══════════════════════════════════════════════════════════════════════════════

def test_source_filter_bns_only(collection, embedder):
    """When source_filter='BNS', no BNSS chunks should be returned."""
    q_emb = embedder.encode(["punishment for murder"]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=10,
        include=["metadatas"],
        where={"source_pdf": "BNS"},
    )
    for m in results["metadatas"][0]:
        assert m.get("source_pdf") == "BNS", (
            f"BNSS chunk appeared despite source_filter=BNS: {m}"
        )

def test_source_filter_bnss_only(collection, embedder):
    """When source_filter='BNSS', no BNS chunks should be returned."""
    q_emb = embedder.encode(["procedure for FIR"]).tolist()
    results = collection.query(
        query_embeddings=q_emb,
        n_results=10,
        include=["metadatas"],
        where={"source_pdf": "BNSS"},
    )
    for m in results["metadatas"][0]:
        assert m.get("source_pdf") == "BNSS", (
            f"BNS chunk appeared despite source_filter=BNSS: {m}"
        )

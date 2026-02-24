"""
src/llm/gemini_client.py

Google AI (Gemini / Gemma) client for the Legal Research RAG Chatbot.

Available models (set GEMINI_MODEL in .env):
  gemini-2.5-flash-lite  →  20 RPD  | 10 RPM | best answer quality
  gemma-3-27b-it         →  14,400 RPD | 30 RPM | best for high usage

Default: gemma-3-27b-it (higher daily limit)
"""

import os
import sys
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

# ── Load API key ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY not found in .env")

genai.configure(api_key=GEMINI_API_KEY)

# ── Model selection ────────────────────────────────────────────────────────────
# Set GEMINI_MODEL in .env to switch models:
#   GEMINI_MODEL=gemini-2.5-flash-lite   → 20 RPD,    10 RPM, best quality
#   GEMINI_MODEL=gemma-3-27b-it          → 14,400 RPD, 30 RPM, high usage
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemma-3-27b-it")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a precise legal research assistant specialising in Indian criminal law.
You are given retrieved excerpts from the Bharatiya Nyaya Sanhita (BNS) and the
Bharatiya Nagarik Suraksha Sanhita (BNSS) — India's new criminal codes that replaced
the Indian Penal Code (IPC) and the Code of Criminal Procedure (CrPC) in 2024.

Rules you must follow:
1. Answer ONLY from the provided context. Do not use outside knowledge.
2. Always cite the source: state the Act (BNS/BNSS), Section number, and Section title.
3. If the answer is not in the context, say "The provided sections do not contain
   sufficient information to answer this question."
4. Be concise and precise. Use plain English — avoid unnecessary legal jargon.
5. When quoting punishment, also mention whether the offence is cognizable/non-cognizable
   and bailable/non-bailable if that information is in the context.
"""


# ── Core function ──────────────────────────────────────────────────────────────

def ask(question: str, chunks: list[dict]) -> str:
    """
    Parameters
    ----------
    question : str
        The user's legal question.
    chunks : list[dict]
        Retrieved chunks from ChromaDB. Each dict must have:
          - 'document'  : the embed text (section content)
          - 'metadata'  : dict with source_pdf, section_number, section_title, etc.

    Returns
    -------
    str
        Answer with citations.
    """
    # Build context block from retrieved chunks
    context_lines = []
    for i, chunk in enumerate(chunks, 1):
        meta  = chunk.get("metadata", {})
        src   = meta.get("source_pdf", "")
        sec   = meta.get("section_number", meta.get("bns_section", ""))
        title = meta.get("section_title", "")
        doc   = chunk.get("document", "")

        header = f"[{i}] {src}"
        if sec:
            header += f" Section {sec}"
        if title:
            header += f" — {title}"
        context_lines.append(f"{header}\n{doc}")

    context = "\n\n---\n\n".join(context_lines)

    # System prompt is folded into user message (works for all models)
    prompt = f"""{SYSTEM_PROMPT}

CONTEXT (retrieved legal sections):
{context}

QUESTION: {question}

Answer based strictly on the context above. Cite section numbers."""

    model    = genai.GenerativeModel(model_name=MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text


# ── Quick test ─────────────────────────────────────────────────────────────────

def _test():
    """
    Standalone test: retrieves chunks from ChromaDB and asks a question.
    Run with: python src/llm/gemini_client.py
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    CHROMA_DIR      = ROOT / "data" / "chroma_db"
    COLLECTION_NAME = "legal_india"
    EMBED_MODEL     = "all-MiniLM-L6-v2"

    print("=" * 55)
    print(f"  LLM Q&A Test  |  model: {MODEL_NAME}")
    print("=" * 55)

    embedder   = SentenceTransformer(EMBED_MODEL)
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)

    questions = [
        "What is the punishment for murder under BNS?",
        "What is the procedure for filing an FIR under BNSS?",
        "Is theft a bailable offence?",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print("-" * 50)

        q_emb   = embedder.encode([q]).tolist()
        results = collection.query(
            query_embeddings=q_emb,
            n_results=5,
            include=["documents", "metadatas", "distances"],
        )

        chunks = [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(
                results["documents"][0],
                results["metadatas"][0],
            )
        ]

        answer = ask(q, chunks)
        print(answer)
        print()


if __name__ == "__main__":
    _test()

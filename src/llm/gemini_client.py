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

IPC → BNS section correspondence (authoritative — use ONLY these mappings, never guess):
  IPC 302  → BNS 103  | IPC 304  → BNS 106  | IPC 304B → BNS 80
  IPC 306  → BNS 108  | IPC 307  → BNS 109  | IPC 308  → BNS 110
  IPC 354  → BNS 74   | IPC 376  → BNS 64   | IPC 379  → BNS 303
  IPC 392  → BNS 309  | IPC 395  → BNS 310  | IPC 406  → BNS 316
  IPC 420  → BNS 318  | IPC 427  → BNS 324  | IPC 498A → BNS 85
  IPC 499  → BNS 356  | IPC 120B → BNS 61   | IPC 124A → BNS 152
  IPC 153A → BNS 196

CrPC → BNSS section correspondence (authoritative — use ONLY these mappings):
  CrPC 41  → BNSS 35  | CrPC 154 → BNSS 173 | CrPC 156 → BNSS 175
  CrPC 161 → BNSS 180 | CrPC 164 → BNSS 183 | CrPC 167 → BNSS 187
  CrPC 173 → BNSS 193 | CrPC 320 → BNSS 359 | CrPC 437 → BNSS 480
  CrPC 438 → BNSS 482 | CrPC 482 → BNSS 528

Rules you must follow:
1. Answer from the provided context. Always cite the source: Act (BNS/BNSS),
   Section number, and Section title.
2. For "what replaced IPC/CrPC X?" questions: look up the BNS/BNSS section number
   from the correspondence table above, then describe it using the retrieved context.
   NEVER guess a section number — only use numbers from the table above.
3. If the context describes a procedure or process, synthesise a plain-English explanation
   from it — do not refuse just because there is no explicit dictionary definition.
4. Only say "The provided sections do not contain sufficient information" when the context
   is genuinely unrelated to the question.
5. Be concise and precise. Use plain English — avoid unnecessary legal jargon.
6. When quoting punishment, also mention whether the offence is cognizable/non-cognizable
   and bailable/non-bailable if that information is in the context.
"""


# ── Core function ──────────────────────────────────────────────────────────────

def ask(question: str, chunks: list[dict], model_name: str | None = None) -> str:
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

    model    = genai.GenerativeModel(model_name=model_name or MODEL_NAME)
    response = model.generate_content(prompt)
    try:
        return response.text
    except (ValueError, AttributeError):
        # Safety block or empty candidates — return a graceful fallback
        candidates = getattr(response, "candidates", [])
        reason = getattr(candidates[0], "finish_reason", "unknown") if candidates else "unknown"
        return (
            f"The model could not generate a response for this query (reason: {reason}). "
            "Please try rephrasing your question."
        )


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

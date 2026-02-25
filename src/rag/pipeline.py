"""
src/rag/pipeline.py

Full RAG pipeline for the Legal Research Chatbot.

Flow:
  question → query expansion → ChromaDB retrieval (fetch_k=20)
           → cross-encoder re-rank (keep_k=5) → LLM generation → structured result

Public API:
  result = query(question, source_filter=None, fetch_k=20, keep_k=5)

  result = {
      "answer":      str,           # LLM answer with inline citations
      "citations":   list[str],     # e.g. ["BNS Section 103", "BNSS Section 173"]
      "chunks_used": list[dict],    # re-ranked chunks passed to LLM (for debug/UI)
      "model_used":  str,           # which LLM was used
      "query_used":  str,           # expanded query actually sent to ChromaDB
  }
"""

import re
import sys
from functools import lru_cache
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

# Allow running as script from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.llm.gemini_client import ask, MODEL_NAME

sys.stdout.reconfigure(encoding="utf-8")

ROOT            = Path(__file__).resolve().parents[2]
CHROMA_DIR      = ROOT / "data" / "chroma_db"
COLLECTION_NAME = "legal_india"
EMBED_MODEL     = "all-MiniLM-L6-v2"
RERANK_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Query expansion map ────────────────────────────────────────────────────────
# Maps legal jargon / old IPC-CrPC section numbers to BNS-BNSS equivalents.
#
# Each entry: regex pattern → keyword string to APPEND to the query.
# Strategy: APPEND (not replace) so the original query is preserved and the
# LLM (which uses the un-expanded question) never sees garbled grammar.
# The expanded form is only used for ChromaDB vector search.
#
# Ordering rule: specific patterns before generic abbreviations, e.g.
#   "IPC Section 302" must appear before "\bIPC\b", otherwise \bIPC\b would
#   rewrite "IPC" to "Indian Penal Code" first and the section pattern would
#   never match.

_EXPANSIONS = {

    # ── FIR variants (before plain \bFIR\b) ───────────────────────────────────
    r"\bzero\s+FIR\b": "Section 173 BNSS information cognizable offence any police station",
    r"\be-?FIR\b":     "Section 173 BNSS electronic communication information cognizable",

    # ── IPC section → BNS section (handles "IPC 302", "IPC Section 302",
    #   "Section 302 IPC", "302 IPC") ────────────────────────────────────────
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))302\b":    "BNS Section 103 murder",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)304[Bb]\b":                                             "BNS Section 80 dowry death",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)304\b":                                                 "BNS Section 106 culpable homicide not amounting murder",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)306\b":                                                 "BNS Section 108 abetment suicide",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)307\b":                                                 "BNS Section 109 attempt to murder",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)308\b":                                                 "BNS Section 110 attempt culpable homicide",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)354\b":                                                 "BNS Section 74 assault criminal force woman",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)376\b":                                                 "BNS Section 64 rape",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)379\b":                                                 "BNS Section 303 theft",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)392\b":                                                 "BNS Section 309 robbery",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)395\b":                                                 "BNS Section 310 dacoity",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)420\b":                                                 "BNS Section 318 cheating",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)498[Aa]\b":                                             "BNS Section 85 cruelty husband wife",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)124[Aa]\b":                                             "BNS Section 152 sovereignty unity integrity India",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)153[Aa]\b":                                             "BNS Section 196 enmity groups religion",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)120[Bb]\b":                                             "BNS Section 61 criminal conspiracy",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)406\b":                                                 "BNS Section 316 criminal breach of trust",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)409\b":                                                 "BNS Section 316 criminal breach of trust public servant",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)427\b":                                                 "BNS Section 324 mischief",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)499\b":                                                 "BNS Section 356 defamation",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?)500\b":                                                 "BNS Section 356 defamation punishment",

    # ── Bare-number + context word (e.g. "420 case", "booked under 302") ──────
    r"\b(?:302\s+case|booked\s+(?:under\s+)?302|accused\s+(?:of\s+)?302)\b":  "BNS Section 103 murder",
    r"\b(?:307\s+case|booked\s+(?:under\s+)?307|accused\s+(?:of\s+)?307)\b":  "BNS Section 109 attempt to murder",
    r"\b(?:376\s+case|booked\s+(?:under\s+)?376|accused\s+(?:of\s+)?376)\b":  "BNS Section 64 rape",
    r"\b(?:420\s+case|booked\s+(?:under\s+)?420|accused\s+(?:of\s+)?420)\b":  "BNS Section 318 cheating",
    r"\b(?:498[Aa]\s+case|booked\s+(?:under\s+)?498[Aa])\b":                  "BNS Section 85 cruelty husband wife",

    # ── CrPC section → BNSS section ───────────────────────────────────────────
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?)41\b":  "BNSS Section 35 arrest without warrant police",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?)154\b": "BNSS Section 173 information cognizable offence FIR",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?)156\b": "BNSS Section 175 investigation cognizable offence",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?)161\b": "BNSS Section 180 examination witnesses police",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?)164\b": "BNSS Section 183 recording confession statement",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?)167\b": "BNSS Section 187 remand custody detention",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?)173\b": "BNSS Section 193 report police officer investigation charge sheet",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?)320\b": "BNSS Section 359 compounding offences",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?)437\b": "BNSS Section 480 bail bailable offence",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?)438\b": "BNSS Section 482 anticipatory bail",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?)482\b": "BNSS Section 528 inherent powers High Court",

    # ── Generic abbreviations (after specific section patterns) ───────────────
    r"\bFIR\b":          "First Information Report information cognizable offence police station",
    r"\bIPC\b":          "Indian Penal Code",
    r"\bCrPC\b":         "Code of Criminal Procedure",
    r"\bBNS\b":          "Bharatiya Nyaya Sanhita",
    r"\bBNSS\b":         "Bharatiya Nagarik Suraksha Sanhita",
    r"\bS\.?\s*(\d+)\b": r"Section \1",   # "S.103" → "Section 103"

    # ── Legal jargon absent from statutory text ────────────────────────────────
    r"\bcharge[\s-]?sheet\b":      "Section 193 BNSS report police officer investigation",
    r"\bchallan\b":                "Section 193 BNSS report police officer investigation",
    r"\bremand\b":                 "Section 187 BNSS custody detention investigation",
    r"\banticipatory\s+bail\b":    "Section 482 BNSS anticipatory bail direction",
    r"\bdefault\s+bail\b":         "Section 187 BNSS bail default sixty ninety days",
    r"\bsedition\b":               "Section 152 BNS sovereignty unity integrity India acts endangering",
    r"\bsnatching\b":              "Section 304 BNS snatching theft",
    r"\bcommunity\s+service\b":    "Section 4 BNS punishment community service",
    r"\borganis[e]?d\s+crime\b":   "Section 111 BNS organised crime syndicate",
    r"\borganiz[e]?d\s+crime\b":   "Section 111 BNS organised crime syndicate",
    r"\bpetty\s+organis[e]?d\b":   "Section 112 BNS petty organised crime pickpocket theft",
    r"\bterror(?:ism|ist)\b":      "Section 113 BNS terrorist act",
    r"\bmob\s+lynching\b":         "Section 103 BNS murder five or more persons",
    r"\bdowry\s+death\b":          "Section 80 BNS dowry death",
    r"\bhit[\s-]and[\s-]run\b":    "Section 106 BNS causing death rash negligent act escape",
    r"\bfalse\s+promise\s+to\s+marry\b": "Section 69 BNS sexual intercourse deceitful means promise marry",
    r"\bcriminal\s+conspiracy\b":  "Section 61 BNS criminal conspiracy",
    r"\bstalking\b":               "Section 78 BNS stalking woman",
    r"\bvoyeurism\b":              "Section 77 BNS voyeurism private act",
    r"\bacid\s+attack\b":          "Section 124 BNS acid attack grievous hurt",
    r"\btrafficking\b":            "Section 143 BNS trafficking person",
    r"\bpanchnama\b":              "Section 194 BNSS police inquest report death seizure",
    r"\b(?:absconder|proclaimed\s+offender)\b": "Section 84 BNSS proclaimed offender absconding",
    r"\bcurfew\b":                 "Section 163 BNSS order prevent assembly",
    r"\bSection\s+144\b":          "Section 163 BNSS order prevent assembly",
    r"\bthana\b":                  "police station officer in charge",
    r"\bchowki\b":                 "police station officer in charge",
    r"\bnon[\s-]cognizable\b":     "non-cognizable offence complaint Magistrate police",
}


def expand_query(question: str) -> str:
    """
    Expand the query for better retrieval by appending contextual keywords.

    The original text is preserved unchanged; matched patterns contribute
    keyword strings that are appended as a suffix. This enriches the
    embedding without distorting query semantics or grammar — the LLM
    (which uses the original un-expanded question) never sees this suffix.
    """
    extras: list[str] = []
    seen_extras: set[str] = set()

    for pattern, replacement in _EXPANSIONS.items():
        if re.search(pattern, question, flags=re.IGNORECASE):
            if r"\1" in replacement:
                # Backreference pattern: resolve each match individually
                for m in re.finditer(pattern, question, flags=re.IGNORECASE):
                    resolved = m.expand(replacement)
                    if resolved not in seen_extras:
                        seen_extras.add(resolved)
                        extras.append(resolved)
            else:
                if replacement not in seen_extras:
                    seen_extras.add(replacement)
                    extras.append(replacement)

    if not extras:
        return question
    return question + " " + " ".join(extras)


# ── Lazy-loaded singletons ─────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embedder():
    return SentenceTransformer(EMBED_MODEL)


@lru_cache(maxsize=1)
def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


@lru_cache(maxsize=1)
def _get_reranker():
    return CrossEncoder(RERANK_MODEL)


# ── Re-ranker ──────────────────────────────────────────────────────────────────

def _rerank(question: str, chunks: list[dict], top_k: int) -> list[dict]:
    """
    Score each (question, chunk_text) pair with a cross-encoder and return
    the top_k chunks sorted by relevance score (descending).

    Scoring is against the original question (not the expanded form) so the
    model judges relevance by what the user actually asked.
    """
    if not chunks:
        return []
    reranker = _get_reranker()
    pairs    = [[question, c["document"]] for c in chunks]
    scores   = reranker.predict(pairs)
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)
    chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
    return chunks[:top_k]


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
    fetch_k: int = 20,                  # broad retrieval from ChromaDB
    keep_k: int  = 5,                   # top chunks after cross-encoder re-rank
    model_name: str | None = None,      # override default model at runtime
) -> dict:
    """
    Run the full RAG pipeline.

    Parameters
    ----------
    question      : User's legal question.
    source_filter : Restrict retrieval to "BNS" or "BNSS". None = search both.
    fetch_k       : Number of candidates to fetch from ChromaDB (wide net).
    keep_k        : Number of best chunks to keep after cross-encoder re-ranking.

    Returns
    -------
    dict with keys:
        answer      — LLM-generated answer string
        citations   — list of citation strings
        chunks_used — re-ranked chunks passed to LLM (for UI display)
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

    # 4. Retrieve fetch_k candidates from ChromaDB
    query_kwargs = dict(
        query_embeddings = q_emb,
        n_results        = fetch_k,
        include          = ["documents", "metadatas", "distances"],
    )
    if where:
        query_kwargs["where"] = where

    results = collection.query(**query_kwargs)

    # Guard: empty results (e.g. overly strict source_filter, or DB issue)
    if not results.get("documents") or not results["documents"][0]:
        return {
            "answer":      "No relevant legal sections found for your query. Try rephrasing or removing the source filter.",
            "citations":   [],
            "chunks_used": [],
            "model_used":  model_name or MODEL_NAME,
            "query_used":  expanded,
        }

    raw_chunks = [
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

    # 4b. Form-number injection: if the query explicitly names a form (e.g.
    #     "form 58"), fetch that form chunk directly and prepend it so the
    #     cross-encoder always sees it — pure vector search misses form chunks
    #     because their content (template text) is semantically distant from
    #     "how to fill form N".
    form_match = re.search(r"\bform\s+(?:no\.?\s*)?(\d+)\b", question, re.IGNORECASE)
    if form_match:
        form_num = form_match.group(1)
        direct = collection.get(
            where={"form_number": int(form_num)},
            include=["documents", "metadatas"],
        )
        if direct["documents"]:
            pinned = {
                "document": direct["documents"][0],
                "metadata": direct["metadatas"][0],
                "distance": 0.0,   # treat as perfect match
            }
            # Prepend only if not already in the candidates
            existing_ids = {c["metadata"].get("chunk_id") for c in raw_chunks}
            if pinned["metadata"].get("chunk_id") not in existing_ids:
                raw_chunks.insert(0, pinned)

    # 5. Cross-encoder re-rank: use *expanded* query so IPC→BNS hints are
    #    visible to the re-ranker (e.g. "IPC 302" → "BNS Section 103 murder").
    chunks = _rerank(expanded, raw_chunks, top_k=keep_k)

    # 6. Generate answer with LLM
    active_model = model_name or MODEL_NAME
    answer = ask(question, chunks, model_name=active_model)

    # 7. Extract citations from retrieved chunks
    citations = _extract_citations(chunks)

    return {
        "answer":      answer,
        "citations":   citations,
        "chunks_used": chunks,
        "model_used":  active_model,
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

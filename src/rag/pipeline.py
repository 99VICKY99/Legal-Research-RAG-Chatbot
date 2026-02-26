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

    # ── IPC section → BNS section ─────────────────────────────────────────────
    # All formats handled: "IPC 302", "IPC Section 302", "Section 302 IPC",
    # "302 of IPC", "Section 302 of IPC", "302 IPC"
    # (lookahead `(?=.*\bIPC\b)` catches suffix patterns like "302 of IPC")
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))302\b":     "BNS Section 103 murder",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))304[Bb]\b": "BNS Section 80 dowry death",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))304\b":     "BNS Section 106 culpable homicide not amounting murder",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))306\b":     "BNS Section 108 abetment suicide",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))307\b":     "BNS Section 109 attempt to murder",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))308\b":     "BNS Section 110 attempt culpable homicide",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))323\b":     "BNS Section 115 voluntarily causing hurt",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))354\b":     "BNS Section 74 assault criminal force woman",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))363\b":     "BNS Section 137 kidnapping",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))375\b":     "BNS Section 63 rape definition consent",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))376\b":     "BNS Section 64 rape",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))379\b":     "BNS Section 303 theft",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))392\b":     "BNS Section 309 robbery",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))395\b":     "BNS Section 310 dacoity",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))406\b":     "BNS Section 316 criminal breach of trust",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))409\b":     "BNS Section 316 criminal breach of trust public servant",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))420\b":     "BNS Section 318 cheating",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))427\b":     "BNS Section 324 mischief",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))447\b":     "BNS Section 329 criminal trespass",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))448\b":     "BNS Section 330 house trespass",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))498[Aa]\b": "BNS Section 85 cruelty husband wife",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))499\b":     "BNS Section 356 defamation",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))500\b":     "BNS Section 356 defamation punishment",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))503\b":     "BNS Section 351 criminal intimidation",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))506\b":     "BNS Section 351 criminal intimidation punishment",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))124[Aa]\b": "BNS Section 152 sovereignty unity integrity India",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))153[Aa]\b": "BNS Section 196 enmity groups religion",
    r"\b(?:IPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bIPC\b))120[Bb]\b": "BNS Section 61 criminal conspiracy",

    # ── Bare-number + context word (e.g. "420 case", "booked under 302",
    #   "u/s 302", "under section 302") ─────────────────────────────────────
    r"\b(?:302\s+case|(?:booked|charged)\s+(?:under\s+)?302|accused\s+(?:of\s+)?302|u/?s\s*302|under\s+[Ss]ection\s+302)\b":    "BNS Section 103 murder",
    r"\b(?:307\s+case|(?:booked|charged)\s+(?:under\s+)?307|u/?s\s*307|under\s+[Ss]ection\s+307)\b":                            "BNS Section 109 attempt to murder",
    r"\b(?:376\s+case|(?:booked|charged)\s+(?:under\s+)?376|u/?s\s*376|under\s+[Ss]ection\s+376)\b":                            "BNS Section 64 rape",
    r"\b(?:420\s+case|(?:booked|charged)\s+(?:under\s+)?420|u/?s\s*420|under\s+[Ss]ection\s+420)\b":                            "BNS Section 318 cheating",
    r"\b(?:498[Aa]\s+case|(?:booked|charged)\s+(?:under\s+)?498[Aa]|u/?s\s*498[Aa])\b":                                         "BNS Section 85 cruelty husband wife",

    # ── CrPC section → BNSS section ───────────────────────────────────────────
    # All formats: "CrPC 154", "CrPC Section 154", "Section 154 CrPC",
    # "154 of CrPC", "Section 154 of CrPC"
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))41\b":   "BNSS Section 35 arrest without warrant police",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))125\b":  "BNSS Section 144 maintenance wife children parents",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))144\b":  "BNSS Section 163 order prevent unlawful assembly",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))154\b":  "BNSS Section 173 information cognizable offence FIR",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))156\b":  "BNSS Section 175 investigation cognizable offence",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?=[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))161\b":  "BNSS Section 180 examination witnesses police",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))164\b":  "BNSS Section 183 recording confession statement",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))167\b":  "BNSS Section 187 remand custody detention",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))173\b":  "BNSS Section 193 report police officer investigation charge sheet",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))197\b":  "BNSS Section 218 prosecution sanction public servant",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))313\b":  "BNSS Section 351 examination accused court statement",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))320\b":  "BNSS Section 359 compounding offences",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))437\b":  "BNSS Section 480 bail bailable offence",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))438\b":  "BNSS Section 482 anticipatory bail",
    r"\b(?:CrPC\s+(?:[Ss]ec(?:tion)?\s+)?|(?:[Ss]ec(?:tion)?\s+)?(?=\d)(?=.*\bCrPC\b))482\b":  "BNSS Section 528 inherent powers High Court",

    # ── Generic abbreviations (after specific section patterns) ───────────────
    r"\bFIR\b":                    "First Information Report information cognizable offence police station",
    r"\bIPC\b":                    "Indian Penal Code",
    r"\bCrPC\b":                   "Code of Criminal Procedure",
    r"\bBNS\b":                    "Bharatiya Nyaya Sanhita",
    r"\bBNSS\b":                   "Bharatiya Nagarik Suraksha Sanhita",
    r"\bS\.?\s*(\d+)\b":           r"Section \1",    # "S.103"   → "Section 103"
    r"\b[Ss]ec\.?\s+(\d+)\b":      r"Section \1",    # "sec 75"  → "Section 75"
    r"\bu\s*/\s*s\b":              "under section",  # "u/s"     → "under section"
    # Specific dhara + famous IPC number: add BNS equivalent so vector search
    # finds the right section (generic dhara → "Section N" alone is ambiguous).
    # These MUST come before the generic dhara pattern below.
    r"\b[Dd]hara\s+302\b":       "BNS Section 103 murder IPC 302",
    r"\b[Dd]hara\s+304[Bb]\b":   "BNS Section 80 dowry death IPC 304B",
    r"\b[Dd]hara\s+307\b":       "BNS Section 109 attempt murder IPC 307",
    r"\b[Dd]hara\s+376\b":       "BNS Section 64 rape IPC 376",
    r"\b[Dd]hara\s+420\b":       "BNS Section 318 cheating IPC 420",
    r"\b[Dd]hara\s+498[Aa]\b":   "BNS Section 85 cruelty husband wife IPC 498A",
    r"\b[Dd]hara\s+(\d+)\b":     r"Section \1",    # Hindi dhara 302 → Section 302 (generic)

    # ── Legal jargon absent from statutory text ────────────────────────────────
    r"\bcharge[\s-]?sheet\b":           "Section 193 BNSS report police officer investigation",
    r"\bchallan\b":                     "Section 193 BNSS report police officer investigation",
    r"\bremand\b":                      "Section 187 BNSS custody detention investigation",
    r"\banticipatory\s+bail\b":         "Section 482 BNSS anticipatory bail direction",
    r"\bdefault\s+bail\b":              "Section 187 BNSS bail default sixty ninety days",
    r"\bmaintenance\b":                 "Section 144 BNSS maintenance wife children parents",
    r"\bsedition\b":                    "Section 152 BNS sovereignty unity integrity India acts endangering",
    r"\bsnatching\b":                   "Section 304 BNS snatching theft",
    r"\bcommunity\s+service\b":         "Section 4 BNS punishment community service",
    r"\borganis[e]?d\s+crime\b":        "Section 111 BNS organised crime syndicate",
    r"\borganiz[e]?d\s+crime\b":        "Section 111 BNS organised crime syndicate",
    r"\bpetty\s+organis[e]?d\b":        "Section 112 BNS petty organised crime pickpocket theft",
    r"\bterror(?:ism|ist)\b":           "Section 113 BNS terrorist act",
    r"\bmob\s+lynching\b":              "Section 103 BNS murder five or more persons",
    r"\bdowry\s+death\b":               "Section 80 BNS dowry death",
    r"\bhit[\s-]and[\s-]run\b":         "Section 106 BNS causing death rash negligent act escape",
    r"\bfalse\s+promise\s+to\s+marry\b":"Section 69 BNS sexual intercourse deceitful means promise marry",
    r"\bcriminal\s+conspiracy\b":       "Section 61 BNS criminal conspiracy",
    r"\bstalking\b":                    "Section 78 BNS stalking woman",
    r"\bvoyeurism\b":                   "Section 77 BNS voyeurism private act",
    r"\bacid\s+attack\b":               "Section 124 BNS acid attack grievous hurt",
    r"\btrafficking\b":                 "Section 143 BNS trafficking person",
    r"\bpanchnama\b":                   "Section 194 BNSS police inquest report death seizure",
    r"\b(?:absconder|proclaimed\s+offender)\b": "Section 84 BNSS proclaimed offender absconding",
    r"\bcurfew\b":                      "Section 163 BNSS order prevent assembly",
    r"\bSection\s+144\b":               "Section 163 BNSS order prevent assembly",
    r"\bthana\b":                       "police station officer in charge",
    r"\bchowki\b":                      "police station officer in charge",
    r"\bnon[\s-]cognizable\b":          "non-cognizable offence complaint Magistrate police",
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


# ── Famous IPC→BNS mappings for bare "section N" queries ──────────────────────
# When a user types "section 302" with NO source prefix they almost certainly
# mean IPC 302 (murder → BNS 103), not BNS 302 (theft-related provision).
# This dict maps the most commonly cited IPC numbers to their BNS equivalents
# so the bare_match injection pins the right chunk.
_FAMOUS_IPC_TO_BNS: dict[int, int] = {
    302: 103,   # murder
    304: 106,   # culpable homicide not amounting to murder
    306: 108,   # abetment of suicide
    307: 109,   # attempt to murder
    323: 115,   # voluntarily causing hurt
    354: 74,    # assault/criminal force on woman
    363: 137,   # kidnapping
    375: 63,    # rape (definition)
    376: 64,    # rape (punishment)
    379: 303,   # theft
    392: 309,   # robbery
    395: 310,   # dacoity
    406: 316,   # criminal breach of trust
    420: 318,   # cheating
    447: 329,   # criminal trespass
    498: 85,    # 498A cruelty by husband/relatives (int part of "498A")
    499: 356,   # defamation
    503: 351,   # criminal intimidation
    506: 351,   # criminal intimidation (punishment)
}


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

    # 4b/4c. Direct-fetch injection for explicit form or section references.
    #
    #  Problem: vector search + cross-encoder can miss the right chunk when:
    #    - Form content is generic template text with no semantic signal
    #    - Section titles are short PDF fragments (e.g. "Punishment") that
    #      don't uniquely identify the section among 20 similar candidates
    #
    #  Solution: detect "form N" or "BNS/BNSS Section N" in the query, fetch
    #  the chunk directly by metadata, and GUARANTEE it appears in the final
    #  top-k by force-inserting after re-ranking if the cross-encoder cut it.

    pinned_chunk = None

    form_match = re.search(r"\bform\s+(?:no\.?\s*)?(\d+)\b", question, re.IGNORECASE)
    if form_match:
        direct = collection.get(
            where={"form_number": int(form_match.group(1))},
            include=["documents", "metadatas"],
        )
        if direct["documents"]:
            pinned_chunk = {
                "document": direct["documents"][0],
                "metadata": direct["metadatas"][0],
                "distance": 0.0,
            }

    # Track all chunks to pin (can be multiple for Table I sub-rows)
    pinned_chunks: list[dict] = []

    if not pinned_chunk:
        # Match "BNS Section N" / "BNSS Section N" / "BNS sec N" / "BNS 103" (explicit source)
        # [A-Za-z]? handles alphanumeric sections like "498A" (captures letter too)
        sec_match = re.search(
            r"\b(BNS|BNSS)\s+(?:(?:[Ss]ection|[Ss]ec\.?)\s+)?(\d{1,3}[A-Za-z]?)\b", question, re.IGNORECASE
        )
        # Fallback: bare "section N" / "sec N" with no source prefix
        # [A-Za-z]? handles "section 498A" style alphanumeric section numbers
        bare_match = (
            None if sec_match else
            re.search(r"\b(?:[Ss]ection|[Ss]ec\.?)\s+(\d+[A-Za-z]?)\b", question)
        )

        if sec_match or bare_match:
            if sec_match:
                src_pdfs = [sec_match.group(1).upper()]
                # Strip any trailing letter (e.g. "498A" → 498) for int lookup
                sec_num  = int(re.match(r'\d+', sec_match.group(2)).group())
            else:
                # No source specified — try both BNS and BNSS, inject all found
                src_pdfs = ["BNS", "BNSS"]
                # Strip any trailing letter (e.g. "498A" → 498) for int lookup
                sec_num  = int(re.match(r'\d+', bare_match.group(1)).group())

                # GAP 2: "section 302" most likely means IPC 302 → BNS 103 (murder),
                # not BNS 302. Pre-inject the famous BNS equivalent as primary pinned
                # chunk so the right section is guaranteed in the final result.
                bns_equiv = _FAMOUS_IPC_TO_BNS.get(sec_num)
                if bns_equiv is not None and bns_equiv != sec_num:
                    direct_equiv = collection.get(
                        where={"$and": [
                            {"source_pdf": "BNS"},
                            {"section_number": bns_equiv},
                            {"chunk_type": "section"},
                        ]},
                        include=["documents", "metadatas"],
                    )
                    if direct_equiv["documents"]:
                        pinned_chunk = {
                            "document": direct_equiv["documents"][0],
                            "metadata": direct_equiv["metadatas"][0],
                            "distance": 0.0,
                        }

            sec_num_str = str(sec_num)

            for src_pdf in src_pdfs:
                # 4c-i. Pin the section chunk itself
                direct = collection.get(
                    where={"$and": [
                        {"source_pdf": src_pdf},
                        {"section_number": sec_num},
                        {"chunk_type": "section"},
                    ]},
                    include=["documents", "metadatas"],
                )
                if direct["documents"]:
                    candidate = {
                        "document": direct["documents"][0],
                        "metadata": direct["metadatas"][0],
                        "distance": 0.0,
                    }
                    # First found becomes the primary pinned_chunk
                    if pinned_chunk is None:
                        pinned_chunk = candidate
                    else:
                        # Additional source (bare match, both BNS+BNSS) → add as pinned
                        existing_ids = {c["metadata"].get("chunk_id") for c in raw_chunks}
                        if candidate["metadata"].get("chunk_id") not in existing_ids:
                            pinned_chunks.append(candidate)

                # 4c-ii. If query asks about bail/cognizability, also pin all
                #        Table I rows for this section (e.g. "103", "103(1)",
                #        "103(2)"). ChromaDB can't do prefix matching on
                #        bns_section strings, so fetch all table1 rows and
                #        filter in Python.
                bail_words = re.search(
                    r"\b(cognizable|bailable|bail|non-bailable|non bailable)\b",
                    question, re.IGNORECASE,
                )
                if bail_words and src_pdf == "BNS":
                    all_t1 = collection.get(
                        where={"chunk_type": "table1"},
                        include=["documents", "metadatas"],
                    )
                    existing_ids = {c["metadata"].get("chunk_id") for c in raw_chunks}
                    for doc, meta in zip(all_t1["documents"], all_t1["metadatas"]):
                        bns_sec = str(meta.get("bns_section", ""))
                        # match exact ("103") or sub-section ("103(1)")
                        if bns_sec == sec_num_str or bns_sec.startswith(sec_num_str + "("):
                            if meta.get("chunk_id") not in existing_ids:
                                pinned_chunks.append({
                                    "document": doc,
                                    "metadata": meta,
                                    "distance": 0.0,
                                })
                                existing_ids.add(meta.get("chunk_id"))

    # 4d. Table II injection: only 3 chunks, but they rank ~39 in vector
    #     search because their content is fragmented PDF table text.
    #     Always include them when any bail/cognizability query is detected.
    bail_any = re.search(
        r"\b(cognizable|bailable|bail|non-bailable|non bailable)\b",
        question, re.IGNORECASE,
    )
    if bail_any:
        t2_result = collection.get(
            where={"chunk_type": "table2"},
            include=["documents", "metadatas"],
        )
        existing_ids = {c["metadata"].get("chunk_id") for c in raw_chunks}
        for doc, meta in zip(t2_result["documents"], t2_result["metadatas"]):
            if meta.get("chunk_id") not in existing_ids:
                raw_chunks.append({"document": doc, "metadata": meta, "distance": 0.0})

    # Prepend pinned chunk(s) into candidates so cross-encoder scores them
    if pinned_chunk:
        existing_ids = {c["metadata"].get("chunk_id") for c in raw_chunks}
        if pinned_chunk["metadata"].get("chunk_id") not in existing_ids:
            raw_chunks.insert(0, pinned_chunk)
    for pc in pinned_chunks:
        existing_ids = {c["metadata"].get("chunk_id") for c in raw_chunks}
        if pc["metadata"].get("chunk_id") not in existing_ids:
            raw_chunks.append(pc)

    # 5. Cross-encoder re-rank: use *expanded* query so IPC→BNS hints are
    #    visible to the re-ranker (e.g. "IPC 302" → "BNS Section 103 murder").
    chunks = _rerank(expanded, raw_chunks, top_k=keep_k)

    # 5b. Guarantee pinned chunk is in final results — if cross-encoder ranked
    #     it below keep_k, force it in at position 0 (drop the last chunk).
    if pinned_chunk:
        pinned_id = pinned_chunk["metadata"].get("chunk_id")
        if not any(c["metadata"].get("chunk_id") == pinned_id for c in chunks):
            chunks = [pinned_chunk] + chunks[:keep_k - 1]
    # Also guarantee at least one Table I row is present when bail/cog words matched
    if pinned_chunks:
        pinned_ids = {pc["metadata"].get("chunk_id") for pc in pinned_chunks}
        if not any(c["metadata"].get("chunk_id") in pinned_ids for c in chunks):
            chunks = [pinned_chunks[0]] + chunks[:keep_k - 1]

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

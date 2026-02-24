"""
src/embeddings/embed_chunks.py

Load chunks.json → embed with sentence-transformers → persist to ChromaDB.

Embedding model : all-MiniLM-L6-v2  (384-dim, fast, good for semantic search)
Vector store    : data/chroma_db/
Collection name : legal_india
"""

import json
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parents[2]
CHUNKS_JSON = ROOT / "data" / "processed" / "chunks.json"
CHROMA_DIR  = ROOT / "data" / "chroma_db"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "legal_india"
EMBED_MODEL     = "all-MiniLM-L6-v2"
BATCH_SIZE      = 64


# ── Build the text string that gets embedded for each chunk ───────────────────

def build_embed_text(chunk: dict) -> str:
    """
    Construct the string to embed.  Richer context → better retrieval.
    """
    src = chunk.get("source_pdf", "")

    # ── BNS / BNSS section ────────────────────────────────────────────────────
    if "section_number" in chunk:
        sec   = chunk["section_number"]
        title = chunk.get("section_title") or ""
        ch    = chunk.get("chapter_title") or ""
        sp    = chunk.get("sub_part") or ""
        body  = chunk.get("content", "")

        header = f"[{src}] Chapter {chunk.get('chapter_number','')} — {ch}"
        if sp:
            header += f" | {sp}"
        header += f"\nSection {sec}"
        if title:
            header += f" — {title}"
        return f"{header}\n{body}"

    # ── First Schedule Table II rule (check BEFORE Table I — "Table II".startswith("Table I") is True)
    if chunk.get("table", "").startswith("Table II"):
        return f"[BNSS First Schedule Table II] {chunk.get('rule', '')}"

    # ── First Schedule Table I row ────────────────────────────────────────────
    if chunk.get("table", "").startswith("Table I"):
        sec = chunk.get("bns_section", "")
        off = chunk.get("offence", "")
        pun = chunk.get("punishment", "")
        cog = chunk.get("cognizable", "")
        bai = chunk.get("bailable", "")
        crt = chunk.get("court", "")
        return (
            f"[BNS First Schedule Table I] Section {sec}: {off}. "
            f"Punishment: {pun}. Cognizable: {cog}. "
            f"Bailable: {bai}. Court: {crt}."
        )

    # ── Second Schedule form ──────────────────────────────────────────────────
    if chunk.get("schedule") == "Second Schedule":
        num   = chunk.get("form_number", "")
        title = chunk.get("form_title", "")
        body  = chunk.get("content", "")
        return f"[BNSS Second Schedule Form {num}] {title}\n{body}"

    # Fallback
    return chunk.get("content", "") or json.dumps(chunk)


# ── Flatten metadata — ChromaDB requires str / int / float / bool only ────────

def build_metadata(chunk: dict) -> dict:
    """
    Extract filterable metadata fields; replace None with '' or -1.
    """
    meta = {
        "chunk_id":      chunk.get("chunk_id", -1),
        "source_pdf":    chunk.get("source_pdf", ""),
        "page":          chunk.get("page") if chunk.get("page") is not None else -1,
    }

    # Section chunks
    if "section_number" in chunk:
        meta["chunk_type"]     = "section"
        meta["section_number"] = chunk["section_number"]
        meta["chapter_number"] = chunk.get("chapter_number") or ""
        meta["chapter_title"]  = chunk.get("chapter_title")  or ""
        meta["sub_part"]       = chunk.get("sub_part")        or ""
        meta["section_title"]  = chunk.get("section_title")   or ""
        return meta

    # Table II (check BEFORE Table I — "Table II".startswith("Table I") is True)
    if chunk.get("table", "").startswith("Table II"):
        meta["chunk_type"] = "table2"
        meta["schedule"]   = "First Schedule"
        meta["table"]      = "Table II"
        return meta

    # Table I
    if chunk.get("table", "").startswith("Table I"):
        meta["chunk_type"]   = "table1"
        meta["schedule"]     = "First Schedule"
        meta["table"]        = "Table I"
        meta["bns_section"]  = chunk.get("bns_section", "")
        return meta

    # Second Schedule
    if chunk.get("schedule") == "Second Schedule":
        meta["chunk_type"]   = "form"
        meta["schedule"]     = "Second Schedule"
        meta["form_number"]  = chunk.get("form_number", -1)
        meta["form_title"]   = chunk.get("form_title", "")
        return meta

    meta["chunk_type"] = "unknown"
    return meta


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  ChromaDB Embedding Pipeline")
    print("=" * 55)

    # 1. Load chunks
    print(f"\n[1/4] Loading chunks from {CHUNKS_JSON.name} …")
    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
    print(f"      → {len(chunks)} chunks loaded")

    # 2. Load embedding model
    print(f"\n[2/4] Loading embedding model: {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    print(f"      → model loaded (dim={model.get_sentence_embedding_dimension()})")

    # 3. Connect to / create ChromaDB collection
    print(f"\n[3/4] Connecting to ChromaDB at {CHROMA_DIR} …")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop and recreate for a fresh embed (idempotent re-runs)
    existing = list(client.list_collections())   # v0.6+ returns names directly
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"      → deleted existing '{COLLECTION_NAME}' collection")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},   # cosine similarity
    )
    print(f"      → created collection '{COLLECTION_NAME}'")

    # 4. Embed and upsert in batches
    print(f"\n[4/4] Embedding and inserting {len(chunks)} chunks (batch={BATCH_SIZE}) …")

    ids       = []
    texts     = []
    metadatas = []

    for chunk in chunks:
        ids.append(str(chunk["chunk_id"]))
        texts.append(build_embed_text(chunk))
        metadatas.append(build_metadata(chunk))

    # Process in batches
    total = len(chunks)
    for start in tqdm(range(0, total, BATCH_SIZE), desc="Embedding"):
        end        = min(start + BATCH_SIZE, total)
        batch_ids  = ids[start:end]
        batch_txt  = texts[start:end]
        batch_meta = metadatas[start:end]

        embeddings = model.encode(batch_txt, show_progress_bar=False).tolist()

        collection.add(
            ids        = batch_ids,
            embeddings = embeddings,
            documents  = batch_txt,
            metadatas  = batch_meta,
        )

    print(f"\n{'='*55}")
    print(f"  Stored  : {collection.count()} vectors")
    print(f"  Location: {CHROMA_DIR}")
    print(f"{'='*55}")

    # ── Quick smoke test ──────────────────────────────────────────────────────
    print("\n── Smoke test: query 'punishment for murder' ──")
    q_emb = model.encode(["punishment for murder"]).tolist()
    results = collection.query(
        query_embeddings = q_emb,
        n_results        = 3,
        include          = ["documents", "metadatas", "distances"],
    )
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        src  = meta.get("source_pdf", "")
        sec  = meta.get("section_number", meta.get("bns_section", ""))
        dist_pct = round((1 - dist) * 100, 1)   # cosine → similarity %
        print(f"  [{src} sec {sec}] sim={dist_pct}%  {doc[:80]!r}")


if __name__ == "__main__":
    main()

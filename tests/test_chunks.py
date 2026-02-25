"""
tests/test_chunks.py
Validates the quality of data/processed/chunks.json — no server or LLM needed.

Actual chunk structure (from inspection):
  Section chunks : source_pdf, section_number (int), section_title, content, ...
  Table I rows   : source_pdf, schedule, bns_section (str), offence, punishment,
                   cognizable, bailable, court  — NO content field
  (chunk_type field is None for all chunks)

Run with: pytest tests/test_chunks.py -v
"""
import json
import re
from pathlib import Path

import pytest

ROOT        = Path(__file__).resolve().parents[1]
CHUNKS_PATH = ROOT / "data" / "processed" / "chunks.json"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def chunks():
    return json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))

@pytest.fixture(scope="module")
def bns_sections(chunks):
    """Section-body chunks from BNS (have integer section_number + content)."""
    return [c for c in chunks
            if c.get("source_pdf") == "BNS"
            and isinstance(c.get("section_number"), int)
            and c.get("content", "").strip()]

@pytest.fixture(scope="module")
def bnss_sections(chunks):
    """Section-body chunks from BNSS."""
    return [c for c in chunks
            if c.get("source_pdf") == "BNSS"
            and isinstance(c.get("section_number"), int)
            and c.get("content", "").strip()]

@pytest.fixture(scope="module")
def table1_rows(chunks):
    """BNSS First Schedule Table I rows (have bns_section field, no content)."""
    return [c for c in chunks if c.get("bns_section") is not None]


# ══════════════════════════════════════════════════════════════════════════════
# Chunk counts
# ══════════════════════════════════════════════════════════════════════════════

def test_total_chunk_count(chunks):
    assert len(chunks) > 1000, f"Only {len(chunks)} total chunks"

def test_bns_unique_section_count(bns_sections):
    """BNS must have exactly 358 unique section numbers."""
    unique = len(set(c["section_number"] for c in bns_sections))
    assert unique == 358, (
        f"Expected 358 unique BNS sections, got {unique}. "
        "Parser bug may have regressed — check _fix_tok / section detection."
    )

def test_bnss_unique_section_count(bnss_sections):
    """BNSS should have between 520 and 560 unique sections."""
    unique = len(set(c["section_number"] for c in bnss_sections))
    assert 520 <= unique <= 560, f"Expected 520-560 BNSS sections, got {unique}"

def test_table1_row_count(table1_rows):
    """Table I should have 430-445 rows."""
    assert 430 <= len(table1_rows) <= 445, (
        f"Expected 430-445 Table I rows, got {len(table1_rows)}"
    )

def test_bns_source_pdf_values(chunks):
    valid = {"BNS", "BNSS"}
    invalid = {c.get("source_pdf") for c in chunks if c.get("source_pdf") not in valid}
    assert not invalid, f"Unexpected source_pdf values: {invalid}"


# ══════════════════════════════════════════════════════════════════════════════
# Key BNS sections must exist with correct content
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("sec,keyword", [
    (103, "murder"),
    (64,  "rape"),
    (80,  "dowry"),
    (85,  "cruelty"),
    (108, "suicide"),
    (109, "murder"),       # attempt to murder
    (303, "theft"),
    (309, "robbery"),
    (310, "robbery"),   # dacoity = robbery by 5+ persons; "robbery" is in BNS 310 text
    (318, "cheat"),
    (61,  "conspiracy"),
    (74,  "assault"),
    (356, "defamation"),
])
def test_bns_key_section_content(bns_sections, sec, keyword):
    matching = [c for c in bns_sections if c.get("section_number") == sec]
    assert matching, f"BNS Section {sec} not found in chunks"
    text = " ".join(c["content"].lower() for c in matching)
    assert keyword in text, (
        f"BNS Section {sec} content doesn't contain '{keyword}'\n"
        f"  Preview: {text[:300]!r}"
    )

@pytest.mark.parametrize("sec", [173, 35, 187, 193, 480, 482, 528])
def test_bnss_key_section_present(bnss_sections, sec):
    matching = [c for c in bnss_sections if c.get("section_number") == sec]
    assert matching, f"BNSS Section {sec} not found in chunks"


# ══════════════════════════════════════════════════════════════════════════════
# Key Table I entries must be present
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("bns_sec", ["103", "64", "303", "309", "310"])
def test_table1_key_sections_present(table1_rows, bns_sec):
    """Key offences must appear in Table I (bns_section may include subsection, e.g. '103(1)')."""
    matching = [r for r in table1_rows
                if str(r.get("bns_section", "")).startswith(bns_sec)]
    assert matching, f"BNS Section {bns_sec} not found in Table I rows"

def test_table1_rows_have_required_fields(table1_rows):
    """Every Table I row must have offence, punishment, cognizable, bailable."""
    required = {"bns_section", "offence", "punishment", "cognizable", "bailable"}
    bad = [r for r in table1_rows if not required.issubset(r.keys())]
    assert not bad, f"{len(bad)} Table I rows missing required fields"


# ══════════════════════════════════════════════════════════════════════════════
# Content quality — no concatenation artifacts
# ══════════════════════════════════════════════════════════════════════════════

def test_no_long_concatenated_runs(bns_sections):
    """No BNS section should have 20+ consecutive lowercase chars (concatenation bug)."""
    bad_re = re.compile(r"[a-z]{20,}")
    problems = []
    for c in bns_sections:
        hits = bad_re.findall(c.get("content", ""))
        if hits:
            problems.append((c.get("section_number"), hits[:2]))
    assert not problems, (
        f"{len(problems)} BNS sections with concatenated text (first 5):\n"
        + "\n".join(f"  Section {s}: {h}" for s, h in problems[:5])
    )

def test_section_chunks_have_content(bns_sections, bnss_sections):
    """All section chunks must have non-empty content."""
    all_secs = bns_sections + bnss_sections
    empty = [c for c in all_secs if not c.get("content", "").strip()]
    assert not empty, f"{len(empty)} section chunks have empty content"

def test_bns_section_numbers_in_range(bns_sections):
    """All BNS section numbers must be in 1..358."""
    out = [c for c in bns_sections if not (1 <= c.get("section_number", 0) <= 358)]
    assert not out, (
        f"{len(out)} BNS sections out of range: "
        + str([c.get("section_number") for c in out[:10]])
    )

def test_all_section_chunks_have_source_pdf(chunks):
    """Every chunk must have a source_pdf field."""
    missing = [c for c in chunks if not c.get("source_pdf")]
    assert not missing, f"{len(missing)} chunks missing source_pdf"

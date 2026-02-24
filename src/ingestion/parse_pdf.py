"""
src/ingestion/parse_pdf.py

Parses BNS and BNSS PDFs into structured JSON chunks with full metadata.

Chunk types produced:
  1. BNS sections      — source_pdf, chapter, sub_part (optional), section
  2. BNSS sections     — source_pdf, chapter, sub_part (optional), section
  3. First Schedule T1 — offence classification table (BNS sections)
  4. First Schedule T2 — generic offence classification rules (other laws)
  5. Second Schedule   — 58 legal form templates
"""

import sys
import re
import json
import pdfplumber
import wordninja
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DIR  = Path("data/raw")
OUT_DIR  = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BNS_PDF  = RAW_DIR / "250883_english_01042024.pdf"
BNSS_PDF = RAW_DIR / "250884_2_english_01042024.pdf"

# ── Text utilities ─────────────────────────────────────────────────────────────

_JUNK = re.compile(
    r"_{5,}"
    r"|—{3,}"                        # em-dash separator line (e.g. "—————")
    r"|THE GAZETTE OF INDIA EXTRAORDINARY"
    r"|\[Part\s*II"
    r"|Sec\.\s*1\]"
    r"|CG-DL-E"
    r"|xxxGIDExxx"
    r"|jftLVªh"                      # Hindi gazette header
    r"|lañ Mhñ"
    r"|DIWAKAR SINGH"                # BNS signatory name
    r"|UPLOADED BY THE MANAGER"      # publisher line
    r"|CONTROLLER OF PUBLICATIONS"   # publisher line (raw)
    r"|Legislative Counsel"          # signatory title line (after fix_bns)
    r"|MGIPMRND"                     # BNS print reference
    r"|Govt\.\s*of India",           # signatory suffix — safe; not in law text
    re.IGNORECASE,
)

def scrub(line: str) -> str:
    """Strip PDF artifacts from a line; return '' if the line is junk."""
    line = line.strip()
    if not line:
        return ""
    if _JUNK.search(line):
        return ""
    if re.fullmatch(r"\d{1,3}", line):   # lone page numbers
        return ""
    return line


def fix_bns(text: str) -> str:
    """
    Re-insert spaces that were lost during BNS PDF generation.
    BNS has words concatenated: 'NavyorAirForce' → 'Navy or Air Force'
    BNSS does NOT need this fix.

    Strategy (two passes):
    1. Regex — handles lowercase→uppercase boundaries quickly
    2. wordninja — handles fully-lowercase concatenated runs (e.g. 'maybecalledthe')
    """
    # Pass 1: insert spaces at obvious boundaries
    text = re.sub(r"([a-z])([A-Z])",   r"\1 \2",  text)  # camelCase
    text = re.sub(r"([a-zA-Z])(\d)",   r"\1 \2",  text)  # letter→digit
    text = re.sub(r"(\d)([a-zA-Z])",   r"\1 \2",  text)  # digit→letter
    text = re.sub(r"\)([A-Za-z])",     r") \1",   text)  # ")Whoever"→") Whoever"
    text = re.sub(r",([a-zA-Z])",      r", \1",   text)  # "act,whoever"→"act, whoever"
    text = re.sub(r";([a-zA-Z])",      r"; \1",   text)  # "act;or"→"act; or"
    text = re.sub(r"––([A-Za-z])",     r"–– \1",  text)  # "––Whoever"→"–– Whoever"

    # Legal terms that are valid single words — wordninja must not split these
    _NO_SPLIT = {
        "sanhita", "bharatiya", "nyaya", "suraksha", "sakshya", "adhiniyam",
        "thereof", "therein", "thereto", "therewith", "therefrom",
        "wherein", "hereby", "herein", "whereas", "whereby",
        "aforesaid", "aforementioned", "notwithstanding", "imprisonment",
        "investigation", "misappropriation", "misrepresentation",
        "superintendent", "administration", "communication",
    }

    # Pass 2: wordninja on each line — lower threshold to 4 to catch short
    # pairs like "inthe", "ofthe"; handle punctuation-embedded tokens
    fixed_lines = []
    for line in text.splitlines():
        tokens = line.split()
        fixed_tokens = []
        for tok in tokens:
            fixed_tokens.extend(_fix_tok(tok, _NO_SPLIT))
        fixed_lines.append(" ".join(fixed_tokens))
    return "\n".join(fixed_lines)



# Compiled once here; used inside _fix_tok on every token.
# Non-raw strings so \u201c etc. are real Unicode escapes, not 6-char literals.
_PUNCT_SPLIT_RE = re.compile('([.,;:!?()\[\]"\'\u201c\u201d\u2018\u2019\-\u2013\u2014])')
_SINGLE_PUNCT_RE = re.compile('[.,;:!?()\[\]"\'\u201c\u201d\u2018\u2019\-\u2013\u2014]')


def _fix_tok(tok: str, no_split: set) -> list:
    """
    Split a single whitespace-delimited token if it looks concatenated.
    Returns a list of properly separated word strings.

    Examples:
      "inthe"              → ["in", "the"]
      "Sanhita,unless"     → ["Sanhita,", "unless"]   (comma stays on prev word)
      "imprisonmentforlife"→ ["imprisonment", "for", "life"]
      "Sanhita"            → ["Sanhita"]               (no_split, kept as-is)
    """
    # Split on internal punctuation so we can process each alpha sub-part.
    # Keep the punctuation delimiter and attach it back to the preceding word.
    parts = _PUNCT_SPLIT_RE.split(tok)

    out_words: list[str] = []   # accumulates final word tokens

    for part in parts:
        if not part:
            continue

        # Pure punctuation — glue onto the last word without a space
        if _SINGLE_PUNCT_RE.fullmatch(part):
            if out_words:
                out_words[-1] += part
            else:
                out_words.append(part)
            continue

        # Alpha sub-part — try wordninja if it looks concatenated
        if (len(part) >= 4
                and part.isalpha()
                and part.lower() not in no_split):

            if part.isupper() and len(part) > 8:
                # ALL-CAPS e.g. "OFOFFENCES"
                out_words.extend(
                    p.capitalize() for p in wordninja.split(part.lower())
                )
            elif len(part) > 1 and part[1:].islower():
                # Sentence-start or all-lowercase e.g. "Itshall", "inthe"
                words = wordninja.split(part.lower())
                if words and part[0].isupper():
                    words[0] = words[0].capitalize()
                out_words.extend(words)
            elif re.search(r'[a-z]{8,}', part):
                # Mixed-case with a substantial lowercase run, e.g.
                # "IIIandinthefollowingsections", "AStateprisonerorprisonerofwar"
                words = wordninja.split(part.lower())
                if words and part[0].isupper():
                    words[0] = words[0].capitalize()
                out_words.extend(words)
            else:
                out_words.append(part)
        else:
            out_words.append(part)

    return out_words if out_words else [tok]


# ── Section detection ─────────────────────────────────────────────────────────

# Matches section starts like:
#   "103."  "103.(1)"  "22.(1)"  "141.(1)(a)"  "198.(a)"
# Allows multiple chained sub-sections: (1), (a), (ii) etc.
_SEC_RE = re.compile(r"(?<!\d)(\d{1,3})\.(?:\s*\([a-z0-9]+\))*\s*[A-Z]")


def find_section(line: str, max_sec: int):
    """
    Return section number if line starts (or contains) a new section heading.
    Returns None if no valid section is found.
    """
    for m in _SEC_RE.finditer(line):
        n = int(m.group(1))
        if not (1 <= n <= max_sec):
            continue
        # Reject if the digit is part of a larger number (e.g. year 2023)
        if m.start() > 0 and line[m.start() - 1].isdigit():
            continue
        # Section number should not appear very deep in a long content line
        if m.start() > 90:
            continue
        return n
    return None


def extract_title(line: str, sec_num: int) -> str:
    """Extract the marginal note that appears before the section number."""
    m = re.search(rf"(?<!\d){sec_num}\.", line)
    if m and m.start() > 0:
        title = line[: m.start()].strip().rstrip(".,;:")
        if 2 < len(title) < 80:
            return title
    return ""


# ── BNS parser ─────────────────────────────────────────────────────────────────

def parse_bns() -> list:
    """
    Parse BNS PDF into section chunks.
    Hierarchy: Chapter → Sub-part ('Of …', optional) → Section
    Special: fixes concatenated-word spacing in BNS PDF.
    """
    chunks = []
    ch_num = ch_title = sub_part = None
    sec_num = sec_title = None
    buf = []
    start_page = None
    expect_ch_title = False

    def flush():
        nonlocal sec_num, sec_title, buf, start_page
        if sec_num and buf:
            content = " ".join(buf).strip()
            if len(content) > 20:
                chunks.append({
                    "source_pdf":     "BNS",
                    "chapter_number": ch_num,
                    "chapter_title":  ch_title,
                    "sub_part":       sub_part,
                    "section_number": sec_num,
                    "section_title":  sec_title or None,
                    "content":        content,
                    "page":           start_page,
                })
        sec_num = sec_title = None
        buf.clear()
        start_page = None

    with pdfplumber.open(BNS_PDF) as pdf:
        for pg_i, page in enumerate(pdf.pages, 1):
            raw  = page.extract_text() or ""
            text = fix_bns(raw)

            # Iterate raw and fixed lines in parallel — fix_bns preserves line count
            for orig_line, raw_line in zip(raw.splitlines(), text.splitlines()):
                line = scrub(raw_line)
                if not line:
                    continue

                # ── Chapter heading: check the ORIGINAL (unfixed) line first
                # because fix_bns may corrupt "CHAPTERVIII" → "Chapter Vi I I"
                m = re.match(r"^CHAPTER\s*([IVXLCDM]+)", orig_line.strip())
                if m:
                    flush()
                    ch_num = m.group(1)
                    ch_title = None
                    sub_part = None
                    expect_ch_title = True
                    continue

                # ── Chapter title (first non-junk line after heading) ─────────
                if expect_ch_title:
                    ch_title = line.title() if line.isupper() else line
                    expect_ch_title = False
                    continue

                # ── BNS Sub-part: "Of kidnapping, abduction…" ────────────────
                # Must be short (< 80 chars) and not contain sentence-like content
                # (no mid-line commas after the first word, no section references)
                if (re.match(r"^Of\s+[a-z]", line)
                        and len(line) < 80
                        and not re.search(r"section\s+\d|,\s+[a-z]{4}", line)):
                    flush()
                    sub_part = line
                    continue

                # ── Section start ─────────────────────────────────────────────
                # Use orig_line (raw, unfixed) for detection — fix_bns can corrupt
                # sub-section numbers: "(1)" → "( 1)", breaking _SEC_RE.
                sec = find_section(scrub(orig_line), max_sec=358)
                if sec:
                    flush()
                    sec_num   = sec
                    sec_title = extract_title(line, sec) or None
                    buf       = [line]
                    start_page = pg_i
                    continue

                # ── Continuation of current section ───────────────────────────
                if sec_num is not None:
                    buf.append(line)

    flush()
    return chunks


# ── BNSS parser ────────────────────────────────────────────────────────────────

def parse_bnss() -> list:
    """
    Parse BNSS main body (pages 1–157, sections 1–531).
    Hierarchy: Chapter → Sub-part ('A.—…', optional, in 8 chapters) → Section
    BNSS text is clean — no spacing fix needed.
    """
    chunks = []
    ch_num = ch_title = sub_part = None
    sec_num = sec_title = None
    buf = []
    start_page = None
    expect_ch_title = False

    def flush():
        nonlocal sec_num, sec_title, buf, start_page
        if sec_num and buf:
            content = " ".join(buf).strip()
            if len(content) > 20:
                chunks.append({
                    "source_pdf":     "BNSS",
                    "chapter_number": ch_num,
                    "chapter_title":  ch_title,
                    "sub_part":       sub_part,
                    "section_number": sec_num,
                    "section_title":  sec_title or None,
                    "content":        content,
                    "page":           start_page,
                })
        sec_num = sec_title = None
        buf.clear()
        start_page = None

    with pdfplumber.open(BNSS_PDF) as pdf:
        # Pages 1–157 contain sections; 158+ are schedules
        for pg_i, page in enumerate(pdf.pages[:157], 1):
            text = page.extract_text() or ""

            for raw_line in text.splitlines():
                line = scrub(raw_line)
                if not line:
                    continue

                # ── Chapter heading: "CHAPTER I" ──────────────────────────────
                # No strict end-anchor: handles "CHAPTER XXIX before ..." edge case
                m = re.match(r"^CHAPTER\s+([IVXLCDM]+)", line)
                if m:
                    flush()
                    ch_num = m.group(1)
                    ch_title = None
                    sub_part = None
                    expect_ch_title = True
                    continue

                # ── Chapter title ──────────────────────────────────────────────
                if expect_ch_title:
                    ch_title = line.title() if line.isupper() else line
                    expect_ch_title = False
                    continue

                # ── BNSS Sub-part: "A.—Summons", "B.—Warrant of arrest" ───────
                mp = re.match(r"^([A-F])[.\u2014\-]{1,2}\s*(.+)", line)
                if mp and len(line) < 110:
                    flush()
                    sub_part = f"{mp.group(1)} — {mp.group(2).strip()}"
                    continue

                # ── Section start ──────────────────────────────────────────────
                sec = find_section(line, max_sec=531)
                if sec:
                    flush()
                    sec_num   = sec
                    sec_title = extract_title(line, sec) or None
                    buf       = [line]
                    start_page = pg_i
                    continue

                # ── Continuation ───────────────────────────────────────────────
                if sec_num is not None:
                    buf.append(line)

    flush()
    return chunks


# ── First Schedule — Table I ───────────────────────────────────────────────────

# Column x-boundaries based on ACTUAL DATA start positions (not header labels).
# Headers are center-aligned and sit ~50px right of where data actually starts.
# Measured from pdfplumber word extraction on page 159:
# Col 1: Section     x < 93   (just the section number, e.g. "49", "103(1)")
# Col 2: Offence    93 ≤ x < 192
# Col 3: Punishment 192 ≤ x < 285
# Col 4: Cognizable 285 ≤ x < 369
# Col 5: Bailable   369 ≤ x < 453
# Col 6: Court      x ≥ 453
_T1_BREAKS = [93, 192, 285, 369, 453]

def _x_to_col(x: float) -> int:
    """Map a word's x-position to a column index (0-based)."""
    for i, boundary in enumerate(_T1_BREAKS):
        if x < boundary:
            return i
    return len(_T1_BREAKS)


def parse_schedule1_table1() -> list:
    """
    Parse Table I of First Schedule (pages 158–189).
    Uses word-position extraction because pdfplumber cannot detect this
    PDF's table borders.

    Each row = one BNS section's offence classification:
    section | offence | punishment | cognizable | bailable | court
    """
    chunks = []

    with pdfplumber.open(BNSS_PDF) as pdf:
        for pg_i, page in enumerate(pdf.pages[157:189], 158):
            words = page.extract_words() or []

            # Skip page header / footer words (y < 60 or y > page height-40)
            h = float(page.height)
            words = [w for w in words if 60 < float(w["top"]) < h - 40]

            # Sort top→bottom, then left→right
            words.sort(key=lambda w: (round(float(w["top"]) / 6), float(w["x0"])))

            # Group words into lines by y-position (tolerance = 1 rounded unit)
            # Each entry: (y_rounded, [word_dicts])
            lines: list[tuple] = []
            for w in words:
                y = round(float(w["top"]) / 6)
                if lines and abs(y - lines[-1][0]) <= 1:
                    lines[-1][1].append(w)
                else:
                    lines.append((y, [w]))

            # Build rows: a new row starts when col-0 (section number) appears
            current: list[list] = [[] for _ in range(6)]

            def _save_row():
                cols = [" ".join(c).strip() for c in current]
                sec = cols[0]
                if re.match(r"^\d", sec):
                    chunks.append({
                        "source_pdf":  "BNSS",
                        "schedule":    "First Schedule",
                        "table":       "Table I — Offences Under the Bharatiya Nyaya Sanhita",
                        "bns_section": sec,
                        "offence":     cols[1],
                        "punishment":  cols[2],
                        "cognizable":  cols[3],
                        "bailable":    cols[4],
                        "court":       cols[5],
                        "page":        pg_i,
                    })

            for _, line_words in lines:
                # Check if this line starts a new row (first word lands in col 0
                # and looks like a section number)
                first_word = line_words[0]["text"].strip()
                first_col  = _x_to_col(float(line_words[0]["x0"]))

                # Skip column-header row ("1 2 3 4 5 6") and table-heading rows
                if first_word in {"1", "Section", "Offence", "Punishment",
                                   "Cognizable", "Bailable", "Court", "cognizable",
                                   "bailable", "triable", "THE", "CLASSIFICATION",
                                   "EXPLANATORY", "I."}:
                    continue

                if first_col == 0 and re.match(r"^\d{1,3}(\([a-z0-9]+\))?$", first_word):
                    _save_row()
                    current = [[] for _ in range(6)]

                # Place each word into the right column bucket
                for w in line_words:
                    col = _x_to_col(float(w["x0"]))
                    if col < 6:
                        current[col].append(w["text"])

            _save_row()  # save last row

    return chunks


# ── First Schedule — Table II ──────────────────────────────────────────────────

def parse_schedule1_table2() -> list:
    """
    Parse Table II of First Schedule (page 189).
    3 generic rules for offences under laws other than BNS.
    """
    chunks = []
    with pdfplumber.open(BNSS_PDF) as pdf:
        text = pdf.pages[188].extract_text() or ""

    # Table II starts after the heading on this page
    idx = text.find("II.—CLASSIFICATION OF OFFENCES AGAINST OTHER LAWS")
    if idx == -1:
        return chunks

    block = text[idx:]
    lines = [l.strip() for l in block.splitlines() if l.strip()]

    # Header line and column labels to skip
    skip = {"1", "2", "3", "4", "1 2 3 4",
            "Offence", "Cognizable or", "non-cognizable.",
            "Bailable or", "non-bailable.", "By what court", "triable."}

    # Collect rule rows — lines starting with "If punishable"
    buf = []
    for line in lines[1:]:   # skip the table heading itself
        if line in skip or line.startswith("II."):
            continue
        if re.match(r"^If punishable", line):
            if buf:
                chunks.append(_t2_chunk(" ".join(buf)))
            buf = [line]
        elif buf:
            buf.append(line)

    if buf:
        chunks.append(_t2_chunk(" ".join(buf)))

    # Fallback: save entire Table II block as one chunk
    if not chunks:
        chunks.append(_t2_chunk(block.strip()))

    return chunks


def _t2_chunk(rule_text: str) -> dict:
    return {
        "source_pdf": "BNSS",
        "schedule":   "First Schedule",
        "table":      "Table II — Classification of Offences Against Other Laws",
        "rule":       rule_text.strip(),
        "page":       189,
    }


# ── Second Schedule — Forms ────────────────────────────────────────────────────

def parse_second_schedule() -> list:
    """
    Parse Second Schedule (pages 190–249): 58 legal form templates.
    Each form is one chunk.
    """
    chunks = []
    form_num = form_title = None
    buf = []
    start_pg = None

    def flush():
        nonlocal form_num, form_title, buf, start_pg
        if form_num and buf:
            content = " ".join(buf).strip()
            if content:
                chunks.append({
                    "source_pdf":  "BNSS",
                    "schedule":    "Second Schedule",
                    "table":       "Second Schedule — Legal Forms",
                    "form_number": form_num,
                    "form_title":  form_title,
                    "content":     content,
                    "page":        start_pg,
                })
        form_num = form_title = None
        buf.clear()
        start_pg = None

    with pdfplumber.open(BNSS_PDF) as pdf:
        for pg_i, page in enumerate(pdf.pages[189:], 190):
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                line = scrub(raw_line)
                if not line:
                    continue

                m = re.match(r"^FORM\s+No\.?\s*(\d+)", line, re.IGNORECASE)
                if m:
                    flush()
                    form_num  = int(m.group(1))
                    form_title = line
                    buf        = []
                    start_pg   = pg_i
                    continue

                if form_num is not None:
                    buf.append(line)

    flush()
    return chunks


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  BNS + BNSS PDF Parser")
    print("=" * 55)

    print("\n[1/5] Parsing BNS sections …")
    bns = parse_bns()
    print(f"      → {len(bns)} section chunks")

    print("\n[2/5] Parsing BNSS sections …")
    bnss = parse_bnss()
    print(f"      → {len(bnss)} section chunks")

    print("\n[3/5] Parsing First Schedule — Table I …")
    t1 = parse_schedule1_table1()
    print(f"      → {len(t1)} offence rows")

    print("\n[4/5] Parsing First Schedule — Table II …")
    t2 = parse_schedule1_table2()
    print(f"      → {len(t2)} rule rows")

    print("\n[5/5] Parsing Second Schedule (Forms) …")
    forms = parse_second_schedule()
    print(f"      → {len(forms)} form chunks")

    all_chunks = bns + bnss + t1 + t2 + forms

    # Stamp every chunk with a sequential 1-based chunk_id
    for i, chunk in enumerate(all_chunks, 1):
        chunk["chunk_id"] = i

    print("\n" + "=" * 55)
    print(f"  Total chunks : {len(all_chunks)}")
    print("=" * 55)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = OUT_DIR / "chunks.json"
    out.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n  Saved → {out}")

    # ── Quick sanity check ────────────────────────────────────────────────────
    print("\n── Sample chunks ────────────────────────────────────────")
    for chunk in all_chunks[:2]:
        print(json.dumps(chunk, ensure_ascii=False, indent=2)[:500])
        print()


if __name__ == "__main__":
    main()

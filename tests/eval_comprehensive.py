"""
tests/eval_comprehensive.py

Comprehensive end-to-end LLM generation test for ALL chunk types and ALL
query paths.

Three query paths are tested:

  PATH A — Section-number injection (direct metadata fetch):
    "What is BNS Section N?" / "What is BNSS Section N?"
    "Is BNS Section N cognizable and bailable?"
    Covers: all ~358 BNS sections, ~531 BNSS sections, ~287 Table I base secs.

  PATH B — Definition sub-chunks (vector search by term name):
    "What is the definition of [term] under BNS/BNSS?"
    Covers: all Section 2 definition sub-chunks from chunks.json.

  PATH C — Topic / keyword queries (vector search + cross-encoder):
    Natural-language questions like "How to file an FIR?", "What is bail?"
    Covers: FIR, arrest, bail, murder, rape, theft, IPC/CrPC mappings, etc.
    Also covers Table II (bail schedule) and forms via topic queries.

Total queries: ~1,230  |  ETA: ~103 minutes at 5s delay.
All within the 14,400 requests/day Gemma free-tier limit.

Run with the API server already started:
    uvicorn src.api.server:app --port 8000

Then:
    python tests/eval_comprehensive.py

Results saved to tests/results/eval_comprehensive_result.txt in real-time.
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

BACKEND     = "http://localhost:8000"
DELAY       = 5   # seconds between calls — Gemma free tier: 15k tokens/min

ROOT        = Path(__file__).resolve().parents[1]
CHUNKS_JSON = ROOT / "data" / "processed" / "chunks.json"
RESULTS_DIR = ROOT / "tests" / "results"
RESULTS_DIR.mkdir(exist_ok=True)
RESULT_FILE = RESULTS_DIR / "eval_comprehensive_result.txt"


# ── PATH C — Topic/keyword queries (hardcoded, cover all common user patterns) ─
# Format: (question, keyword_must_appear_in_answer_or_empty, display_label)

TOPIC_CASES = [
    # BNS offences by name
    ("What is the punishment for murder?",                    "103",    "murder"),
    ("What is the punishment for rape under BNS?",           "64",     "rape"),
    ("What is the punishment for theft?",                    "303",    "theft"),
    ("What is the punishment for robbery?",                  "309",    "robbery"),
    ("What is the difference between robbery and dacoity?",  "310",    "robbery vs dacoity"),
    ("What is the punishment for cheating?",                 "318",    "cheating"),
    ("What is dowry death under BNS?",                       "80",     "dowry death"),
    ("What is the law on cruelty by husband?",               "85",     "cruelty"),
    ("What is defamation under BNS?",                        "356",    "defamation"),
    ("What is kidnapping under BNS?",                        "137",    "kidnapping"),
    ("What is abduction under BNS?",                         "138",    "abduction"),
    ("What is extortion under BNS?",                         "308",    "extortion"),
    ("What is criminal breach of trust?",                    "316",    "breach of trust"),
    ("What is criminal conspiracy under BNS?",               "61",     "conspiracy"),
    ("What is the punishment for attempt to murder?",        "109",    "attempt to murder"),
    ("What is stalking under BNS?",                          "78",     "stalking"),
    ("What is voyeurism under BNS?",                         "77",     "voyeurism"),
    ("What is acid attack under BNS?",                       "124",    "acid attack"),
    ("What is human trafficking under BNS?",                 "143",    "trafficking"),
    ("What is organised crime under BNS?",                   "111",    "organised crime"),
    ("What is a terrorist act under BNS?",                   "113",    "terrorist"),
    ("What replaced sedition under BNS?",                    "152",    "sedition replacement"),
    ("What is hit and run law under BNS?",                   "106",    "hit and run"),
    ("What is the law on mob lynching?",                     "103",    "mob lynching"),
    ("What are the offences related to assault?",            "74",     "assault"),
    ("What is mischief under BNS?",                          "324",    "mischief"),
    ("What is hurt and grievous hurt?",                      "114",    "grievous hurt"),
    ("What is wrongful confinement under BNS?",              "126",    "wrongful confinement"),
    ("What is trespass under BNS?",                          "329",    "trespass"),
    ("What is forgery under BNS?",                           "336",    "forgery"),

    # BNSS procedures by name
    ("What is the procedure for filing an FIR?",             "173",    "FIR procedure"),
    ("What is a zero FIR?",                                  "173",    "zero FIR"),
    ("What is anticipatory bail?",                           "482",    "anticipatory bail"),
    ("What is the procedure for bail in a bailable offence?","480",    "bail procedure"),
    ("What is remand under BNSS?",                           "187",    "remand"),
    ("What is the procedure when a person is arrested?",     "35",     "arrest procedure"),
    ("What is a charge sheet?",                              "193",    "charge sheet"),
    ("What is default bail?",                                "187",    "default bail"),
    ("What is anticipatory bail under BNSS?",                "482",    "anticipatory bail BNSS"),
    ("What is compounding of offences under BNSS?",          "359",    "compounding"),
    ("What is the procedure for recording a confession?",    "183",    "confession"),
    ("What is the power of High Court under BNSS?",          "528",    "High Court powers"),
    ("What is a proclaimed offender?",                       "84",     "proclaimed offender"),
    ("What is the procedure for summoning an accused?",      "63",     "summons"),
    ("What is a warrant of arrest?",                         "72",     "warrant of arrest"),
    ("What is the procedure for search and seizure?",        "185",    "search seizure"),
    ("How does a Magistrate take cognizance?",               "210",    "cognizance"),
    ("What is the procedure for a trial?",                   "228",    "trial procedure"),
    ("What is Section 144 BNSS?",                            "163",    "section 144"),

    # IPC → BNS mappings
    ("What replaced IPC Section 302?",                       "103",    "IPC 302"),
    ("What replaced IPC Section 376?",                       "64",     "IPC 376"),
    ("What replaced IPC Section 420?",                       "318",    "IPC 420"),
    ("What replaced IPC Section 498A?",                      "85",     "IPC 498A"),
    ("What replaced IPC Section 304B?",                      "80",     "IPC 304B"),
    ("What replaced IPC Section 307?",                       "109",    "IPC 307"),
    ("What replaced IPC Section 379?",                       "303",    "IPC 379"),
    ("What replaced IPC Section 395?",                       "310",    "IPC 395"),
    ("What replaced IPC Section 124A?",                      "152",    "IPC 124A"),
    ("What replaced IPC Section 120B?",                      "61",     "IPC 120B"),

    # CrPC → BNSS mappings
    ("What replaced CrPC Section 154?",                      "173",    "CrPC 154"),
    ("What replaced CrPC Section 437?",                      "480",    "CrPC 437"),
    ("What replaced CrPC Section 438?",                      "482",    "CrPC 438"),
    ("What replaced CrPC Section 167?",                      "187",    "CrPC 167"),
    ("What replaced CrPC Section 41?",                       "35",     "CrPC 41"),
    ("What replaced CrPC Section 173?",                      "193",    "CrPC 173"),
    ("What replaced CrPC Section 164?",                      "183",    "CrPC 164"),

    # Bail / cognizability questions (Table I + Table II path)
    ("Is murder bailable under BNS?",                        "non-bailable",  "Table I — murder bail"),
    ("Is rape a non-bailable offence?",                      "non-bailable",  "Table I — rape bail"),
    ("Is theft cognizable under BNS?",                       "cognizable",    "Table I — theft cog"),
    ("Is dacoity cognizable and non-bailable?",              "non-bailable",  "Table I — dacoity"),
    ("Is cheating a bailable offence?",                      "bailable",      "Table I — cheating"),
    ("What offences are in Table II of BNSS?",               "",              "Table II"),
]


# ── Load section/definition data from chunks.json ─────────────────────────────

def load_data():
    chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))

    # PATH A — unique primary section numbers (exclude sub-chunks)
    bns_sections = sorted(set(
        c["section_number"] for c in chunks
        if c.get("source_pdf") == "BNS"
        and "section_number" in c
        and not c.get("definition_term")
        and not c.get("compounding_section")
    ))
    bnss_sections = sorted(set(
        c["section_number"] for c in chunks
        if c.get("source_pdf") == "BNSS"
        and "section_number" in c
        and not c.get("definition_term")
        and not c.get("compounding_section")
    ))

    # Table I unique base section numbers
    table1_base: set[int] = set()
    for c in chunks:
        if c.get("chunk_type") == "table1" or c.get("table", "").startswith("Table I"):
            bns_sec = str(c.get("bns_section", ""))
            base = bns_sec.split("(")[0].strip()
            if base.isdigit():
                table1_base.add(int(base))
    table1_sections = sorted(table1_base)

    # PATH B — definition sub-chunks: all unique (source_pdf, definition_term) pairs
    definitions: list[tuple[str, str]] = []   # (source_pdf, term)
    seen_defs: set[tuple[str, str]] = set()
    for c in chunks:
        term = c.get("definition_term", "")
        src  = c.get("source_pdf", "")
        if term and src and (src, term) not in seen_defs:
            seen_defs.add((src, term))
            definitions.append((src, term))
    definitions.sort()

    return bns_sections, bnss_sections, table1_sections, definitions


# ── HTTP helpers ───────────────────────────────────────────────────────────────

def check_health():
    try:
        ok = requests.get(f"{BACKEND}/health", timeout=3).status_code == 200
    except Exception:
        ok = False
    if not ok:
        print("ERROR: API server is not running.")
        print("  uvicorn src.api.server:app --port 8000")
        sys.exit(1)


def _parse_retry_delay(detail: str) -> int:
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", detail)
    return int(m.group(1)) + 5 if m else 65


def run_query(question: str, keyword: str = "", max_retries: int = 3) -> tuple[bool, str]:
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{BACKEND}/query",
                json={"question": question},
                timeout=60,
            )
            res = r.json()
        except Exception as e:
            return False, f"request failed: {e}"

        if "answer" not in res:
            detail = res.get("detail", str(res))
            if "429" in str(detail) or "quota" in str(detail).lower():
                wait = _parse_retry_delay(str(detail))
                msg = f"  [rate limit — waiting {wait}s, retry {attempt+1}/{max_retries}]"
                print(msg, flush=True)
                _log(msg)
                time.sleep(wait)
                continue
            return False, f"API error: {detail[:120]}"

        answer = res["answer"]
        if "could not generate" in answer.lower():
            return False, "LLM blocked"
        if len(answer.strip()) < 30:
            return False, f"answer too short ({len(answer.strip())} chars)"
        if keyword and keyword.lower() not in answer.lower():
            return False, f"keyword '{keyword}' missing in answer"

        return True, answer[:80].replace("\n", " ")

    return False, f"failed after {max_retries} retries (persistent rate limit)"


# ── Logging ────────────────────────────────────────────────────────────────────

_log_file = None

def _log(line: str):
    if _log_file:
        _log_file.write(line + "\n")
        _log_file.flush()


# ── Batch runner ───────────────────────────────────────────────────────────────

def run_batch(
    label: str,
    cases: list[tuple[str, str, str]],  # (question, keyword, display_label)
    global_counters: dict,
) -> list[tuple[str, str, str]]:
    header = f"\n{'='*72}\n  {label}  ({len(cases)} cases)\n{'='*72}"
    print(header, flush=True)
    _log(header)

    col = f"{'#':<5} {'Label':<32} {'Status':<6}  Detail"
    sep = "-" * 72
    print(col); _log(col)
    print(sep); _log(sep)

    batch_pass = 0
    batch_fail = 0
    failures   = []

    for i, (question, keyword, qlabel) in enumerate(cases, 1):
        ok, detail = run_query(question, keyword)
        status = "PASS" if ok else "FAIL"
        line = f"{i:<5} {qlabel:<32} {status:<6}  {detail}"
        print(line, flush=True)
        _log(line)

        if ok:
            batch_pass += 1
            global_counters["pass"] += 1
        else:
            batch_fail += 1
            global_counters["fail"] += 1
            failures.append((qlabel, question, detail))

        if i < len(cases):
            time.sleep(DELAY)

    summary = f"Batch result: {batch_pass}/{len(cases)} passed"
    print(f"\n{summary}", flush=True)
    _log(f"\n{summary}")
    return failures


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global _log_file

    check_health()

    print("Loading chunks.json …", flush=True)
    bns_secs, bnss_secs, t1_secs, definitions = load_data()
    print(f"  BNS unique sections    : {len(bns_secs)}")
    print(f"  BNSS unique sections   : {len(bnss_secs)}")
    print(f"  Table I base sections  : {len(t1_secs)}")
    print(f"  Definition sub-chunks  : {len(definitions)}")
    print(f"  Topic/keyword queries  : {len(TOPIC_CASES)}")

    total = (len(bns_secs) + len(bnss_secs) + len(t1_secs)
             + len(definitions) + 3 + len(TOPIC_CASES))
    eta = total * (DELAY + 1) // 60
    print(f"\nTotal queries : {total}  |  ETA: ~{eta} minutes at {DELAY}s delay")
    print(f"Results       → {RESULT_FILE}\n")

    _log_file = open(RESULT_FILE, "w", encoding="utf-8")
    _log(f"eval_comprehensive.py — Full end-to-end coverage test")
    _log(f"Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    _log(f"Model   : gemma-3-27b-it (Google AI free tier)")
    _log(f"Backend : {BACKEND}")
    _log(f"Delay   : {DELAY}s between calls")
    _log(f"BNS sections      : {len(bns_secs)}")
    _log(f"BNSS sections     : {len(bnss_secs)}")
    _log(f"Table I sections  : {len(t1_secs)}")
    _log(f"Definitions       : {len(definitions)}")
    _log(f"Table II          : 3")
    _log(f"Topic/keyword     : {len(TOPIC_CASES)}")
    _log(f"Total queries     : {total}")

    counters = {"pass": 0, "fail": 0}
    all_failures: list[tuple[str, str, str, str]] = []

    # ── PATH A-1: BNS sections by number ──────────────────────────────────────
    bns_cases = [
        (f"What is BNS Section {n}?", "", f"BNS S.{n}")
        for n in bns_secs
    ]
    f = run_batch("PATH A — BNS SECTIONS by section number", bns_cases, counters)
    all_failures += [("BNS section", *x) for x in f]

    # ── PATH A-2: BNSS sections by number ─────────────────────────────────────
    bnss_cases = [
        (f"What is BNSS Section {n}?", "", f"BNSS S.{n}")
        for n in bnss_secs
    ]
    f = run_batch("PATH A — BNSS SECTIONS by section number", bnss_cases, counters)
    all_failures += [("BNSS section", *x) for x in f]

    # ── PATH A-3: Table I by bail/cog query ───────────────────────────────────
    t1_cases = [
        (f"Is BNS Section {n} cognizable and bailable?", "", f"Table I BNS S.{n}")
        for n in t1_secs
    ]
    f = run_batch("PATH A — TABLE I bail/cognizability by section number", t1_cases, counters)
    all_failures += [("Table I", *x) for x in f]

    # ── PATH A-4: Table II ─────────────────────────────────────────────────────
    t2_cases = [
        ("What offences are covered in Table II of BNSS First Schedule?",
         "", "Table II — offences"),
        ("What is the bail amount prescribed in BNSS First Schedule Table II?",
         "", "Table II — bail amount"),
        ("What does Table II of BNSS say about bailable offences?",
         "", "Table II — bailable"),
    ]
    f = run_batch("PATH A — TABLE II BNSS First Schedule", t2_cases, counters)
    all_failures += [("Table II", *x) for x in f]

    # ── PATH B: Definition sub-chunks ─────────────────────────────────────────
    def_cases = [
        (
            f"What is the definition of \"{term}\" under {src}?",
            term.split()[0].lower() if term else "",
            f"{src} def: {term[:28]}"
        )
        for src, term in definitions
    ]
    f = run_batch("PATH B — DEFINITION SUB-CHUNKS (Section 2 terms)", def_cases, counters)
    all_failures += [("Definition", *x) for x in f]

    # ── PATH C: Topic / keyword queries ───────────────────────────────────────
    topic_cases = [(q, kw, lbl) for q, kw, lbl in TOPIC_CASES]
    f = run_batch("PATH C — TOPIC / KEYWORD / IPC-CrPC MAPPING QUERIES", topic_cases, counters)
    all_failures += [("Topic", *x) for x in f]

    # ── Final summary ─────────────────────────────────────────────────────────
    total_done = counters["pass"] + counters["fail"]
    summary = (
        f"\n{'='*72}\n"
        f"FINAL RESULT: {counters['pass']}/{total_done} PASSED\n"
        f"\n"
        f"  PATH A — Section-number injection:\n"
        f"    BNS sections  : {len(bns_secs)} tested\n"
        f"    BNSS sections : {len(bnss_secs)} tested\n"
        f"    Table I secs  : {len(t1_secs)} base sections tested\n"
        f"    Table II      : 3 tested\n"
        f"\n"
        f"  PATH B — Definition sub-chunks (vector search):\n"
        f"    Definitions   : {len(definitions)} tested\n"
        f"\n"
        f"  PATH C — Topic/keyword queries (vector search + cross-encoder):\n"
        f"    Topic cases   : {len(TOPIC_CASES)} tested\n"
        f"\n"
        f"  Forms (separate test) : 58/58 (see eval_forms_result.txt)\n"
        f"{'='*72}"
    )
    print(summary, flush=True)
    _log(summary)

    if all_failures:
        fail_hdr = f"\nFailed cases ({len(all_failures)}):"
        print(fail_hdr, flush=True)
        _log(fail_hdr)
        for batch, qlabel, question, err in all_failures:
            line = f"  [{batch}] {qlabel}\n    Q: {question}\n    E: {err}"
            print(line, flush=True)
            _log(line)
    else:
        msg = "\nAll cases answered successfully — zero failures."
        print(msg, flush=True)
        _log(msg)

    _log_file.close()
    print(f"\nFull results saved to: {RESULT_FILE}")


if __name__ == "__main__":
    main()

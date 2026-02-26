"""
tests/eval_all_chunks.py

End-to-end coverage test for all chunk categories and query styles.

Tests ~60 representative cases covering:
  - BNS sections  via section-number, topic-name, IPC mapping
  - BNSS sections via section-number, keyword, CrPC mapping
  - Table I rows  via bail/cognizability queries
  - Table II      via bail schedule queries

Does NOT test all 1500 chunks individually — instead verifies that each
access path (injection, vector search, IPC/CrPC expansion) works end-to-end
with the LLM generating a real answer.

Run with the API server already started:
    uvicorn src.api.server:app --port 8000

Then:
    python tests/eval_all_chunks.py
"""

import re
import sys
import time
import requests

BACKEND = "http://localhost:8000"
DELAY   = 5   # seconds between calls — Gemma free tier: 15k tokens/min

# ── Test cases ─────────────────────────────────────────────────────────────────
# Format: (category, query_style, question, keyword_expected_in_answer)
#   keyword_expected_in_answer: a word/phrase that should appear in the LLM answer
#   to confirm the right section was retrieved (case-insensitive).
#   Set to "" to only check for non-empty answer.

CASES = [

    # ── BNS sections — by explicit section number ─────────────────────────────
    ("BNS section-number", "BNS S.103", "What is BNS Section 103?",           "murder"),
    ("BNS section-number", "BNS S.64",  "What is BNS Section 64?",            "rape"),
    ("BNS section-number", "BNS S.303", "What is BNS Section 303?",           "theft"),
    ("BNS section-number", "BNS S.309", "What is BNS Section 309?",           "robbery"),
    ("BNS section-number", "BNS S.310", "What is BNS Section 310?",           "dacoity"),
    ("BNS section-number", "BNS S.80",  "What is BNS Section 80?",            "dowry"),
    ("BNS section-number", "BNS S.85",  "What is BNS Section 85?",            "cruelty"),
    ("BNS section-number", "BNS S.318", "What is BNS Section 318?",           "cheating"),
    ("BNS section-number", "BNS S.356", "What is BNS Section 356?",           "defamation"),
    ("BNS section-number", "BNS S.111", "What is BNS Section 111?",           "organised"),

    # ── BNS sections — by topic / name ───────────────────────────────────────
    ("BNS topic",          "murder",    "What is the punishment for murder?",          "103"),
    ("BNS topic",          "rape",      "What is the punishment for rape under BNS?",  "64"),
    ("BNS topic",          "theft",     "What is the punishment for theft?",           "303"),
    ("BNS topic",          "dacoity",   "What is the difference between robbery and dacoity?", "310"),
    ("BNS topic",          "dowry",     "What is dowry death under BNS?",              "80"),
    ("BNS topic",          "stalking",  "What is the law on stalking?",                "78"),
    ("BNS topic",          "assault",   "What are the offences related to assault?",   "74"),
    ("BNS topic",          "sedition",  "What replaced sedition law in BNS?",          "152"),

    # ── BNS sections — via IPC mapping (query expansion path) ─────────────────
    ("BNS IPC-map",        "IPC 302",   "What replaced IPC Section 302?",     "103"),
    ("BNS IPC-map",        "IPC 376",   "What replaced IPC Section 376?",     "64"),
    ("BNS IPC-map",        "IPC 420",   "What replaced IPC Section 420?",     "318"),
    ("BNS IPC-map",        "IPC 498A",  "What replaced IPC Section 498A?",    "85"),
    ("BNS IPC-map",        "IPC 304B",  "What replaced IPC Section 304B?",    "80"),
    ("BNS IPC-map",        "IPC 307",   "What replaced IPC Section 307?",     "109"),

    # ── BNSS sections — by explicit section number ────────────────────────────
    ("BNSS section-number","BNSS S.173","What is BNSS Section 173?",          "information"),
    ("BNSS section-number","BNSS S.482","What is BNSS Section 482?",          "anticipatory"),
    ("BNSS section-number","BNSS S.480","What is BNSS Section 480?",          "bail"),
    ("BNSS section-number","BNSS S.187","What is BNSS Section 187?",          "remand"),
    ("BNSS section-number","BNSS S.35", "What is BNSS Section 35?",           "arrest"),
    ("BNSS section-number","BNSS S.193","What is BNSS Section 193?",          "report"),
    ("BNSS section-number","BNSS S.528","What is BNSS Section 528?",          "High Court"),

    # ── BNSS sections — by keyword / procedure ───────────────────────────────
    ("BNSS keyword",       "FIR",       "What is the procedure for filing an FIR?",     "173"),
    ("BNSS keyword",       "zero FIR",  "What is a zero FIR?",                          "173"),
    ("BNSS keyword",       "bail",      "What is the procedure for bail?",              "480"),
    ("BNSS keyword",       "anticipatory bail", "What is anticipatory bail?",           "482"),
    ("BNSS keyword",       "remand",    "What is the remand procedure?",                "187"),
    ("BNSS keyword",       "arrest",    "What is the procedure when a person is arrested?", "35"),
    ("BNSS keyword",       "chargesheet","What is a charge sheet?",                     "193"),

    # ── BNSS sections — via CrPC mapping ─────────────────────────────────────
    ("BNSS CrPC-map",      "CrPC 154",  "What replaced CrPC Section 154?",    "173"),
    ("BNSS CrPC-map",      "CrPC 437",  "What replaced CrPC Section 437?",    "480"),
    ("BNSS CrPC-map",      "CrPC 438",  "What replaced CrPC Section 438?",    "482"),
    ("BNSS CrPC-map",      "CrPC 167",  "What replaced CrPC Section 167?",    "187"),
    ("BNSS CrPC-map",      "CrPC 41",   "What replaced CrPC Section 41?",     "35"),

    # ── Table I — bail / cognizability (metadata injection path) ─────────────
    ("Table I",  "murder bail",   "Is murder bailable under BNS?",                "non-bailable"),
    ("Table I",  "theft bail",    "Is theft cognizable under BNS?",               "cognizable"),
    ("Table I",  "rape bail",     "Is rape a non-bailable offence?",              "non-bailable"),
    ("Table I",  "dacoity bail",  "Is dacoity cognizable and non-bailable?",      "non-bailable"),
    ("Table I",  "cheating bail", "Is cheating a bailable offence under BNS?",   "bailable"),

    # ── Table II — bail schedule ───────────────────────────────────────────────
    ("Table II", "Table II",      "What offences are covered in Table II of BNSS First Schedule?", ""),
    ("Table II", "bail amount",   "What is the bail amount for non-bailable offences?",            ""),
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def check_health():
    try:
        ok = requests.get(f"{BACKEND}/health", timeout=3).status_code == 200
    except Exception:
        ok = False
    if not ok:
        print("ERROR: API server is not running. Start it first:")
        print("  uvicorn src.api.server:app --port 8000")
        sys.exit(1)


def _parse_retry_delay(detail: str) -> int:
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", detail)
    return int(m.group(1)) + 5 if m else 65


def run_case(question: str, keyword: str, max_retries: int = 3) -> tuple[bool, str]:
    """Returns (passed, detail). Auto-retries on 429."""
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
                print(f"    [rate limit — waiting {wait}s, retry {attempt+1}/{max_retries}]")
                time.sleep(wait)
                continue
            return False, f"API error: {detail[:100]}"

        answer = res["answer"]
        if "could not generate" in answer.lower():
            return False, f"LLM blocked"
        if len(answer.strip()) < 20:
            return False, f"answer too short"
        if keyword and keyword.lower() not in answer.lower():
            return False, f"keyword '{keyword}' missing in answer"

        return True, answer[:80].replace("\n", " ")

    return False, f"failed after {max_retries} retries (persistent rate limit)"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    check_health()

    results_by_cat: dict[str, list[bool]] = {}
    total_pass = 0
    total_fail = 0
    failures   = []

    print(f"Testing {len(CASES)} cases against {BACKEND}  (delay={DELAY}s)\n")
    print(f"{'#':<4} {'Category':<22} {'Style':<18} {'Status':<6}  Detail")
    print("-" * 90)

    for i, (category, style, question, keyword) in enumerate(CASES, 1):
        ok, detail = run_case(question, keyword)
        status = "PASS" if ok else "FAIL"
        print(f"{i:<4} {category:<22} {style:<18} {status:<6}  {detail}")

        results_by_cat.setdefault(category, []).append(ok)
        if ok:
            total_pass += 1
        else:
            total_fail += 1
            failures.append((i, category, style, question, detail))

        if i < len(CASES):
            time.sleep(DELAY)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"OVERALL: {total_pass}/{len(CASES)} passed\n")

    print(f"{'Category':<25}  {'Pass/Total'}")
    print("-" * 40)
    for cat, results in results_by_cat.items():
        p = sum(results)
        t = len(results)
        bar = "✓" * p + "✗" * (t - p)
        print(f"{cat:<25}  {p}/{t}  {bar}")

    if failures:
        print(f"\nFailed cases:")
        for idx, cat, style, q, detail in failures:
            print(f"  [{idx}] {cat} / {style}")
            print(f"       Q: {q}")
            print(f"       E: {detail}")


if __name__ == "__main__":
    main()

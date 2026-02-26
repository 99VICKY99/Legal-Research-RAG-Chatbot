"""
tests/eval_edge_cases.py

Targeted edge-case test for all query patterns fixed in the latest pipeline update.

Covers:
  GAP 1  — "BNS 103" / "BNSS 173" format (no "Section" word)
  GAP 2  — bare "section 302" → famous IPC → BNS equivalent
  GAP 3  — "302 of IPC" / "Section 376 of IPC" suffix format
  GAP 4  — new IPC mappings (323, 363, 375, 447, 448, 503, 506)
  GAP 5  — "154 of CrPC" suffix + new CrPC mappings (125, 144, 197, 313)
  GAP 6  — "u/s N" abbreviation
  GAP 7  — "dhara N" Hindi pattern
  GAP 8  — "sec N" abbreviation (bare + with source prefix)
  JARGON — legal terms: mob lynching, organised crime, zero FIR, maintenance,
            challan, panchnama, absconder, curfew, community service,
            hit-and-run, stalking, acid attack, trafficking, false promise,
            default bail, voyeurism

Run with the API server already started:
    uvicorn src.api.server:app --port 8000

Then:
    python tests/eval_edge_cases.py
"""

import re
import sys
import time
import requests

BACKEND = "http://localhost:8000"
DELAY   = 5   # seconds between calls — Gemma free tier: 15k tokens/min

# ── Test cases ─────────────────────────────────────────────────────────────────
# Format: (category, query_style, question, keyword_expected_in_answer)
# keyword is checked case-insensitively in the LLM answer.
# Set to "" to only check for a non-empty answer.

CASES = [

    # ── GAP 1: "BNS N" / "BNSS N" — no "Section" word ───────────────────────
    ("GAP1 BNS-N",  "BNS 103",  "What is BNS 103?",             "murder"),
    ("GAP1 BNS-N",  "BNS 64",   "What is BNS 64?",              "rape"),
    ("GAP1 BNS-N",  "BNS 80",   "What is BNS 80?",              "dowry"),
    ("GAP1 BNS-N",  "BNS 318",  "What is BNS 318?",             "cheating"),
    ("GAP1 BNSS-N", "BNSS 173", "What is BNSS 173?",            "information"),
    ("GAP1 BNSS-N", "BNSS 482", "What is BNSS 482?",            "anticipatory"),
    ("GAP1 BNSS-N", "BNSS 480", "What is BNSS 480?",            "bail"),
    ("GAP1 BNSS-N", "BNSS 187", "What is BNSS 187?",            "remand"),

    # ── GAP 2: bare "section N" → famous IPC → BNS equivalent ───────────────
    ("GAP2 bare-sec",  "sec 302",  "What is section 302?",         "murder"),
    ("GAP2 bare-sec",  "sec 376",  "What is section 376?",         "rape"),
    ("GAP2 bare-sec",  "sec 420",  "What is section 420?",         "cheating"),
    ("GAP2 bare-sec",  "sec 307",  "What is section 307?",         "attempt"),
    ("GAP2 bare-sec",  "sec 498A", "What is section 498A?",        "cruelty"),
    ("GAP2 bare-sec",  "sec 379",  "What is section 379?",         "theft"),
    ("GAP2 bare-sec",  "sec 395",  "What is section 395?",         "dacoity"),

    # ── GAP 3: "N of IPC" / "Section N of IPC" suffix format ─────────────────
    ("GAP3 IPC-suffix", "302 of IPC",         "What is 302 of IPC?",              "murder"),
    ("GAP3 IPC-suffix", "Sec 376 of IPC",     "What is Section 376 of IPC?",      "rape"),
    ("GAP3 IPC-suffix", "498A of IPC",        "What is 498A of IPC?",             "cruelty"),
    ("GAP3 IPC-suffix", "420 of IPC",         "What is 420 of IPC?",              "cheating"),
    ("GAP3 IPC-suffix", "Section 302 of IPC", "What is Section 302 of IPC?",      "murder"),

    # ── GAP 4: new IPC mappings ───────────────────────────────────────────────
    ("GAP4 new-IPC", "IPC 323", "What replaced IPC Section 323?",  "115"),
    ("GAP4 new-IPC", "IPC 363", "What replaced IPC Section 363?",  "kidnapping"),
    ("GAP4 new-IPC", "IPC 375", "What is IPC Section 375?",        "63"),
    ("GAP4 new-IPC", "IPC 447", "What replaced IPC Section 447?",  "trespass"),
    ("GAP4 new-IPC", "IPC 503", "What replaced IPC Section 503?",  "intimidation"),

    # ── GAP 5: "N of CrPC" suffix + new CrPC mappings ────────────────────────
    ("GAP5 CrPC-suffix", "154 of CrPC",         "What is 154 of CrPC?",           "173"),
    ("GAP5 CrPC-suffix", "Section 438 of CrPC", "What is Section 438 of CrPC?",   "anticipatory"),
    ("GAP5 CrPC-suffix", "437 of CrPC",         "What is 437 of CrPC?",           "bail"),
    ("GAP5 new-CrPC",    "CrPC 125",            "What replaced CrPC Section 125?","144"),
    ("GAP5 new-CrPC",    "CrPC 144",            "What replaced CrPC Section 144?","163"),
    ("GAP5 new-CrPC",    "CrPC 197",            "What replaced CrPC Section 197?","218"),

    # ── GAP 6: "u/s N" abbreviation ──────────────────────────────────────────
    ("GAP6 u/s", "u/s 302",  "What is u/s 302?",                   "murder"),
    ("GAP6 u/s", "u/s 376",  "What is the punishment u/s 376?",    "rape"),
    ("GAP6 u/s", "u/s 420",  "What is a u/s 420 case?",            "cheating"),

    # ── GAP 7: "dhara N" Hindi pattern ───────────────────────────────────────
    ("GAP7 dhara", "dhara 302", "dhara 302 kya hai?",              "murder"),
    ("GAP7 dhara", "dhara 376", "dhara 376 kya hai?",              "rape"),
    ("GAP7 dhara", "dhara 420", "dhara 420 kya hai?",              "cheating"),

    # ── GAP 8: "sec N" abbreviation (with and without source prefix) ─────────
    ("GAP8 sec-N", "BNSS sec 173", "What is BNSS sec 173?",        "information"),
    ("GAP8 sec-N", "BNS sec 85",   "What is BNS sec 85?",          "cruelty"),
    ("GAP8 sec-N", "BNS sec 103",  "What is BNS sec 103?",         "murder"),

    # ── JARGON: legal terms not in statutory text ─────────────────────────────
    ("JARGON", "zero FIR",            "What is a zero FIR?",                          "173"),
    ("JARGON", "mob lynching",        "What is the law against mob lynching?",         "103"),
    ("JARGON", "organised crime",     "What is organised crime under BNS?",            "111"),
    ("JARGON", "community service",   "What is community service as punishment?",      "4"),
    ("JARGON", "hit and run",         "What is the punishment for hit and run?",       "106"),
    ("JARGON", "stalking",            "What is the law on stalking?",                  "78"),
    ("JARGON", "acid attack",         "What is the provision for acid attack?",        "124"),
    ("JARGON", "trafficking",         "What is human trafficking under BNS?",          "143"),
    ("JARGON", "false promise marry", "What is false promise to marry under BNS?",     "69"),
    ("JARGON", "dowry death",         "What is dowry death under BNS?",               "80"),
    ("JARGON", "voyeurism",           "What is voyeurism under BNS?",                  "77"),
    ("JARGON", "maintenance",         "What is the law on maintenance for wife?",      "144"),
    ("JARGON", "challan",             "What is a challan in criminal law?",            "193"),
    ("JARGON", "panchnama",           "What is a panchnama?",                          "194"),
    ("JARGON", "absconder",           "What happens to a proclaimed absconder?",       "proclaimed"),
    ("JARGON", "default bail",        "What is default bail?",                         "sixty"),
    ("JARGON", "curfew",              "Can a Magistrate impose curfew?",               "163"),
    ("JARGON", "sedition replaced",   "What replaced sedition law in BNS?",           "152"),
    ("JARGON", "snatching",           "What is snatching under BNS?",                 "304"),
    ("JARGON", "criminal conspiracy", "What is criminal conspiracy under BNS?",       "61"),
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
            return False, "LLM blocked"
        if len(answer.strip()) < 20:
            return False, f"answer too short"
        if keyword and keyword.lower() not in answer.lower():
            return False, f"keyword '{keyword}' missing"

        return True, answer[:80].replace("\n", " ")

    return False, f"failed after {max_retries} retries (persistent rate limit)"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    check_health()

    results_by_cat: dict[str, list[bool]] = {}
    total_pass = 0
    total_fail = 0
    failures   = []

    print(f"Edge-case test — {len(CASES)} cases against {BACKEND}  (delay={DELAY}s)\n")
    print(f"{'#':<4} {'Category':<20} {'Style':<22} {'Status':<6}  Detail")
    print("-" * 92)

    for i, (category, style, question, keyword) in enumerate(CASES, 1):
        ok, detail = run_case(question, keyword)
        status = "PASS" if ok else "FAIL"
        print(f"{i:<4} {category:<20} {style:<22} {status:<6}  {detail}")

        results_by_cat.setdefault(category, []).append(ok)
        if ok:
            total_pass += 1
        else:
            total_fail += 1
            failures.append((i, category, style, question, detail))

        if i < len(CASES):
            time.sleep(DELAY)

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print(f"OVERALL: {total_pass}/{len(CASES)} passed\n")

    print(f"{'Category':<22}  {'Pass/Total'}")
    print("-" * 40)
    for cat, results in results_by_cat.items():
        p = sum(results)
        t = len(results)
        bar = "✓" * p + "✗" * (t - p)
        print(f"{cat:<22}  {p}/{t}  {bar}")

    if failures:
        print(f"\nFailed cases:")
        for idx, cat, style, q, detail in failures:
            print(f"  [{idx}] {cat} / {style}")
            print(f"       Q: {q}")
            print(f"       E: {detail}")
    else:
        print("\nAll edge cases passed!")


if __name__ == "__main__":
    main()

"""
tests/eval_e2e.py
End-to-end evaluation against the live API server.

Requires the API server to be running:
    uvicorn src.api.server:app --port 8000

Run with:
    python tests/eval_e2e.py                  # all cases
    python tests/eval_e2e.py --fast           # skip slow LLM cases, test API only
    python tests/eval_e2e.py --model gemini-2.5-flash-lite   # override model

Output: coloured pass/fail table + final score.
"""
import argparse
import sys
import time
from pathlib import Path

import requests

ROOT    = Path(__file__).resolve().parents[1]
API_URL = "http://localhost:8000"

# ── Eval cases ────────────────────────────────────────────────────────────────
# Format: (question, [required_keywords_in_answer], description)
# Keywords are checked case-insensitively in the full answer text.
EVAL_CASES = [

    # ── Basic punishment questions ───────────────────────────────────────────
    (
        "What is the punishment for murder?",
        ["103", "death", "imprisonment"],
        "Murder punishment — BNS S.103",
    ),
    (
        "What is the punishment for rape under BNS?",
        ["64", "imprisonment"],
        "Rape punishment — BNS S.64",
    ),
    (
        "What is the punishment for theft?",
        ["303", "imprisonment"],
        "Theft punishment — BNS S.303",
    ),
    (
        "What is the punishment for dacoity?",
        ["310", "imprisonment"],
        "Dacoity punishment — BNS S.310",
    ),
    (
        "What is the punishment for robbery?",
        ["309", "imprisonment"],
        "Robbery punishment — BNS S.309",
    ),
    (
        "What is the punishment for cheating?",
        ["318", "imprisonment"],
        "Cheating punishment — BNS S.318",
    ),
    (
        "What is the punishment for dowry death?",
        ["80", "dowry"],
        "Dowry death — BNS S.80",
    ),
    (
        "What is the punishment for defamation?",
        ["356"],
        "Defamation — BNS S.356",
    ),

    # ── Procedure questions ──────────────────────────────────────────────────
    (
        "What is the procedure for filing an FIR?",
        ["173", "police", "cognizable"],
        "FIR procedure — BNSS S.173",
    ),
    (
        "What is the procedure when a person is arrested?",
        ["BNSS", "arrest"],
        "Arrest procedure — BNSS",
    ),
    (
        "What is anticipatory bail?",
        ["482", "bail"],
        "Anticipatory bail — BNSS S.482",
    ),
    (
        "What is a charge sheet?",
        ["193", "report"],
        "Charge sheet — BNSS S.193",
    ),

    # ── Cognizable/bailable status ───────────────────────────────────────────
    (
        "Is kidnapping a bailable offence?",
        ["bailable"],
        "Kidnapping bail status",
    ),
    (
        "Is theft a cognizable offence?",
        ["cognizable"],
        "Theft cognizability",
    ),
    (
        "Is murder cognizable and non-bailable?",
        ["cognizable", "non-bailable"],
        "Murder cognizable/non-bailable",
    ),

    # ── Comparison questions ──────────────────────────────────────────────────
    (
        "What is the difference between robbery and dacoity?",
        ["309", "310"],
        "Robbery vs dacoity",
    ),
    (
        "What are the offences related to assault?",
        ["BNS", "assault"],
        "Assault offences",
    ),

    # ── IPC → BNS replacement ────────────────────────────────────────────────
    (
        "What replaced IPC Section 302?",
        ["103", "murder"],
        "IPC 302 → BNS 103",
    ),
    (
        "What replaced IPC Section 307?",
        ["109", "murder"],
        "IPC 307 → BNS 109",
    ),
    (
        "What replaced IPC Section 376?",
        ["64", "rape"],
        "IPC 376 → BNS 64",
    ),
    (
        "What replaced IPC Section 420?",
        ["318", "cheating"],
        "IPC 420 → BNS 318",
    ),
    (
        "What replaced IPC Section 498A?",
        ["85", "cruelty"],
        "IPC 498A → BNS 85",
    ),
    (
        "What replaced IPC Section 304B?",
        ["80", "dowry"],
        "IPC 304B → BNS 80",
    ),

    # ── CrPC → BNSS replacement ──────────────────────────────────────────────
    (
        "What replaced CrPC Section 154?",
        ["173"],
        "CrPC 154 → BNSS 173",
    ),
    (
        "What replaced CrPC Section 437?",
        ["480", "bail"],
        "CrPC 437 → BNSS 480",
    ),
    (
        "What replaced CrPC Section 438?",
        ["482", "bail"],
        "CrPC 438 → BNSS 482",
    ),

    # ── Edge cases ────────────────────────────────────────────────────────────
    (
        "IPC 302",
        ["103", "murder"],
        "Bare 'IPC 302' (no question words)",
    ),
    (
        "WHAT IS MURDER?",
        ["103"],
        "All-caps query",
    ),
    (
        "What is zero FIR?",
        ["173", "police station"],
        "Zero FIR concept",
    ),
    (
        "What is the punishment for murder by a group of five or more persons?",
        ["103", "death"],
        "Group murder BNS S.103(2)",
    ),
    (
        "What is remand?",
        ["187", "custody"],
        "Remand — BNSS S.187",
    ),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def _colour(text, code):
    return f"{code}{text}{RESET}"

def check_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def run_case(question: str, required: list[str], model: str) -> tuple[bool, str]:
    """Returns (passed, answer_snippet)."""
    try:
        r = requests.post(
            f"{API_URL}/query",
            json={"question": question, "model_name": model},
            timeout=90,
        )
        r.raise_for_status()
        answer  = r.json().get("answer", "")
        al      = answer.lower()
        missing = [kw for kw in required if kw.lower() not in al]
        return (not missing), answer[:300]
    except Exception as e:
        return False, f"ERROR: {e}"


def run_eval(model: str, fast: bool = False, delay: float = 2.0):
    if not check_health():
        print(f"{_colour('ERROR', RED)}: API server not running at {API_URL}")
        print("  Start: uvicorn src.api.server:app --port 8000")
        sys.exit(1)

    cases = EVAL_CASES
    print(f"\n{BOLD}{'='*68}{RESET}")
    print(f"  LegalAI End-to-End Eval  |  {len(cases)} cases  |  model: {model}")
    print(f"{BOLD}{'='*68}{RESET}\n")

    passed = failed = 0
    failures = []

    for i, (question, required, desc) in enumerate(cases, 1):
        label = f"[{i:02d}/{len(cases)}]"
        ok, snippet = run_case(question, required, model)

        if ok:
            print(f"{label} {_colour('PASS', GREEN)}  {desc}")
            passed += 1
        else:
            print(f"{label} {_colour('FAIL', RED)}  {desc}")
            print(f"       Q: {question}")
            print(f"       Missing keywords from answer.")
            print(f"       Ans: {snippet[:200]}...")
            failed += 1
            failures.append((desc, question, snippet))

        if i < len(cases):
            time.sleep(delay)

    # ── Summary ───────────────────────────────────────────────────────────────
    pct   = 100 * passed // len(cases)
    color = GREEN if pct >= 80 else (YELLOW if pct >= 60 else RED)
    print(f"\n{BOLD}{'='*68}{RESET}")
    print(f"  Score: {_colour(f'{passed}/{len(cases)}  ({pct}%)', color)}")
    print(f"{BOLD}{'='*68}{RESET}")

    if failures:
        print(f"\n{_colour('Failed cases:', RED)}")
        for desc, q, _ in failures:
            print(f"  • {desc}")
            print(f"      {q}")

    return passed, failed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LegalAI end-to-end eval")
    parser.add_argument("--model",  default="gemma-3-27b-it",
                        help="Model name to use (default: gemma-3-27b-it)")
    parser.add_argument("--fast",   action="store_true",
                        help="Skip LLM calls — only check /health endpoint")
    parser.add_argument("--delay",  type=float, default=2.0,
                        help="Seconds between API calls (default: 2)")
    args = parser.parse_args()

    if args.fast:
        if check_health():
            print(f"{_colour('API server OK', GREEN)} at {API_URL}")
            sys.exit(0)
        else:
            print(f"{_colour('API server DOWN', RED)} at {API_URL}")
            sys.exit(1)

    passed, failed = run_eval(model=args.model, fast=args.fast, delay=args.delay)
    sys.exit(0 if failed == 0 else 1)

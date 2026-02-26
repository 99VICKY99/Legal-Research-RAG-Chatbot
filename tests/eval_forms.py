"""
tests/eval_forms.py

End-to-end test: send "form N" for every form 1-58 to the live API server
and verify the LLM returns a real answer (not a block/error).

Run with the API server already started:
    uvicorn src.api.server:app --port 8000

Then in a second terminal:
    python tests/eval_forms.py
"""

import re
import sys
import time
import requests

BACKEND = "http://localhost:8000"
DELAY   = 5   # seconds between calls — Gemma free tier: 15k tokens/min


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
    """Extract retry_delay seconds from a 429 error string. Returns 65 if not found."""
    m = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", detail)
    return int(m.group(1)) + 5 if m else 65


def test_form(n: int, max_retries: int = 3) -> tuple[bool, str]:
    """Returns (passed, detail). Auto-retries on 429 rate limit."""
    for attempt in range(max_retries):
        try:
            r = requests.post(
                f"{BACKEND}/query",
                json={"question": f"form {n}"},
                timeout=60,
            )
            res = r.json()
        except Exception as e:
            return False, f"request failed: {e}"

        if "answer" not in res:
            detail = res.get("detail", str(res))
            # 429 rate limit — wait and retry
            if "429" in str(detail) or "quota" in str(detail).lower():
                wait = _parse_retry_delay(str(detail))
                print(f"Form {n:<3}  [rate limit — waiting {wait}s before retry {attempt+1}/{max_retries}]")
                time.sleep(wait)
                continue
            return False, f"no 'answer' key — got: {detail[:120]}"

        answer = res["answer"]
        if "could not generate" in answer.lower():
            return False, f"LLM blocked — {answer[:120]}"
        if len(answer.strip()) < 20:
            return False, f"answer too short: {answer!r}"

        return True, answer[:80].replace("\n", " ")

    return False, f"failed after {max_retries} retries (persistent rate limit)"


def main():
    check_health()

    passed = []
    failed = []

    print(f"Testing forms 1–58 against {BACKEND}  (delay={DELAY}s, auto-retry on 429)\n")
    print(f"{'Form':<6}  {'Status':<6}  Detail")
    print("-" * 72)

    for n in range(1, 59):
        ok, detail = test_form(n)
        status = "PASS" if ok else "FAIL"
        print(f"Form {n:<3}  {status:<6}  {detail}")
        (passed if ok else failed).append(n)
        if n < 58:
            time.sleep(DELAY)

    print("\n" + "=" * 72)
    print(f"Results: {len(passed)}/58 passed")
    if failed:
        print(f"Failed:  {failed}")
    else:
        print("All 58 forms answered successfully.")


if __name__ == "__main__":
    main()

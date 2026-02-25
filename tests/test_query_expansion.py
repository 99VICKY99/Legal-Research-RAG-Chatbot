"""
tests/test_query_expansion.py
Unit tests for src/rag/pipeline.py::expand_query

Pure unit tests — no ChromaDB, no LLM, no network.
Run with: pytest tests/test_query_expansion.py -v
"""
import sys
import unittest.mock as mock
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Patch heavy optional deps so we can import pipeline without installing them
with mock.patch.dict("sys.modules", {
    "google.generativeai":       mock.MagicMock(),
    "chromadb":                  mock.MagicMock(),
    "sentence_transformers":     mock.MagicMock(),
}):
    from src.rag.pipeline import expand_query


# ══════════════════════════════════════════════════════════════════════════════
# IPC → BNS expansion
# ══════════════════════════════════════════════════════════════════════════════

class TestIPCToBNS:
    """IPC section numbers must expand to the correct BNS section hint."""

    @pytest.mark.parametrize("question,expected", [
        # "IPC Section NNN" variants
        ("What replaced IPC Section 302?",     "BNS Section 103 murder"),
        ("IPC Section 307 attempt to murder",  "BNS Section 109 attempt to murder"),
        ("IPC section 376 rape case",          "BNS Section 64 rape"),
        ("IPC Section 420 cheating",           "BNS Section 318 cheating"),
        ("IPC Section 498A cruelty",           "BNS Section 85 cruelty husband wife"),
        ("IPC Section 304B dowry",             "BNS Section 80 dowry death"),
        ("IPC Section 306 suicide",            "BNS Section 108 abetment suicide"),
        ("IPC Section 308 attempt",            "BNS Section 110 attempt culpable homicide"),
        ("IPC Section 354 assault woman",      "BNS Section 74 assault criminal force woman"),
        ("IPC Section 379 theft",              "BNS Section 303 theft"),
        ("IPC Section 392 robbery",            "BNS Section 309 robbery"),
        ("IPC Section 395 dacoity",            "BNS Section 310 dacoity"),
        ("IPC Section 406 trust",              "BNS Section 316 criminal breach of trust"),
        ("IPC Section 427 mischief",           "BNS Section 324 mischief"),
        ("IPC Section 499 defamation",         "BNS Section 356 defamation"),
        ("IPC Section 120B conspiracy",        "BNS Section 61 criminal conspiracy"),
        ("IPC Section 124A sedition",          "BNS Section 152 sovereignty unity integrity India"),
        ("IPC Section 153A religion enmity",   "BNS Section 196 enmity groups religion"),
        # "IPC NNN" (no "Section" word)
        ("What is IPC 302?",                   "BNS Section 103 murder"),
        ("IPC 307 case registered",            "BNS Section 109 attempt to murder"),
        # bare number + contextual word
        ("booked under 302",                   "BNS Section 103 murder"),
        ("accused of 307",                     "BNS Section 109 attempt to murder"),
        ("376 case filed",                     "BNS Section 64 rape"),
        ("420 case cheating",                  "BNS Section 318 cheating"),
        ("498A case against husband",          "BNS Section 85 cruelty husband wife"),
    ])
    def test_expansion(self, question, expected):
        got = expand_query(question)
        assert expected in got, (
            f"\n  Input:    {question!r}"
            f"\n  Expected: {expected!r} in expanded query"
            f"\n  Got:      {got!r}"
        )

    def test_original_text_preserved(self):
        q = "What replaced IPC Section 302?"
        assert expand_query(q).startswith(q)

    def test_no_duplicate_hints(self):
        """Same hint must not appear twice even if the pattern matches multiple times."""
        q = "IPC 302 and IPC Section 302"
        got = expand_query(q)
        assert got.count("BNS Section 103 murder") == 1, (
            f"Hint duplicated in: {got!r}"
        )

    @pytest.mark.parametrize("question", [
        "ipc section 302 murder",
        "IPC SECTION 302 MURDER",
        "Ipc Section 302 Murder",
    ])
    def test_case_insensitive(self, question):
        got = expand_query(question)
        assert "BNS Section 103 murder" in got, (
            f"Case-insensitive match failed for: {question!r}\n  Got: {got!r}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CrPC → BNSS expansion
# ══════════════════════════════════════════════════════════════════════════════

class TestCrPCToBNSS:
    @pytest.mark.parametrize("question,expected", [
        ("What is CrPC Section 154?",        "BNSS Section 173 information cognizable offence FIR"),
        ("CrPC 156 investigation",           "BNSS Section 175 investigation cognizable offence"),
        ("CrPC 161 witness examination",     "BNSS Section 180 examination witnesses police"),
        ("CrPC 164 confession",              "BNSS Section 183 recording confession statement"),
        ("CrPC 167 remand",                  "BNSS Section 187 remand custody detention"),
        ("CrPC 173 charge sheet",            "BNSS Section 193 report police officer investigation charge sheet"),
        ("CrPC 437 bail",                    "BNSS Section 480 bail bailable offence"),
        ("CrPC 438 anticipatory bail",       "BNSS Section 482 anticipatory bail"),
        ("CrPC 482 High Court",              "BNSS Section 528 inherent powers High Court"),
        ("CrPC Section 41 arrest",           "BNSS Section 35 arrest without warrant police"),
        ("CrPC 320 compounding",             "BNSS Section 359 compounding offences"),
    ])
    def test_expansion(self, question, expected):
        got = expand_query(question)
        assert expected in got, (
            f"\n  Input:    {question!r}"
            f"\n  Expected: {expected!r}"
            f"\n  Got:      {got!r}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FIR variants
# ══════════════════════════════════════════════════════════════════════════════

class TestFIR:
    def test_plain_fir(self):
        got = expand_query("How to file a FIR?")
        assert "First Information Report" in got

    def test_zero_fir(self):
        got = expand_query("What is a zero FIR?")
        assert "Section 173 BNSS information cognizable offence any police station" in got

    def test_e_fir(self):
        got = expand_query("Can I file an e-FIR online?")
        assert "Section 173 BNSS electronic" in got

    def test_e_fir_no_hyphen(self):
        got = expand_query("Submit eFIR online")
        assert "Section 173 BNSS electronic" in got

    def test_zero_fir_before_plain_fir(self):
        """zero FIR hint must appear before generic FIR hint (ordering check)."""
        got = expand_query("What is a zero FIR?")
        idx_zero  = got.find("any police station")
        idx_plain = got.find("First Information Report")
        assert idx_zero < idx_plain or idx_plain == -1, (
            f"Ordering wrong — zero FIR hint should come first: {got!r}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Generic abbreviations & legal jargon
# ══════════════════════════════════════════════════════════════════════════════

class TestGenericExpansions:
    @pytest.mark.parametrize("question,expected", [
        ("What does BNS say about theft?",       "Bharatiya Nyaya Sanhita"),
        ("BNSS procedure for arrest",            "Bharatiya Nagarik Suraksha Sanhita"),
        ("What was the IPC?",                    "Indian Penal Code"),
        ("CrPC procedure",                       "Code of Criminal Procedure"),
        ("What is S.103 of BNS?",               "Section 103"),
        ("When is a charge sheet filed?",        "Section 193 BNSS report police officer investigation"),
        ("What is a challan under BNSS?",        "Section 193 BNSS report police officer investigation"),
        ("What is remand in BNSS?",              "Section 187 BNSS custody detention investigation"),
        ("How to get anticipatory bail?",        "Section 482 BNSS anticipatory bail"),
        ("What is a non-cognizable offence?",    "non-cognizable offence complaint Magistrate police"),
        ("Explain chowki jurisdiction",          "police station officer in charge"),
    ])
    def test_expansion(self, question, expected):
        got = expand_query(question)
        assert expected in got, (
            f"\n  Input:    {question!r}"
            f"\n  Expected: {expected!r}"
            f"\n  Got:      {got!r}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# No-expansion cases (query must remain unchanged)
# ══════════════════════════════════════════════════════════════════════════════

class TestNoExpansion:
    @pytest.mark.parametrize("question", [
        "What is the punishment for murder?",
        "What is the difference between robbery and dacoity?",
        "What is the weather today?",
    ])
    def test_unchanged(self, question):
        assert expand_query(question) == question, (
            f"Expected no expansion for: {question!r}"
        )

    def test_empty_string(self):
        assert expand_query("") == ""

    def test_very_long_query(self):
        q = "What is the legal procedure " * 50
        got = expand_query(q)
        assert got == q  # no patterns match → unchanged


# ══════════════════════════════════════════════════════════════════════════════
# Ordering guard: specific patterns before generic ones
# ══════════════════════════════════════════════════════════════════════════════

class TestExpansionOrdering:
    def test_ipc_section_302_before_generic_ipc(self):
        """'IPC Section 302' must match the specific pattern, not just add 'Indian Penal Code'."""
        q = "What replaced IPC Section 302?"
        got = expand_query(q)
        assert "BNS Section 103 murder" in got, "Specific IPC 302 pattern must fire"
        # Indian Penal Code hint may also appear (from generic \bIPC\b), that's fine

    def test_ipc_499_vs_ipc_500(self):
        """IPC 499 (defamation) and IPC 500 (punishment for defamation) expand differently."""
        got_499 = expand_query("IPC Section 499 defamation")
        got_500 = expand_query("IPC Section 500 defamation")
        assert "BNS Section 356 defamation" in got_499
        assert "BNS Section 356 defamation punishment" in got_500

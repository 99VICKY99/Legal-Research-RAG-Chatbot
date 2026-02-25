"""
conftest.py — pytest path setup so all tests can import from project root.
"""
import sys
from pathlib import Path

# Add project root to sys.path so 'src.*' imports work from any test file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

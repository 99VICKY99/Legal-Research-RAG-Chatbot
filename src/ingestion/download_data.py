"""
Data acquisition script for BNS documents.

Two ways to get the data:
  1. Manual: Place the PDF files directly in data/raw/
  2. Auto-download: Run this script — it will download from the official
     Gazette of India website automatically.
"""

import os
import sys
import urllib.request


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# Official Gazette of India sources for the Bharatiya Nyaya Sanhita (BNS)
# Published: December 25, 2023 | Effective: July 1, 2024
DOCUMENTS = [
    {
        "filename": "250883_english_01042024.pdf",
        "url": "https://egazette.gov.in/WriteReadData/2023/250883.pdf",
        "description": "BNS Gazette Part 1 (Sections 1–112)",
    },
    {
        "filename": "250884_2_english_01042024.pdf",
        "url": "https://egazette.gov.in/WriteReadData/2023/250884.pdf",
        "description": "BNS Gazette Part 2 (Sections 113–358 + Schedules)",
    },
]


def download_with_progress(url, filepath, description):
    """Download a file with a progress indicator."""
    print(f"  Downloading: {description}")
    print(f"  URL: {url}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            percent = min(percent, 100)
            print(f"\r  Progress: {percent}%", end="", flush=True)

    try:
        urllib.request.urlretrieve(url, filepath, reporthook)
        print(f"\r  Done: {os.path.basename(filepath)}")
        return True
    except Exception as e:
        print(f"\r  Failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False


def acquire_data():
    os.makedirs(DATA_DIR, exist_ok=True)

    all_present = True
    needs_download = []

    # Check which files already exist
    print("Checking data/raw/ for existing files...\n")
    for doc in DOCUMENTS:
        filepath = os.path.join(DATA_DIR, doc["filename"])
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  [FOUND] {doc['filename']} ({size_mb:.1f} MB) — skipping download")
        else:
            print(f"  [MISSING] {doc['filename']} — will download")
            needs_download.append(doc)
            all_present = False

    if all_present:
        print("\nAll documents already present. Ready to build index.")
        return

    # Download missing files
    print(f"\nDownloading {len(needs_download)} missing file(s) from official source...\n")
    failed = []

    for doc in needs_download:
        filepath = os.path.join(DATA_DIR, doc["filename"])
        success = download_with_progress(doc["url"], filepath, doc["description"])
        if not success:
            failed.append(doc)
        print()

    if failed:
        print("Some downloads failed. Please download manually and place in data/raw/:")
        for doc in failed:
            print(f"\n  File    : {doc['filename']}")
            print(f"  Source  : {doc['url']}")
            print(f"  Or visit: https://www.indiacode.nic.in (search 'Bharatiya Nyaya Sanhita')")
        sys.exit(1)
    else:
        print("All documents downloaded successfully. Ready to build index.")


if __name__ == "__main__":
    print("=" * 55)
    print("  BNS Document Acquisition")
    print("=" * 55)
    print()
    acquire_data()

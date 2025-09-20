#!/usr/bin/env python3
"""
Fetch Real GPT-OSS Modeling Files
Downloads official modeling files from HuggingFace Hub
"""

import os
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

MODEL_ID = "openai/gpt-oss-20b"

# All possible custom files HuggingFace might need for GPT-OSS
CUSTOM_FILES = [
    "modeling_gpt_oss.py",
    "configuration_gpt_oss.py",
    "tokenization_gpt_oss.py",
    "modeling_utils.py",        # in case they split helpers
    "convert_gpt_oss_weights.py" # optional utility
]

def get_local_snapshot():
    """Locate the local snapshot folder for GPT-OSS model."""
    base = Path(os.environ["USERPROFILE"]) / ".cache" / "huggingface" / "hub" / f"models--{MODEL_ID.replace('/', '--')}"
    snap_dir = base / "snapshots"
    snapshots = list(snap_dir.glob("*"))
    if not snapshots:
        raise RuntimeError(f"No local snapshots found in {snap_dir}")
    # Pick the latest snapshot by modified time
    snapshots.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return snapshots[0]

def fetch_and_copy(filename, snapshot_path):
    """Download a file from HuggingFace Hub and copy it into local snapshot."""
    try:
        print(f"Fetching {filename}...")
        file_path = hf_hub_download(repo_id=MODEL_ID, filename=filename, repo_type="model")
        target_file = snapshot_path / filename
        target_file.write_text(Path(file_path).read_text(encoding="utf-8"), encoding="utf-8")
        print(f"SUCCESS: Saved {filename} to {target_file}")
        return True
    except Exception as e:
        print(f"WARNING: Could not fetch {filename}: {e}")
        return False

def main():
    print(f"=== Fetching GPT-OSS custom files from {MODEL_ID} ===")

    try:
        snapshot_path = get_local_snapshot()
        print(f"Using local snapshot: {snapshot_path}")
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return False

    # First, list repo files on HF
    try:
        print("Querying HuggingFace Hub for available files...")
        repo_files = list_repo_files(MODEL_ID, repo_type="model")
        print(f"Found {len(repo_files)} files in repo")

        # Show which custom files exist
        print("\nChecking for custom modeling files:")
        for f in CUSTOM_FILES:
            if f in repo_files:
                print(f"  [EXISTS] {f}")
            else:
                print(f"  [MISSING] {f}")
    except Exception as e:
        print(f"WARNING: Could not list repo files: {e}")
        print("Will attempt to download standard files anyway...")
        repo_files = CUSTOM_FILES  # fallback

    # Download required files if they exist
    print("\nDownloading available files...")
    downloaded = 0
    for f in CUSTOM_FILES:
        if f in repo_files:
            if fetch_and_copy(f, snapshot_path):
                downloaded += 1
        else:
            print(f"SKIPPING {f}, not found in repo")

    if downloaded > 0:
        print(f"\nSUCCESS! Downloaded {downloaded} files to your local snapshot.")
        print("Now run: python test_gpt_oss_minimal.py")
    else:
        print("\nWARNING: No custom files were downloaded. The model may not have custom implementations.")
        print("This could mean the model uses standard transformers classes.")

    return downloaded > 0

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
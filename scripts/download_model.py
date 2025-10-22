#!/usr/bin/env python3
"""
Download a full Hugging Face model (weights + tokenizer + configs)
to a local directory for offline or vLLM use.

Usage:
    python download_model.py meta-llama/Llama-3.1-8B-Instruct
"""

import os
import sys
import platform
from huggingface_hub import snapshot_download

# Ensure Python version is 3.8+
if platform.python_version_tuple() < ('3', '8'):
    print("❌ Python 3.8 or higher is required.")
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        print("❌ Usage: python download_model.py <repo_id>")
        print("Example: python download_model.py meta-llama/Llama-3.1-8B-Instruct")
        sys.exit(1)

    # Get model repo ID from CLI argument
    model_id = sys.argv[1].strip()

    # Local directory is based on the last part of the repo_id
    model_name = model_id.split("/")[-1]
    local_dir = os.path.join("/tmp/models", model_name)

    # Optional: read token from environment (if gated/private)
    hf_token = os.getenv("HF_TOKEN", None)

    print(f"📦 Downloading model: {model_id}")
    print(f"📂 Target directory: {local_dir}")

    os.makedirs(local_dir, exist_ok=True)

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # get a full copy (no symlinks)
            resume_download=True,          # resume if interrupted
            token=hf_token,                # optional auth token
        )
        print(f"✅ Download complete! Files saved in: {local_dir}")

    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        print("Ensure your HF_TOKEN is valid and the model ID is correct.")
        sys.exit(1)

    print("\nYou can now run vLLM with:")
    print(f"python -m vllm.entrypoints.openai.api_server --model {local_dir}")

if __name__ == "__main__":
    main()


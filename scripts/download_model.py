#!/usr/bin/env python3
"""Download Qwen3-0.6B base model from Hugging Face."""

from huggingface_hub import snapshot_download
import os

MODEL_NAME = "Qwen/Qwen3-0.6B"
LOCAL_DIR = "./models/base/qwen3-0.6b"

print(f"Downloading {MODEL_NAME} to {LOCAL_DIR}...")
print("This may take several minutes depending on your connection.")

snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"\nModel downloaded successfully to {LOCAL_DIR}")

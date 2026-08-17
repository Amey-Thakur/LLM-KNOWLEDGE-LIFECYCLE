"""
===============================================================================
FILE         : sync_space.py
PROJECT      : The Knowledge Lifecycle of Large Language Models
PURPOSE      : Mirror the space/ directory to the Hugging Face Space, so the
               live demonstration is always exactly what this repository holds.
TECH STACK   : Python 3, huggingface_hub
AUTHORS      : Amey Thakur (https://github.com/Amey-Thakur)
               Sarvesh Talele (https://github.com/sarveshtalele)
REPOSITORY   : https://github.com/Amey-Thakur/LLM-KNOWLEDGE-LIFECYCLE
RELEASE DATE : August 18, 2026
LICENSE      : CC BY 4.0
===============================================================================

Run by hand when the demonstration changes. It pushes one way, from the
repository to the Space, and deletes files on the Space that no longer exist
here, so the two cannot drift apart.

GitHub is the source of truth: space/ always holds exactly what the Space
serves. Editing the Space through the Hugging Face web interface is therefore
not durable, since the next run overwrites it. Change space/ instead.

    HF_TOKEN=<token with write access> python .github/scripts/sync_space.py

The token is read from the environment and never written to disk.
"""

import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

# The Space this repository owns. Hard-coded rather than passed in, because a
# mistyped identifier would publish this demonstration to the wrong account.
SPACE_ID = "ameythakur/llm-knowledge-lifecycle"

# Only the deployed demonstration is mirrored. The paper, the experiments, and
# the repository README stay on GitHub.
SOURCE = Path(__file__).resolve().parents[2] / "space"

# Housekeeping that the Space has no use for.
IGNORE = ["*.pyc", "__pycache__/*", ".DS_Store"]


def main() -> int:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("::error::HF_TOKEN secret is not set; the Space was not updated.")
        print("Create a fine-grained token with write access to "
              f"{SPACE_ID} and add it as the HF_TOKEN repository secret.")
        return 1

    if not SOURCE.is_dir():
        print(f"::error::{SOURCE} does not exist.")
        return 1

    api = HfApi(token=token)

    account = api.whoami()["name"]
    owner = SPACE_ID.split("/")[0]
    if account != owner:
        print(f"::error::token belongs to '{account}', not '{owner}'.")
        return 1

    files = sorted(p.name for p in SOURCE.iterdir() if p.is_file())
    print(f"Mirroring {len(files)} file(s) to {SPACE_ID}: {', '.join(files)}")

    # delete_patterns removes anything on the Space that is no longer in
    # space/, which is what makes this a mirror rather than an accumulation.
    api.upload_folder(
        folder_path=str(SOURCE),
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Knowledge Lifecycle",
        ignore_patterns=IGNORE,
        delete_patterns="*",
    )

    print(f"Space updated: https://huggingface.co/spaces/{SPACE_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Subset excluded-docs dataset to 250 documents and push to HuggingFace.

This script loads the full excluded-docs dataset from abehandlerorg/excluded-docs,
randomly samples 250 documents, and pushes them to abehandlerorg/excluded-docs-mini.
"""

import random
from pathlib import Path

from datasets import load_dataset


def push_to_hub(ds, repo_name):
    """Push dataset to HuggingFace hub."""
    pathto = Path.home() / ".cache" / "huggingface" / "token_write"
    with open(pathto, "r") as inf:
        hf_token = inf.read().strip("\n")
    print(f"▶ Pushing to hub: {repo_name}")
    ds.push_to_hub(
        repo_name,
        private=False,
        max_shard_size="5GB",
        token=hf_token,
    )


def main():
    """Load excluded-docs, sample 250 docs, and push to excluded-docs-mini."""
    print("▶ Loading dataset from abehandlerorg/excluded-docs...")
    dataset = load_dataset("abehandlerorg/excluded-docs")

    ds = dataset["train"]

    print(f"▶ Original dataset size: {len(ds)}")

    # Randomly sample 250 documents
    n_samples = min(250, len(ds))
    indices = random.sample(range(len(ds)), n_samples)
    ds_subset = ds.select(indices)

    print(f"▶ Sampled {len(ds_subset)} documents")

    # Push to hub
    push_to_hub(ds_subset, "abehandlerorg/excluded-docs-mini")
    print("✓ Done!")


if __name__ == "__main__":
    random.seed(42)  # Set seed for reproducibility
    main()

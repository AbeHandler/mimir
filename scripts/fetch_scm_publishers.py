#!/usr/bin/env python3
"""
Fetch all datasets ending in _scm from abehandlerorg on HuggingFace
and extract publisher names to configs/single_publishers.txt

Usage:
    python scripts/fetch_scm_publishers.py
"""

from huggingface_hub import HfApi


def main():
    """Fetch HF datasets ending in _scm and extract publisher names."""
    api = HfApi()

    # Fetch all datasets from abehandlerorg
    datasets = api.list_datasets(author="abehandlerorg")

    # Extract publisher names from datasets ending in _scm
    publishers = []
    for dataset in datasets:
        dataset_name = dataset.id

        # Check if dataset ends with _scm
        if dataset_name.endswith("_scm"):
            # Split on /, take the last part, and remove _scm suffix
            publisher = dataset_name.split("/")[-1].replace("_scm", "")
            publishers.append(publisher)

    # Sort for consistency
    publishers.sort()

    # Write to configs/single_publishers.txt
    output_path = "configs/single_publishers.txt"
    with open(output_path, "w") as f:
        for publisher in publishers:
            f.write(f"{publisher}\n")

    print(f"Found {len(publishers)} publishers ending in _scm")
    print(f"Written to {output_path}")

    # Print first few as spot check
    if publishers:
        print("\nSpot check (first 5):")
        for pub in publishers[:5]:
            print(f"  - {pub}")


if __name__ == "__main__":
    main()

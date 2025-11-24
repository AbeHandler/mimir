"""
Create abehandlerorg/confounddatasetxpress by filtering confounddataset.

This script:
1. Loads abehandlerorg/confounddataset from HuggingFace
2. Selects rows where in_blocksbin=0
3. Pushes the filtered dataset to abehandlerorg/confounddatasetxpress
"""

from datasets import load_dataset


def main():
    # Load the original dataset
    print("Loading abehandlerorg/confounddataset...")
    dataset = load_dataset(
        "abehandlerorg/confounddataset",
        split="train",
        download_mode="force_redownload"
    )
    print(f"Original dataset size: {len(dataset)} rows")

    # Filter for rows where in_blocksbin=0
    print("\nFiltering for in_blocksbin=0...")
    filtered_dataset = dataset.filter(lambda x: x["in_blocksbin"] == 0)
    print(f"Filtered dataset size: {len(filtered_dataset)} rows")

    print("\nPushing to abehandlerorg/confounddatasetxpress...")
    filtered_dataset.push_to_hub(
        "abehandlerorg/confounddatasetxpress",
        split="train"
    )
    print("✓ Successfully created abehandlerorg/confounddatasetxpress")


if __name__ == "__main__":
    main()

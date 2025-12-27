#!/usr/bin/env python3
"""Compute SBERT embeddings for individual sentences from sentence-level data."""

import json
import gzip
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any
import random


def load_merged_sentences(filepath: Path) -> Dict[str, Dict[str, Any]]:
    """Load merged sentence data from cache file.

    Args:
        filepath: Path to merged_sentences.jsonl.gz

    Returns:
        Dictionary mapping sentence IDs to merged data
    """
    print(f"Loading merged sentences from {filepath}")

    merged_data = {}
    with gzip.open(filepath, 'rt') as f:
        for line in tqdm(f, desc="Loading sentences"):
            record = json.loads(line)
            sent_id = record.pop('id')
            merged_data[sent_id] = record

    print(f"✓ Loaded {len(merged_data):,} sentences\n")
    return merged_data


def extract_sentence_texts(merged_data: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Extract sentence texts from merged data.

    Args:
        merged_data: Merged sentence data from all sources

    Returns:
        Dictionary mapping sentence IDs to text
    """
    print("Extracting sentence texts...")

    sent_id_to_text = {}

    for sent_id, sources in tqdm(merged_data.items(), desc="Extracting texts"):
        # Get text from first available source
        text = ''
        for source_key in ['pair1_treated_run1', 'pair2_treated_run3', 'pair2_control_run4']:
            source_data = sources.get(source_key)
            if source_data:
                text = source_data.get('text', '')
                if text:
                    break

        if text:
            sent_id_to_text[sent_id] = text

    print(f"✓ Extracted {len(sent_id_to_text):,} sentence texts\n")
    return sent_id_to_text


def compute_sbert_embeddings(
    sent_id_to_text: Dict[str, str],
    model_name: str = 'all-MiniLM-L6-v2',
    batch_size: int = 128
) -> Dict[str, List[float]]:
    """Compute SBERT embeddings for sentences.

    Args:
        sent_id_to_text: Dictionary mapping sentence IDs to text
        model_name: Name of the sentence-transformers model to use
        batch_size: Batch size for encoding

    Returns:
        Dictionary mapping sentence IDs to SBERT embeddings
    """
    print(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"✓ Model loaded (embedding dim: {model.get_sentence_embedding_dimension()})\n")

    print("Computing SBERT embeddings...")

    # Prepare data for batch processing - randomize order
    sent_ids = list(sent_id_to_text.keys())
    random.shuffle(sent_ids)  # Randomize for better sampling if interrupted
    texts = [sent_id_to_text[sid] for sid in sent_ids]

    # Compute embeddings in batches
    sent_id_to_embed = {}

    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches"):
        batch_ids = sent_ids[i:i+batch_size]
        batch_texts = texts[i:i+batch_size]

        # Encode batch
        embeddings = model.encode(
            batch_texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Store embeddings
        for sent_id, embedding in zip(batch_ids, embeddings):
            sent_id_to_embed[sent_id] = embedding.tolist()

    print(f"✓ Computed {len(sent_id_to_embed):,} embeddings\n")
    return sent_id_to_embed


def save_embeddings(
    sent_id_to_embed: Dict[str, List[float]],
    output_file: Path
):
    """Save embeddings to gzipped JSONL file.

    Args:
        sent_id_to_embed: Dictionary mapping sentence IDs to embeddings
        output_file: Path to output file
    """
    print(f"Saving embeddings to {output_file}")

    with gzip.open(output_file, 'wt') as f:
        for sent_id, embedding in tqdm(sent_id_to_embed.items(), desc="Writing embeddings"):
            record = {
                'id': sent_id,
                'embedding': embedding
            }
            f.write(json.dumps(record) + '\n')

    print(f"✓ Saved {len(sent_id_to_embed):,} embeddings")


def main():
    """Main entry point."""
    # Input file
    merged_file = Path("/tmp/merged_sentences.jsonl.gz")

    # Output file
    output_file = Path("/tmp/sentence_sbert_embeddings.jsonl.gz")

    if not merged_file.exists():
        print(f"Error: {merged_file} not found")
        print("Please run analyze_sentences.py first to generate merged sentences")
        return

    if output_file.exists():
        print(f"✓ SBERT embeddings already exist: {output_file}")
        print("Delete this file to recompute embeddings.")
        return

    # Load merged sentences
    merged_data = load_merged_sentences(merged_file)

    # Extract sentence texts
    sent_id_to_text = extract_sentence_texts(merged_data)

    # Compute SBERT embeddings
    sent_id_to_embed = compute_sbert_embeddings(
        sent_id_to_text,
        model_name='all-MiniLM-L6-v2',
        batch_size=128
    )

    # Save embeddings
    save_embeddings(sent_id_to_embed, output_file)

    print(f"\n{'='*80}")
    print(f"SBERT EMBEDDING COMPUTATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute SBERT embeddings for sentences")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute embeddings even if output file exists"
    )

    args = parser.parse_args()

    if args.recompute:
        output_file = Path("/tmp/sentence_sbert_embeddings.jsonl.gz")
        if output_file.exists():
            print(f"Deleting existing embeddings file: {output_file}")
            output_file.unlink()

    main()

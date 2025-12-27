#!/usr/bin/env python3
"""Analyze and merge sentence-level data from multiple JSONL files."""

import json
import gzip
import numpy as np
from tqdm import tqdm as tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Optional, Tuple


def extract_source_key_from_path(file_path: Path) -> Optional[str]:
    """
    Extract source identifier from file path.

    Args:
        file_path: Path to sentences.jsonl file

    Returns:
        Source key like 'pair1_treated_run1' or None if not identified
    """
    parts = file_path.parts
    for part in parts:
        if 'sutva_click2houston_com' in part and '_pair' in part:
            if '_pair1_treated_run1' in part:
                return 'pair1_treated_run1'
            elif '_pair2_treated_run3' in part:
                return 'pair2_treated_run3'
            elif '_pair2_control_run4' in part and '_filtered' not in part:
                return 'pair2_control_run4'
    return None


def load_jsonl_file(file_path: Path, source_key: str, base_dir: Path) -> Tuple[Dict[str, Any], int]:
    """
    Load a single JSONL file and extract sentence data.

    Args:
        file_path: Path to the JSONL file
        source_key: Source identifier for this file
        base_dir: Base directory for relative path display

    Returns:
        Tuple of (data dictionary keyed by sentence ID, count of loaded records)
    """
    print(f"Processing {source_key}: {file_path.relative_to(base_dir)}")

    data = {}
    line_count = 0

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                record = json.loads(line)
                doc_id = record.get('id')

                if doc_id:
                    data[doc_id] = record
                    line_count += 1

                if line_num % 100000 == 0:
                    print(f"  Processed {line_num:,} lines...")

            except json.JSONDecodeError as e:
                print(f"  Warning: JSON decode error at line {line_num}: {e}")
                continue

    print(f"  Loaded {line_count:,} records from {source_key}\n")
    return data, line_count


def load_sentence_level_data(base_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load sentence-level data from multiple JSONL files and merge by ID.

    Args:
        base_dir: Base directory containing sentences.jsonl files

    Returns:
        Dictionary mapping sentence IDs to merged data from all files
        {
            'sentence_id': {
                'pair1_treated_run1': {...},
                'pair2_treated_run3': {...},
                'pair2_control_run4': {...}
            }
        }
    """
    # Find all sentences.jsonl files
    jsonl_files = list(base_dir.rglob("sentences.jsonl"))

    if not jsonl_files:
        raise ValueError(f"No sentences.jsonl files found in {base_dir}")

    print(f"Found {len(jsonl_files)} sentences.jsonl files\n")

    # Dictionary to store merged data: {id: {source: data}}
    merged_data = defaultdict(dict)

    for file_path in jsonl_files:
        source_key = extract_source_key_from_path(file_path)

        if source_key is None:
            print(f"Warning: Could not identify source for {file_path.relative_to(base_dir)}")
            continue

        # Load data from this file
        file_data, _ = load_jsonl_file(file_path, source_key, base_dir)

        # Merge into combined dictionary
        for doc_id, record in file_data.items():
            merged_data[doc_id][source_key] = record

    return dict(merged_data)


def compute_sentence_loss(tokens: List[Dict]) -> float:
    """
    Compute average loss for a sentence from token log probabilities.

    Args:
        tokens: List of token dictionaries with 'log_prob' field

    Returns:
        Average negative log probability (loss) for the sentence
    """
    if not tokens:
        return np.nan

    log_probs = []
    for token in tokens:
        log_prob = token.get('log_prob')
        if log_prob is not None and not np.isnan(log_prob):
            log_probs.append(log_prob)

    if not log_probs:
        return np.nan

    # Return negative average log probability (loss)
    return -np.mean(log_probs)


def analyze_sentences(merged_data: Dict[str, Dict[str, Any]], sent_id2embed: Dict[str, List[float]] = None) -> Dict[str, Dict[str, float]]:
    """
    Compute ATT (Average Treatment Effect) for each sentence for both run1 and run3.
    ATT_run1 = loss_run1 - loss_run4
    ATT_run3 = loss_run3 - loss_run4

    Args:
        merged_data: Merged sentence data from all sources
        sent_id2embed: Optional dictionary mapping sentence IDs to SBERT embeddings

    Returns:
        Dictionary mapping sentence IDs to analysis results:
        {
            'sentence_id': {
                'loss_run1': float,
                'loss_run3': float,
                'loss_run4': float,
                'att_run1': float,
                'att_run3': float,
                'text': str,
                'doc_id': str,
                'sentence_number': int,
                'sentence_position': float,
                'embedding': list (if sent_id2embed provided)
            }
        }
    """
    print(f"\n{'='*80}")
    print(f"COMPUTING SENTENCE-LEVEL ATT FOR RUN1 AND RUN3")
    print(f"{'='*80}\n")

    # First pass: count sentences per document to compute total_sentences
    print("Counting sentences per document...")
    doc_sentence_counts = defaultdict(int)
    for sent_id in merged_data.keys():
        # Parse sentence ID format: d{doc_num}s{sentence_num}
        if sent_id.startswith('d') and 's' in sent_id:
            doc_part = sent_id.split('s')[0]  # e.g., "d0"
            doc_sentence_counts[doc_part] += 1

    print(f"Found {len(doc_sentence_counts):,} documents")

    # Collect stats on sentence IDs and embeddings
    sent_ids_with_embeddings = set()
    sent_ids_without_embeddings = set()

    results = {}

    for sent_id, sources in tqdm(merged_data.items(), desc="Analyzing sentences"):
        # Get data for all runs
        run1_data = sources.get('pair1_treated_run1')
        run3_data = sources.get('pair2_treated_run3')
        run4_data = sources.get('pair2_control_run4')

        # Compute losses for available runs
        loss_run1 = compute_sentence_loss(run1_data.get('tokens', [])) if run1_data else np.nan
        loss_run3 = compute_sentence_loss(run3_data.get('tokens', [])) if run3_data else np.nan
        loss_run4 = compute_sentence_loss(run4_data.get('tokens', [])) if run4_data else np.nan

        # Compute ATT for run1 and run3
        att_run1 = loss_run1 - loss_run4 if not np.isnan(loss_run1) and not np.isnan(loss_run4) else np.nan
        att_run3 = loss_run3 - loss_run4 if not np.isnan(loss_run3) and not np.isnan(loss_run4) else np.nan

        # Get text from first available source
        text = ''
        doc_id = ''
        for data in [run1_data, run3_data, run4_data]:
            if data:
                text = data.get('text', '')
                doc_id = data.get('doc_id', '')
                break

        # Extract sentence position information from sent_id (format: d{doc_num}s{sentence_num})
        sentence_number = -1
        sentence_position = np.nan
        if sent_id.startswith('d') and 's' in sent_id:
            try:
                parts = sent_id.split('s')
                doc_part = parts[0]  # e.g., "d0"
                sentence_number = int(parts[1])  # e.g., 0, 1, 2, ...
                total_sentences = doc_sentence_counts.get(doc_part, 0)
                if total_sentences > 0:
                    sentence_position = sentence_number / total_sentences
            except (ValueError, IndexError):
                pass

        # Build result dictionary
        result = {
            'loss_run1': loss_run1,
            'loss_run3': loss_run3,
            'loss_run4': loss_run4,
            'att_run1': att_run1,
            'att_run3': att_run3,
            'text': text,
            'doc_id': doc_id,
            'sentence_number': sentence_number,
            'sentence_position': sentence_position
        }

        # Add SBERT embedding if available
        if sent_id2embed is not None:
            embedding = sent_id2embed.get(sent_id)
            if embedding is not None:
                result['embedding'] = embedding
                sent_ids_with_embeddings.add(sent_id)
            else:
                sent_ids_without_embeddings.add(sent_id)

        results[sent_id] = result

    # Print summary statistics for both runs
    valid_att_run1 = [r['att_run1'] for r in results.values() if not np.isnan(r['att_run1'])]
    valid_att_run3 = [r['att_run3'] for r in results.values() if not np.isnan(r['att_run3'])]

    print(f"\nATT Statistics for RUN1 (run1 - run4):")
    if valid_att_run1:
        print(f"  Valid ATTs: {len(valid_att_run1):,}")
        print(f"  Mean ATT: {np.mean(valid_att_run1):.6f}")
        print(f"  Median ATT: {np.median(valid_att_run1):.6f}")
        print(f"  Std ATT: {np.std(valid_att_run1):.6f}")
        print(f"  Min ATT: {np.min(valid_att_run1):.6f}")
        print(f"  Max ATT: {np.max(valid_att_run1):.6f}")

    print(f"\nATT Statistics for RUN3 (run3 - run4):")
    if valid_att_run3:
        print(f"  Valid ATTs: {len(valid_att_run3):,}")
        print(f"  Mean ATT: {np.mean(valid_att_run3):.6f}")
        print(f"  Median ATT: {np.median(valid_att_run3):.6f}")
        print(f"  Std ATT: {np.std(valid_att_run3):.6f}")
        print(f"  Min ATT: {np.min(valid_att_run3):.6f}")
        print(f"  Max ATT: {np.max(valid_att_run3):.6f}")

    # Print detailed embedding statistics
    if sent_id2embed is not None:
        with_embeddings = sum(1 for r in results.values() if 'embedding' in r)
        without_embeddings = len(results) - with_embeddings

        print(f"\n{'='*80}")
        print(f"SBERT EMBEDDING MATCHING STATISTICS")
        print(f"{'='*80}")
        print(f"\nSentence-level stats:")
        print(f"  Total sentences: {len(results):,}")
        print(f"  Sentences with SBERT embeddings: {with_embeddings:,} ({100*with_embeddings/len(results):.1f}%)")
        print(f"  Sentences without SBERT embeddings: {without_embeddings:,} ({100*without_embeddings/len(results):.1f}%)")

        # Show sample sentence IDs without embeddings
        if sent_ids_without_embeddings:
            sample_missing = list(sent_ids_without_embeddings)[:5]
            print(f"\nSample sentence IDs without embeddings:")
            for sent_id in sample_missing:
                print(f"  - {sent_id}")
            if len(sent_ids_without_embeddings) > 5:
                print(f"  ... and {len(sent_ids_without_embeddings) - 5:,} more")

    return results


def analyze_merged_data(merged_data: Dict[str, Dict[str, Any]]):
    """Analyze the merged sentence-level data."""

    total_ids = len(merged_data)

    # Count how many sources each ID appears in
    source_counts = defaultdict(int)
    for doc_id, sources in merged_data.items():
        source_counts[len(sources)] += 1

    print(f"\n{'='*80}")
    print(f"MERGED DATA SUMMARY")
    print(f"{'='*80}")
    print(f"Total unique IDs: {total_ids:,}")
    print(f"\nID distribution by number of sources:")
    for num_sources in sorted(source_counts.keys(), reverse=True):
        count = source_counts[num_sources]
        pct = 100 * count / total_ids
        print(f"  {num_sources} source(s): {count:,} IDs ({pct:.1f}%)")

    # Find which sources are present
    all_sources = set()
    for sources_dict in merged_data.values():
        all_sources.update(sources_dict.keys())

    print(f"\nSources found:")
    for source in sorted(all_sources):
        count = sum(1 for d in merged_data.values() if source in d)
        print(f"  {source}: {count:,} IDs")

    # Show example merged record
    if merged_data:
        print(f"\n{'='*80}")
        print(f"EXAMPLE MERGED RECORD")
        print(f"{'='*80}")
        example_id = next(iter(merged_data))
        example_data = merged_data[example_id]
        print(f"\nID: {example_id}")
        print(f"Sources: {list(example_data.keys())}")
        print(f"\nSample data from first source:")
        first_source = next(iter(example_data.keys()))
        sample_data = example_data[first_source]
        # Print first few keys
        for key in list(sample_data.keys())[:5]:
            value = sample_data[key]
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"  {key}: {value}")


def load_merged_from_cache(cache_file: Path) -> Dict[str, Dict[str, Any]]:
    """Load merged data from cache file."""
    merged_data = {}
    print(f"Loading from cache: {cache_file}")

    with gzip.open(cache_file, 'rt') as f:
        for line in tqdm(f, desc="Loading cached data"):
            record = json.loads(line)
            doc_id = record.pop('id')
            merged_data[doc_id] = record

    print(f"✓ Loaded {len(merged_data):,} records from cache\n")
    return merged_data


def rebuild_merged_data(base_dir: Path, output_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Rebuild merged data from source files and save to cache.

    Args:
        base_dir: Base directory containing sentences.jsonl files
        output_file: Path to save merged data cache file

    Returns:
        Dictionary mapping sentence IDs to merged data from all sources
    """
    print(f"No cached file found at {output_file}")
    print(f"Rebuilding from source files...\n")

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return {}

    print(f"Loading sentence-level data from {base_dir}\n")

    # Load and merge data
    merged_data = load_sentence_level_data(base_dir)

    # Analyze the merged data
    analyze_merged_data(merged_data)

    # Save merged data
    print(f"\n{'='*80}")
    print(f"SAVING MERGED DATA")
    print(f"{'='*80}")
    print(f"Writing to: {output_file}")

    with gzip.open(output_file, 'wt') as f:
        for doc_id, sources_data in tqdm(merged_data.items(), desc="Writing merged data"):
            record = {
                'id': doc_id,
                **sources_data
            }
            f.write(json.dumps(record) + '\n')

    print(f"✓ Wrote {len(merged_data):,} merged records")

    return merged_data


def clean_cache(write_dir: Path = Path("/tmp")):
    """Delete cached files to prepare for a fresh run.

    Args:
        write_dir: Directory containing cached files
    """
    cache_files = [
        write_dir / "merged_sentences.jsonl.gz",
        write_dir / "sentence_att_results.jsonl.gz"
    ]

    print(f"\n{'='*80}")
    print(f"CLEANING CACHE")
    print(f"{'='*80}\n")

    deleted_count = 0
    for cache_file in cache_files:
        if cache_file.exists():
            cache_file.unlink()
            print(f"✓ Deleted: {cache_file}")
            deleted_count += 1
        else:
            print(f"  Skipped (not found): {cache_file}")

    if deleted_count > 0:
        print(f"\n✓ Deleted {deleted_count} cache file(s)")
    else:
        print("\nNo cache files found to delete")


def main():
    """Main entry point."""
    base_dir = Path("tmp/sentences/tmp_results")
    write_dir = Path("/tmp")
    output_file = write_dir / "merged_sentences.jsonl.gz"
    att_output_file = write_dir / "sentence_att_results.jsonl.gz"

    # Check if both cached files exist
    if output_file.exists() and att_output_file.exists():
        print(f"✓ Both merged file and ATT results already exist:")
        print(f"  - {output_file}")
        print(f"  - {att_output_file}")
        print("Skipping creation. Delete these files to rebuild.")
        return

    # Load or rebuild merged data
    if output_file.exists():
        print(f"✓ Merged file already exists: {output_file}")
        print("Loading from cache...\n")
        merged_data = load_merged_from_cache(output_file)
    else:
        merged_data = rebuild_merged_data(base_dir, output_file)
        if not merged_data:
            return

    # Load or compute SBERT embeddings
    print(f"\n{'='*80}")
    print(f"LOADING SBERT EMBEDDINGS")
    print(f"{'='*80}\n")

    sbert_embeddings_file = write_dir / "sentence_sbert_embeddings.jsonl.gz"

    if not sbert_embeddings_file.exists():
        print(f"SBERT embeddings not found at {sbert_embeddings_file}")
        print("Computing SBERT embeddings...")
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "scripts/R2/scm/compute_sbert_embeddings.py"],
            capture_output=False
        )

        if result.returncode != 0:
            print(f"\n⚠ Warning: compute_sbert_embeddings.py exited with code {result.returncode}")
            print("Proceeding without embeddings...")
            sent_id2embed = None
        else:
            # Load the newly computed embeddings
            sent_id2embed = {}
            with gzip.open(sbert_embeddings_file, 'rt') as f:
                for line in f:
                    record = json.loads(line)
                    sent_id2embed[record['id']] = record['embedding']
            print(f"✓ Loaded {len(sent_id2embed):,} SBERT embeddings\n")
    else:
        print(f"Loading SBERT embeddings from {sbert_embeddings_file}")
        sent_id2embed = {}
        with gzip.open(sbert_embeddings_file, 'rt') as f:
            for line in f:
                record = json.loads(line)
                sent_id2embed[record['id']] = record['embedding']
        print(f"✓ Loaded {len(sent_id2embed):,} SBERT embeddings\n")

    # Compute ATT for each sentence (with SBERT embeddings)
    att_results = analyze_sentences(merged_data, sent_id2embed)

    # Save ATT results
    print(f"\n{'='*80}")
    print(f"SAVING ATT RESULTS")
    print(f"{'='*80}")
    print(f"Writing ATT results to: {att_output_file}")

    with gzip.open(att_output_file, 'wt') as f:
        for sent_id, att_data in tqdm(att_results.items(), desc="Writing ATT results"):
            record = {
                'id': sent_id,
                **att_data
            }
            f.write(json.dumps(record) + '\n')

    print(f"✓ Wrote {len(att_results):,} ATT results")

    # Run prediction analysis
    print(f"\n{'='*80}")
    print(f"RUNNING ATT PREDICTION ANALYSIS")
    print(f"{'='*80}\n")

    import subprocess
    import sys
    result = subprocess.run(
        [sys.executable, "scripts/R2/scm/predict_att.py"],
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\n⚠ Warning: predict_att.py exited with code {result.returncode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze sentence-level data and compute ATT")
    parser.add_argument(
        "--clean-cache",
        action="store_true",
        help="Delete cached files before running"
    )

    args = parser.parse_args()

    if args.clean_cache:
        clean_cache()

    main()

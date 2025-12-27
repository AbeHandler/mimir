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


def analyze_sentences(merged_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute ATT (Average Treatment Effect) for each sentence.
    ATT = loss_run1 - loss_run4

    Args:
        merged_data: Merged sentence data from all sources

    Returns:
        Dictionary mapping sentence IDs to analysis results:
        {
            'sentence_id': {
                'loss_run1': float,
                'loss_run4': float,
                'att': float
            }
        }
    """
    print(f"\n{'='*80}")
    print(f"COMPUTING SENTENCE-LEVEL ATT")
    print(f"{'='*80}\n")

    results = {}

    for sent_id, sources in tqdm(merged_data.items(), desc="Analyzing sentences"):
        # Get data for run1 (treated) and run4 (control)
        run1_data = sources.get('pair1_treated_run1')
        run4_data = sources.get('pair2_control_run4')

        if not run1_data or not run4_data:
            continue

        # Compute losses
        loss_run1 = compute_sentence_loss(run1_data.get('tokens', []))
        loss_run4 = compute_sentence_loss(run4_data.get('tokens', []))

        # Compute ATT = treated - control
        if not np.isnan(loss_run1) and not np.isnan(loss_run4):
            att = loss_run1 - loss_run4
        else:
            att = np.nan

        results[sent_id] = {
            'loss_run1': loss_run1,
            'loss_run4': loss_run4,
            'att': att,
            'text': run1_data.get('text', ''),
            'doc_id': run1_data.get('doc_id', '')
        }

    # Print summary statistics
    valid_atts = [r['att'] for r in results.values() if not np.isnan(r['att'])]

    if valid_atts:
        print(f"\nATT Statistics:")
        print(f"  Total sentences: {len(results):,}")
        print(f"  Valid ATTs: {len(valid_atts):,}")
        print(f"  Mean ATT: {np.mean(valid_atts):.6f}")
        print(f"  Median ATT: {np.median(valid_atts):.6f}")
        print(f"  Std ATT: {np.std(valid_atts):.6f}")
        print(f"  Min ATT: {np.min(valid_atts):.6f}")
        print(f"  Max ATT: {np.max(valid_atts):.6f}")

        # Show some examples
        sorted_by_att = sorted(results.items(), key=lambda x: x[1]['att'] if not np.isnan(x[1]['att']) else 0, reverse=True)

        print(f"\nTop 3 sentences with highest ATT (most affected by treatment):")
        for i, (sent_id, data) in enumerate(sorted_by_att[:3], 1):
            print(f"\n{i}. ID: {sent_id}")
            print(f"   ATT: {data['att']:.6f} (loss_run1: {data['loss_run1']:.6f}, loss_run4: {data['loss_run4']:.6f})")
            print(f"   Text: {data['text'][:100]}...")

        print(f"\nTop 3 sentences with lowest ATT (least affected by treatment):")
        for i, (sent_id, data) in enumerate(sorted_by_att[-3:], 1):
            print(f"\n{i}. ID: {sent_id}")
            print(f"   ATT: {data['att']:.6f} (loss_run1: {data['loss_run1']:.6f}, loss_run4: {data['loss_run4']:.6f})")
            print(f"   Text: {data['text'][:100]}...")

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


def main():
    """Main entry point."""
    base_dir = Path("tmp/sentences/tmp_results")
    write_dir = Path("/tmp")
    output_file = write_dir / "merged_sentences.jsonl.gz"

    # Check if cached merged file exists - skip creation if it does
    if output_file.exists():
        print(f"✓ Merged file already exists: {output_file}")
        print("Skipping creation. Delete this file to rebuild.")
        return

    # Cache doesn't exist, rebuild it
    print(f"No cached file found at {output_file}")
    print(f"Rebuilding from source files...\n")

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return

    print(f"Loading sentence-level data from {base_dir}\n")

    # Load and merge data
    merged_data = load_sentence_level_data(base_dir)

    # Analyze the merged data
    analyze_merged_data(merged_data)

    # Compute ATT for each sentence
    att_results = analyze_sentences(merged_data)

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

    # Save ATT results
    att_output_file = write_dir / "sentence_att_results.jsonl.gz"
    print(f"\nWriting ATT results to: {att_output_file}")

    with gzip.open(att_output_file, 'wt') as f:
        for sent_id, att_data in tqdm(att_results.items(), desc="Writing ATT results"):
            record = {
                'id': sent_id,
                **att_data
            }
            f.write(json.dumps(record) + '\n')

    print(f"✓ Wrote {len(att_results):,} ATT results")


if __name__ == "__main__":
    main()

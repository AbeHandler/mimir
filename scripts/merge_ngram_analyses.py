#!/usr/bin/env python3
"""
Merge N-gram Analyses from Two Models

Takes two ngrams_analysis.json files (e.g., from blocksbin and noblocksbin models),
verifies they have matching n-grams for each document, and outputs a merged JSONL file
with n-gram text and log probabilities from both models.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


def load_ngrams(json_path: str) -> dict:
    """Load n-grams analysis JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def verify_and_merge(
    ngrams1: dict,
    ngrams2: dict,
    model1_name: str,
    model2_name: str
) -> List[dict]:
    """
    Verify that both analyses have matching n-grams and merge them.

    Args:
        ngrams1: N-grams from first model
        ngrams2: N-grams from second model
        model1_name: Name for first model (e.g., 'blocksbin')
        model2_name: Name for second model (e.g., 'noblocksbin')

    Returns:
        List of merged records ready for JSONL output

    Raises:
        ValueError: If n-grams don't match between models
    """
    merged_records = []

    for membership in ['member', 'nonmember']:
        print(f"\nProcessing {membership}s...")

        ngrams1_list = ngrams1[membership]
        ngrams2_list = ngrams2[membership]

        if len(ngrams1_list) != len(ngrams2_list):
            raise ValueError(
                f"Mismatch in {membership} n-gram counts: "
                f"{model1_name}={len(ngrams1_list)}, {model2_name}={len(ngrams2_list)}"
            )

        # Group by doc_id for easier comparison
        ngrams1_by_doc = defaultdict(list)
        ngrams2_by_doc = defaultdict(list)

        for ng in ngrams1_list:
            ngrams1_by_doc[ng['doc_id']].append(ng)

        for ng in ngrams2_list:
            ngrams2_by_doc[ng['doc_id']].append(ng)

        # Verify same doc_ids
        docs1 = set(ngrams1_by_doc.keys())
        docs2 = set(ngrams2_by_doc.keys())

        if docs1 != docs2:
            raise ValueError(
                f"Mismatch in {membership} doc_ids:\n"
                f"  Only in {model1_name}: {docs1 - docs2}\n"
                f"  Only in {model2_name}: {docs2 - docs1}"
            )

        # Verify each doc has same n-grams
        for doc_id in sorted(docs1):
            ng1_list = ngrams1_by_doc[doc_id]
            ng2_list = ngrams2_by_doc[doc_id]

            if len(ng1_list) != len(ng2_list):
                raise ValueError(
                    f"Doc {doc_id} ({membership}) has different n-gram counts: "
                    f"{model1_name}={len(ng1_list)}, {model2_name}={len(ng2_list)}"
                )

            # Check that n-grams match (by text and token_ids)
            for ng1, ng2 in zip(ng1_list, ng2_list):
                if ng1['ngram_text'] != ng2['ngram_text']:
                    raise ValueError(
                        f"Doc {doc_id} ({membership}) has mismatched n-gram texts:\n"
                        f"  {model1_name}: {ng1['ngram_text'][:100]}\n"
                        f"  {model2_name}: {ng2['ngram_text'][:100]}"
                    )

                if ng1['token_ids'] != ng2['token_ids']:
                    raise ValueError(
                        f"Doc {doc_id} ({membership}) has mismatched token_ids for n-gram:\n"
                        f"  Text: {ng1['ngram_text'][:100]}\n"
                        f"  {model1_name}: {ng1['token_ids']}\n"
                        f"  {model2_name}: {ng2['token_ids']}"
                    )

                # Merge the matching n-grams
                merged_record = {
                    'membership': membership,
                    'doc_id': doc_id,
                    'ngram_text': ng1['ngram_text'],
                    'token_ids': ng1['token_ids'],
                    f'{model1_name}_log_probs': ng1['log_probs'],
                    f'{model1_name}_mean_log_prob': ng1['mean_log_prob'],
                    f'{model1_name}_min_log_prob': ng1['min_log_prob'],
                    f'{model1_name}_max_log_prob': ng1['max_log_prob'],
                    f'{model2_name}_log_probs': ng2['log_probs'],
                    f'{model2_name}_mean_log_prob': ng2['mean_log_prob'],
                    f'{model2_name}_min_log_prob': ng2['min_log_prob'],
                    f'{model2_name}_max_log_prob': ng2['max_log_prob'],
                }
                merged_records.append(merged_record)

        print(f"  ✓ Verified and merged {len(ngrams1_list)} n-grams from {len(docs1)} documents")

    return merged_records


def main():
    parser = argparse.ArgumentParser(
        description='Merge n-gram analyses from two models'
    )
    parser.add_argument(
        'ngrams1_path',
        type=str,
        help='Path to first ngrams_analysis.json'
    )
    parser.add_argument(
        'ngrams2_path',
        type=str,
        help='Path to second ngrams_analysis.json'
    )
    parser.add_argument(
        '--model1-name',
        type=str,
        default='blocksbin',
        help='Name for first model (default: blocksbin)'
    )
    parser.add_argument(
        '--model2-name',
        type=str,
        default='noblocksbin',
        help='Name for second model (default: noblocksbin)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSONL path (default: merged_ngrams.jsonl in current directory)'
    )

    args = parser.parse_args()

    # Default output path
    if args.output is None:
        args.output = 'merged_ngrams.jsonl'

    print(f"Loading {args.model1_name} n-grams from: {args.ngrams1_path}")
    ngrams1 = load_ngrams(args.ngrams1_path)

    print(f"Loading {args.model2_name} n-grams from: {args.ngrams2_path}")
    ngrams2 = load_ngrams(args.ngrams2_path)

    print("\nVerifying and merging n-grams...")
    merged_records = verify_and_merge(
        ngrams1,
        ngrams2,
        args.model1_name,
        args.model2_name
    )

    print(f"\nWriting merged records to: {args.output}")
    with open(args.output, 'w') as f:
        for record in merged_records:
            f.write(json.dumps(record) + '\n')

    print(f"\n✓ Successfully wrote {len(merged_records)} merged n-gram records")

    # Print summary statistics
    members = sum(1 for r in merged_records if r['membership'] == 'member')
    nonmembers = sum(1 for r in merged_records if r['membership'] == 'nonmember')
    print(f"\nSummary:")
    print(f"  Members: {members} n-grams")
    print(f"  Non-members: {nonmembers} n-grams")
    print(f"  Total: {len(merged_records)} n-grams")


if __name__ == '__main__':
    main()

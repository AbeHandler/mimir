#!/usr/bin/env python3
"""
N-gram Token Probability Analyzer

Extracts n-grams from documents and their corresponding per-token log probabilities.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def load_results(json_path: str) -> dict:
    """Load the results JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def extract_ngrams_with_probs(
    tokenizer,
    text: str,
    token_probs: List[float],
    n: int = 13
) -> List[Tuple[List[int], List[float], str]]:
    """
    Extract n-grams and their corresponding token log probabilities.

    Args:
        tokenizer: HuggingFace tokenizer
        text: Input text
        token_probs: List of per-token log probabilities
        n: N-gram size (default: 13)

    Returns:
        List of tuples: (token_ids, log_probs, decoded_text)
    """
    # Tokenize the text
    tokens = tokenizer.encode(text, return_tensors="pt")[0].tolist()

    # Note: token_probs has len(tokens) - 1 due to autoregressive shift
    # token_probs[i] is the log prob of token[i+1]

    ngrams = []

    # Slide window over tokens
    for i in range(len(tokens) - n + 1):
        ngram_tokens = tokens[i:i+n]

        # Get corresponding log probs
        # For n-gram at position i, we want probs for tokens i+1 to i+n
        # which are at indices i to i+n-1 in token_probs
        if i + n - 1 <= len(token_probs):
            ngram_probs = token_probs[i:i+n]
        else:
            # Edge case: not enough probs for last n-gram
            continue

        # Decode the n-gram
        ngram_text = tokenizer.decode(ngram_tokens)

        ngrams.append((ngram_tokens, ngram_probs, ngram_text))

    return ngrams


def analyze_results(
    results_path: str,
    model_name: str,
    n: int = 13,
    output_path: str = None,
    max_docs: int = None
) -> None:
    """
    Analyze n-grams and their probabilities from results file.

    Args:
        results_path: Path to loss_results.json
        model_name: Model name for tokenizer
        n: N-gram size
        output_path: Optional output file path
        max_docs: Maximum number of documents to process
    """
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    print(f"Loading results from: {results_path}")
    data = load_results(results_path)

    # Process both members and nonmembers
    all_ngrams = defaultdict(list)

    for membership in ['member', 'nonmember']:
        print(f"\nProcessing {membership}s...")

        ids = list(data['id_to_token_probs'][membership].keys())
        if max_docs:
            ids = ids[:max_docs]

        for doc_id in ids:
            text = data['id_to_text'][membership][doc_id]
            if isinstance(text, list):
                text = text[0]  # Handle substring case

            token_probs_list = data['id_to_token_probs'][membership][doc_id]
            # Take first substring (we assert full_doc=False)
            token_probs = token_probs_list[0]

            # Extract n-grams
            ngrams = extract_ngrams_with_probs(tokenizer, text, token_probs, n)

            for token_ids, probs, text in ngrams:
                all_ngrams[membership].append({
                    'doc_id': doc_id,
                    'ngram_text': text,
                    'token_ids': token_ids,
                    'log_probs': probs,
                    'mean_log_prob': sum(probs) / len(probs),
                    'min_log_prob': min(probs),
                    'max_log_prob': max(probs)
                })

        print(f"  Extracted {len(all_ngrams[membership])} {n}-grams from {len(ids)} documents")

    # Save or print results
    if output_path:
        print(f"\nSaving results to: {output_path}")
        with open(output_path, 'w') as f:
            json.dump(all_ngrams, f, indent=2)
        print(f"Saved {sum(len(v) for v in all_ngrams.values())} total n-grams")
    else:
        # Print sample
        print("\n" + "="*80)
        print(f"Sample {n}-grams from first document:")
        print("="*80)

        for membership in ['member', 'nonmember']:
            if all_ngrams[membership]:
                print(f"\n{membership.upper()}:")
                for i, ngram in enumerate(all_ngrams[membership][:3]):
                    print(f"\n  N-gram {i+1}:")
                    print(f"    Text: {ngram['ngram_text'][:100]}")
                    print(f"    Token IDs: {ngram['token_ids']}")
                    print(f"    Log probs: {[f'{p:.4f}' for p in ngram['log_probs']]}")
                    print(f"    Mean log prob: {ngram['mean_log_prob']:.4f}")
                break  # Only show sample from first membership type


def main():
    parser = argparse.ArgumentParser(
        description='Extract n-grams and their token log probabilities from results'
    )
    parser.add_argument(
        'results_path',
        type=str,
        help='Path to loss_results.json file'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Model name for tokenizer (e.g., dobolyilab/blockbench-blocksbin)'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=13,
        help='N-gram size (default: 13)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (optional)'
    )
    parser.add_argument(
        '--max-docs',
        type=int,
        help='Maximum number of documents to process (optional)'
    )

    args = parser.parse_args()

    analyze_results(
        args.results_path,
        args.model,
        args.n,
        args.output,
        args.max_docs
    )


if __name__ == '__main__':
    main()

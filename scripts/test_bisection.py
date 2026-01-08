#!/usr/bin/env python3
"""
Test bisection attack accuracy against oracle ground truth.

This script validates that the bisection attack can accurately recover
logprob differences with different budget constraints.

Usage:
    python scripts/test_bisection.py
    python scripts/test_bisection.py --model gpt2-medium
    python scripts/test_bisection.py --max-queries 5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import transformers
from src.bisection import (
    recover_token_logprob_difference,
    get_oracle_logprobs
)


def test_bisection_with_budgets(
    model_name: str = "gpt2",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    precision: float = 0.01
):
    """
    Test bisection attack accuracy against oracle with different query budgets.

    Args:
        model_name: HuggingFace model name
        device: Device to run on
        precision: Bisection precision
    """
    print("=" * 70)
    print("Bisection Attack Test - Budget Comparison")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Precision: {precision}")
    print()

    # Load model
    print("Loading model...")
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    # Test cases
    test_cases = [
        ('The classic first program prints "Hello, ', 'World'),
        ('The capital of France is ', 'Paris'),
        ('2 + 2 = ', '4'),
        ('The quick brown ', 'fox'),
    ]

    # Query budgets to test
    budgets = [1, 5, 10]

    # Store results for each budget
    budget_results = {budget: [] for budget in budgets}

    for prompt, target in test_cases:
        print("-" * 70)
        print(f"Prompt: '{prompt}'")
        print(f"Target token: '{target}'")
        print()

        # Get oracle (ground truth)
        oracle_logprob, top_token, top_logprob = get_oracle_logprobs(
            model, tokenizer, prompt, target, device
        )

        print(f"Oracle:")
        print(f"  Top token: '{top_token}' (logprob: {top_logprob:.4f})")
        print(f"  Target '{target}' relative logprob: {oracle_logprob:.4f}")
        print()

        # Test each budget
        for budget in budgets:
            recovered_logprob, queries, _ = recover_token_logprob_difference(
                model, tokenizer, prompt, target,
                device=device,
                precision=precision,
                max_queries=budget
            )

            if recovered_logprob is None:
                error = float('inf')
                status = "FAILED"
            else:
                error = abs(recovered_logprob - oracle_logprob)
                if error < 0.05:
                    status = "✓ EXCELLENT"
                elif error < 0.2:
                    status = "✓ GOOD"
                else:
                    status = "✗ POOR"

            budget_results[budget].append({
                'prompt': prompt,
                'target': target,
                'oracle': oracle_logprob,
                'recovered': recovered_logprob,
                'queries': queries,
                'error': error,
                'status': status
            })

            print(f"  Budget={budget:2d} queries: ", end="")
            if recovered_logprob is None:
                print(f"FAILED (no upper bound found)")
            else:
                print(f"recovered={recovered_logprob:7.4f}, error={error:6.4f}, {status}")

        print()

    # Summary table
    print("=" * 70)
    print("SUMMARY - Error by Query Budget")
    print("=" * 70)
    print(f"{'Budget':<10} {'Success Rate':<15} {'Avg Error':<12} {'Min Error':<12} {'Max Error':<12}")
    print("-" * 70)

    for budget in budgets:
        results = budget_results[budget]
        valid_results = [r for r in results if r['recovered'] is not None]

        if valid_results:
            success_rate = f"{len(valid_results)}/{len(results)}"
            avg_error = sum(r['error'] for r in valid_results) / len(valid_results)
            min_error = min(r['error'] for r in valid_results)
            max_error = max(r['error'] for r in valid_results)

            print(f"{budget:<10} {success_rate:<15} {avg_error:<12.4f} {min_error:<12.4f} {max_error:<12.4f}")
        else:
            print(f"{budget:<10} {'0/' + str(len(results)):<15} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    for budget in budgets:
        results = budget_results[budget]
        valid_results = [r for r in results if r['recovered'] is not None]

        if valid_results:
            avg_error = sum(r['error'] for r in valid_results) / len(valid_results)
            print(f"Budget {budget:2d}: Avg error = {avg_error:.4f} " +
                  f"({len(valid_results)}/{len(results)} successful)")
        else:
            print(f"Budget {budget:2d}: All recoveries failed")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test bisection attack accuracy with multiple query budgets"
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        help='HuggingFace model name (default: gpt2)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run on (default: cuda if available, else cpu)'
    )
    parser.add_argument(
        '--precision',
        type=float,
        default=0.01,
        help='Bisection precision (default: 0.01)'
    )

    args = parser.parse_args()

    test_bisection_with_budgets(
        model_name=args.model,
        device=args.device,
        precision=args.precision
    )


if __name__ == "__main__":
    main()

"""
Debug script to reload pickled inputs from loss_bisection.py and recreate errors.

Usage:
    python debug_bisection.py

This script:
1. Loads debug data from /tmp/mimirdebug.p
2. Displays the context that was being processed when the error occurred
3. Provides utilities to investigate and fix the issue
"""

import pickle
import numpy as np


def load_debug_data():
    """Load the debug data saved during the failed run."""
    with open('/tmp/mimirdebug.p', 'rb') as f:
        data = pickle.load(f)
    return data


def display_debug_info(data):
    """Display information about the failed bisection call."""
    print("=" * 80)
    print("DEBUG INFORMATION")
    print("=" * 80)
    print(f"\nModel: {data['model_name']}")
    print(f"Device: {data['device']}")
    print(f"Vocab size: {data['vocab_size']}")
    print(f"Token index: {data['token_index']}")
    print(f"Target token ID: {data['target_token_id']}")
    print(f"Precision: {data['precision']}")
    print(f"Max queries: {data['max_queries']}")
    print(f"\nContext text (first 200 chars):\n{data['context_text'][:200]}")
    print(f"\nFull document (first 200 chars):\n{data['document'][:200]}")
    print(f"\nContext tokens: {data['context_tokens']}")
    print(f"All tokens shape: {data['all_tokens'].shape if isinstance(data['all_tokens'], np.ndarray) else len(data['all_tokens'])}")
    print("=" * 80)


def recreate_bisection_call(model, tokenizer, data):
    """
    Recreate the exact bisection call that failed.

    Args:
        model: The model to use (you'll need to load this separately)
        tokenizer: The tokenizer (you'll need to load this separately)
        data: Debug data from the pickle file

    Returns:
        The result of recover_token_logprob_difference
    """
    import torch
    from src.bisection import recover_token_logprob_difference

    device = torch.device(data['device'] if torch.cuda.is_available() and 'cuda' in data['device'] else 'cpu')

    print("\nAttempting to recreate bisection call...")
    print(f"Prompt: {data['context_text'][:100]}...")
    print(f"Target token ID: {data['target_token_id']}")

    try:
        result = recover_token_logprob_difference(
            model=model,
            tokenizer=tokenizer,
            prompt=data['context_text'],
            target_token_id=data['target_token_id'],
            device=device,
            precision=data['precision'],
            max_queries=data['max_queries']
        )
        print("\nSUCCESS! Bisection call completed.")
        print(f"Result: {result}")
        return result
    except Exception as e:
        print(f"\nERROR REPRODUCED: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Main debug function."""
    # Load debug data
    print("Loading debug data from /tmp/mimirdebug.p...")
    data = load_debug_data()

    # Display information
    display_debug_info(data)

    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Load your model and tokenizer")
    print("2. Call recreate_bisection_call(model, tokenizer, data) to reproduce the error")
    print("\nExample:")
    print("    from mimir.models import Model")
    print("    from mimir.config import ExperimentConfig")
    print("    ")
    print("    # Load your model")
    print("    config = ExperimentConfig(...)")
    print("    model_wrapper = Model(config, ...)")
    print("    ")
    print("    # Recreate the call")
    print("    recreate_bisection_call(")
    print("        model_wrapper.model,")
    print("        model_wrapper.tokenizer,")
    print("        data")
    print("    )")
    print("=" * 80)

    return data


if __name__ == '__main__':
    data = main()

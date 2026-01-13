"""
Bisection attack for recovering logprob differences using logit bias.

This module provides utilities for using logit bias to recover the difference
between a top token's logprob and any target token's logprob through binary search.

The attack works by:
1. Query model with no bias to get the default (top) token
2. Apply increasing bias to target token until it becomes the top token
3. Binary search to find the exact bias threshold where target token wins
4. The threshold approximates the logit difference between top and target tokens
"""

import torch
import numpy as np
from typing import Optional, Tuple, Callable
from transformers import LogitsProcessor


class LogitBiasProcessor(LogitsProcessor):
    """
    Logits processor that adds a constant bias to specific token(s).

    This is used to simulate the logit_bias parameter available in API-based models.
    """

    def __init__(self, token_id: int, bias: float):
        """
        Initialize the logit bias processor.

        Args:
            token_id: The token ID to apply bias to
            bias: The bias value to add to the token's logit
        """
        self.token_id = token_id
        self.bias = bias

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Apply bias to the specified token's logit.

        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            scores: Logit scores of shape [batch_size, vocab_size]

        Returns:
            Modified scores with bias applied
        """
        scores[:, self.token_id] += self.bias
        return scores


def query_with_bias(
    model,
    tokenizer,
    prompt: str,
    target_token_id: int,
    bias: float,
    device: str = "cuda"
) -> str:
    """
    Query a HuggingFace model with logit bias applied to a target token.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt string
        target_token_id: Token ID to apply bias to
        bias: Bias value to add to target token's logit
        device: Device to run on ('cuda' or 'cpu')

    Returns:
        The generated token as a string
    """
    # Get model's maximum sequence length
    max_length = getattr(model.config, 'max_position_embeddings', None)
    if max_length is None:
        max_length = getattr(model.config, 'n_positions', 1024)

    # Tokenize with truncation from the left (keep most recent context)
    # Reserve 1 token for generation
    # Set truncation side on tokenizer object (not as parameter)
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = 'left'

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length - 1,
        truncation=True
    ).to(device)

    # Restore original truncation side
    tokenizer.truncation_side = original_truncation_side

    # Always create logits processor (bias=0 is a no-op)
    logits_processor = [LogitBiasProcessor(target_token_id, bias)]

    # Generate one token with greedy decoding (temperature=0 equivalent)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,  # Greedy decoding
            logits_processor=logits_processor,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Extract only the new token
    new_token_id = outputs[0, -1].item()
    generated_token = tokenizer.decode([new_token_id])

    return generated_token


def find_upper_bound(
    query_fn: Callable[[int, float], str],
    target_token: str,
    target_token_id: int,
    initial_upper: float = 10.0,
    max_upper: float = 200.0
) -> Optional[float]:
    """
    Find an upper bound where the target token wins.

    Args:
        query_fn: Function that takes (token_id, bias) and returns generated token
        target_token: The target token string we want to win
        target_token_id: The target token ID
        initial_upper: Starting upper bound to try
        max_upper: Maximum upper bound to try

    Returns:
        Upper bound value where target wins, or None if not found
    """
    upper = initial_upper

    while upper <= max_upper:
        result = query_fn(target_token_id, upper)
        if result == target_token:
            return upper
        upper *= 2

    return None


def bisect_logprob_difference(
    query_fn: Callable[[int, float], str],
    target_token: str,
    target_token_id: int,
    precision: float = 0.01,
    initial_upper: float = 10.0,
    max_upper: float = 200.0,
    max_iterations: Optional[int] = None,
    max_queries: Optional[int] = None
) -> Tuple[Optional[float], int]:
    """
    Use bisection to find the logprob of target token relative to top token.

    Args:
        query_fn: Function that takes (token_id, bias) and returns generated token
        target_token: The target token string we want to match
        target_token_id: The target token ID
        precision: Desired precision for bisection (default 0.01)
        initial_upper: Initial upper bound to search from
        max_upper: Maximum upper bound before giving up
        max_iterations: Maximum bisection iterations only (None for unlimited)
        max_queries: Maximum total API calls including upper bound search (None for unlimited)

    Returns:
        Tuple of (relative_logprob, num_queries) where:
            - relative_logprob is logprob(target) - logprob(top), a negative value
            - num_queries is the total number of model queries made

    Note:
        Returns negative value as per Algorithm 1 in the paper.
        If top token has logprob -2.0 and target needs bias of 5.0 to win,
        then target's relative logprob is -5.0.

        If max_queries is set, the function will stop early and return the best
        estimate available within the budget.
    """
    num_queries = 0

    # Find upper bound
    upper = initial_upper
    while upper <= max_upper:
        if max_queries is not None and num_queries >= max_queries:
            # Budget exhausted during upper bound search
            return None, num_queries

        result = query_fn(target_token_id, upper)
        num_queries += 1

        if result == target_token:
            break
        upper *= 2
    else:
        # Failed to find upper bound
        return None, num_queries

    # Bisection
    lower = 0.0
    iterations = 0

    while upper - lower > precision:
        # Check budget constraints
        if max_iterations is not None and iterations >= max_iterations:
            break
        if max_queries is not None and num_queries >= max_queries:
            break

        mid = (lower + upper) / 2
        result = query_fn(target_token_id, mid)
        num_queries += 1
        iterations += 1

        if result == target_token:
            upper = mid
        else:
            lower = mid

    threshold = (lower + upper) / 2
    return -threshold, num_queries


def recover_token_logprob_difference(
    model,
    tokenizer,
    prompt: str,
    target_token: str = None,
    target_token_id: int = None,
    device: str = "cuda",
    precision: float = 0.01,
    max_iterations: Optional[int] = None,
    max_queries: Optional[int] = None
) -> Tuple[Optional[float], int, str]:
    """
    Recover the relative logprob of target token using bisection attack.

    This is a convenience function that wraps the bisection logic for HuggingFace models.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt string
        target_token: Target token string to recover logprob for (if target_token_id not provided)
        target_token_id: Target token ID to recover logprob for (preferred, avoids encoding issues)
        device: Device to run on
        precision: Desired precision for bisection
        max_iterations: Maximum bisection iterations only (None for unlimited)
        max_queries: Maximum total API calls (None for unlimited)

    Returns:
        Tuple of (relative_logprob, num_queries, top_token) where:
            - relative_logprob: logprob(target) - logprob(top), a negative value
            - num_queries: total number of model queries made
            - top_token: the default top token without bias

    Example:
        >>> # Unlimited budget
        >>> logprob, queries, top = recover_token_logprob_difference(
        ...     model, tokenizer, "Hello", target_token="World"
        ... )
        >>> print(f"Target logprob: {logprob}, Queries: {queries}")
        Target logprob: -5.23, Queries: 12

        >>> # Limited budget: max 5 API calls, using token ID directly
        >>> logprob, queries, top = recover_token_logprob_difference(
        ...     model, tokenizer, "Hello", target_token_id=1234, max_queries=5
        ... )
        >>> print(f"Target logprob: {logprob}, Queries: {queries}")
        Target logprob: -5.10, Queries: 5
    """
    # Get target token ID - prefer direct ID to avoid encoding issues
    if target_token_id is None:
        if target_token is None:
            raise ValueError("Must provide either target_token or target_token_id")
        encoded = tokenizer.encode(target_token, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"target_token '{target_token}' encodes to {len(encoded)} tokens, expected 1")
        target_token_id = encoded[0]

    # Validate token ID is within vocabulary bounds
    # Use model vocab size, not tokenizer, since model embeddings may not be resized
    vocab_size = model.config.vocab_size
    if target_token_id < 0 or target_token_id >= vocab_size:
        raise ValueError(f"target_token_id {target_token_id} is out of bounds for model vocabulary size {vocab_size}")

    # Decode target token for comparison (single token should decode cleanly)
    target_token_str = tokenizer.decode([target_token_id])

    # Get default top token (no bias) - counts toward budget
    top_token = query_with_bias(model, tokenizer, prompt, target_token_id, 0.0, device)
    queries_used = 1

    # Adjust budget for remaining queries
    remaining_budget = None if max_queries is None else max_queries - queries_used

    # Create query function
    def query_fn(token_id: int, bias: float) -> str:
        return query_with_bias(model, tokenizer, prompt, token_id, bias, device)

    # Run bisection
    threshold, bisection_queries = bisect_logprob_difference(
        query_fn,
        target_token_str,
        target_token_id,
        precision=precision,
        max_iterations=max_iterations,
        max_queries=remaining_budget
    )

    # Total queries = initial query + bisection queries
    total_queries = queries_used + bisection_queries
    return threshold, total_queries, top_token


def get_oracle_logprobs(
    model,
    tokenizer,
    prompt: str,
    target_token: str,
    device: str = "cuda"
) -> Tuple[float, str, float]:
    """
    Oracle function to get ground truth logprobs (for testing/validation).

    This directly accesses the model's logits to get the true log probabilities
    without using the bisection attack.

    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompt: Input prompt string
        target_token: Target token to get logprob for
        device: Device to run on

    Returns:
        Tuple of (relative_logprob, top_token, top_logprob) where:
            - relative_logprob: logprob(target) - logprob(top), a negative value
            - top_token: the top predicted token
            - top_logprob: the log probability of the top token

    Example:
        >>> rel_logprob, top_token, top_lp = get_oracle_logprobs(
        ...     model, tokenizer, "Hello, ", "World"
        ... )
        >>> print(f"Top: '{top_token}' ({top_lp:.2f})")
        Top: 'there' (-1.23)
        >>> print(f"Target 'World' relative: {rel_logprob:.2f}")
        Target 'World' relative: -5.67
    """
    # Get model's maximum sequence length
    max_length = getattr(model.config, 'max_position_embeddings', None)
    if max_length is None:
        max_length = getattr(model.config, 'n_positions', 1024)

    # Tokenize with truncation from the left (keep most recent context)
    # Set truncation side on tokenizer object (not as parameter)
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = 'left'

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_length,
        truncation=True
    ).to(device)

    # Restore original truncation side
    tokenizer.truncation_side = original_truncation_side

    with torch.no_grad():
        # Get logits for next token
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # [vocab_size]

        # Convert to log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Get top token
        top_token_id = torch.argmax(log_probs).item()
        top_token = tokenizer.decode([top_token_id])
        top_logprob = log_probs[top_token_id].item()

        # Get target token logprob
        target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
        target_logprob = log_probs[target_token_id].item()

        # Relative logprob (negative value)
        relative_logprob = target_logprob - top_logprob

    return relative_logprob, top_token, top_logprob

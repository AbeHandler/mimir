"""
Bisection-based LOSS attack.

Uses logit bias and binary search to recover per-token log probabilities
without requiring the model to expose logprobs directly.

This attack simulates a black-box scenario where:
- The API allows logit bias on individual tokens
- The API returns the generated token (argmax)
- The API does NOT return log probabilities

The attack recovers approximate logprobs using bisection search.
"""

import torch
import numpy as np
from typing import List, Optional
from mimir.attacks.all_attacks import Attack
from mimir.models import Model
from mimir.config import ExperimentConfig
from src.bisection import recover_token_logprob_difference


class LOSSBisectionAttack(Attack):
    """
    LOSS attack using bisection to recover logprobs.

    For each token position in the document:
    1. Use bisection with logit bias to recover the token's relative logprob
    2. Aggregate recovered logprobs to compute document-level score

    This demonstrates that logit bias + argmax queries can leak hidden logprobs.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        target_model: Model,
        queries_per_token: int = 5,
        precision: float = 0.01
    ) -> None:
        """
        Initialize bisection-based LOSS attack.

        Args:
            config: Experiment configuration
            target_model: The model to attack
            queries_per_token: Maximum API calls per token
            precision: Bisection precision (smaller = more accurate but more queries)
        """
        super().__init__(config, target_model, ref_model=None)
        self.queries_per_token = queries_per_token
        self.precision = precision

    @torch.no_grad()
    def _attack(
        self,
        document: str,
        probs: List[float],
        tokens: Optional[np.ndarray] = None,
        **kwargs
    ) -> float:
        """
        Compute LOSS score using bisection to recover logprobs.

        Args:
            document: The text document
            probs: Ground truth log probabilities (NOT USED - for API compatibility)
            tokens: Token IDs (numpy array)
            **kwargs: Additional arguments

        Returns:
            float: Negative mean of recovered log probabilities
        """
        # Tokenize if not provided
        if tokens is None:
            tokenized = self.target_model.tokenizer(document, return_tensors="pt")
            tokens = tokenized.input_ids[0].cpu().numpy()

        # Get device
        device = self.target_model.device

        # Recover logprobs for each token using bisection
        recovered_logprobs = []
        total_queries = 0

        for i in range(len(tokens) - 1):  # -1 because we predict next token
            # Context: tokens[0:i+1]
            # Target: tokens[i+1]
            context_tokens = tokens[:i+1]
            target_token_id = tokens[i+1]

            # Decode context and target
            context_text = self.target_model.tokenizer.decode(context_tokens)
            target_token = self.target_model.tokenizer.decode([target_token_id])

            # Recover relative logprob using bisection
            try:
                relative_logprob, queries_used, _ = recover_token_logprob_difference(
                    model=self.target_model.model,
                    tokenizer=self.target_model.tokenizer,
                    prompt=context_text,
                    target_token=target_token,
                    device=device,
                    precision=self.precision,
                    max_queries=self.queries_per_token
                )

                total_queries += queries_used

                # If bisection failed (budget exhausted), use a penalty value
                if relative_logprob is None:
                    # Large negative value (poor prediction)
                    relative_logprob = -10.0

                recovered_logprobs.append(relative_logprob)

            except Exception as e:
                # If any error occurs, use penalty value
                print(f"Warning: Bisection failed for token {i}: {e}")
                recovered_logprobs.append(-10.0)

        # Compute mean loss
        # Note: recovered_logprobs are already negative (relative to top token)
        # We return -mean to match LOSS convention (higher = worse)
        mean_recovered_logprob = np.mean(recovered_logprobs)

        # Return negative mean (consistent with LOSS attack)
        # Since recovered logprobs are negative, -mean gives positive loss
        return -mean_recovered_logprob

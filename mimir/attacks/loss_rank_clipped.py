"""
    Rank-clipped LOSS attack.

    For each token, if its rank is >= K, replace its log probability with the log probability
    of the token at rank K. This clips the penalty for poorly-predicted tokens.
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from mimir.attacks.all_attacks import Attack
from mimir.models import Model
from mimir.config import ExperimentConfig


class LOSSRankClippedAttack(Attack):

    def __init__(self, config: ExperimentConfig, target_model: Model, k: int = 20) -> None:
        super().__init__(config, target_model, ref_model=None)
        self.k = k

    def _validate_inputs(self, probs: List[float], kwargs: Dict[str, Any]) -> torch.Tensor:
        """
        Validate that required inputs are present and consistent.

        Args:
            probs: List of log probabilities for actual tokens [num_tokens]
            kwargs: Keyword arguments that should contain 'all_probs'

        Returns:
            all_probs: Tensor of shape [num_tokens, vocab_size]

        Raises:
            ValueError: If all_probs is not provided or inputs are inconsistent
        """
        all_probs = kwargs.get('all_probs', None)
        if all_probs is None:
            raise ValueError("all_probs must be provided in kwargs")

        # Validate types and shapes
        # probs: list of floats [num_tokens]
        # all_probs: torch.Tensor [num_tokens, vocab_size]
        if not isinstance(probs, list):
            raise ValueError(f"probs must be a list, got {type(probs)}")

        if not isinstance(all_probs, torch.Tensor):
            raise ValueError(f"all_probs must be a torch.Tensor, got {type(all_probs)}")

        if len(all_probs.shape) != 2:
            raise ValueError(f"all_probs must be 2D [num_tokens, vocab_size], got shape {all_probs.shape}")

        if len(probs) != len(all_probs):
            raise ValueError(f"Length mismatch: {len(probs)} probs vs {len(all_probs)} all_probs")

        return all_probs

    def _get_rank_clipped_log_prob(self, token_log_probs: torch.Tensor, actual_token_log_prob: float) -> float:
        """
        Get rank-clipped log probability for a single token.

        Args:
            token_log_probs: Tensor of log probabilities for all tokens at this position [vocab_size]
            actual_token_log_prob: The actual token's log probability (scalar float)

        Returns:
            float: Clipped log probability
        """
        # Count how many tokens have higher or equal probability (this is the rank)
        # Much faster than full sort - O(n) instead of O(n log n)
        rank = (token_log_probs >= actual_token_log_prob).sum().item()

        # Apply clipping
        if rank < self.k:
            # Use actual token's log probability
            return actual_token_log_prob
        else:
            # Get the k-th largest value using topk (much faster than full sort)
            # topk returns the k largest elements
            kth_log_prob = torch.topk(token_log_probs, self.k, sorted=False)[0].min().item()
            return kth_log_prob

    @torch.no_grad()
    def _attack(self, document: str, probs: List[float], tokens: Optional[np.ndarray] = None, **kwargs) -> float:
        """
        LOSS-score with rank-based clipping.

        For each token:
        - If rank < K: use actual token's log probability
        - If rank >= K: use log probability of token at position K

        Args:
            document: The text document (str)
            probs: List of log probabilities for actual tokens [num_tokens]
            tokens: Optional pre-tokenized tokens (not used, for API compatibility)
            **kwargs: Must include 'all_probs' - Tensor of shape [num_tokens, vocab_size]

        Returns:
            float: Negative mean of clipped log probabilities
        """
        # Validate and extract all_probs: [num_tokens, vocab_size]
        all_probs = self._validate_inputs(probs, kwargs)

        # Process each token position
        clipped_log_probs = []  # Will contain num_tokens floats
        for actual_log_prob, token_all_log_probs in zip(probs, all_probs):
            # actual_log_prob: float
            # token_all_log_probs: [vocab_size]
            clipped_prob = self._get_rank_clipped_log_prob(
                token_all_log_probs,
                actual_log_prob
            )
            clipped_log_probs.append(clipped_prob)

        # Return negative mean (consistent with LOSS attack)
        return -np.mean(clipped_log_probs)

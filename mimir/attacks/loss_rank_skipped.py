"""
    Rank-skipped LOSS attack.

    For each token, if its rank is >= K, skip it entirely (don't include in mean calculation).
    Only compute mean over tokens with rank < K (well-predicted tokens).
"""
import torch
import numpy as np
from typing import List, Dict, Any, Optional
from mimir.attacks.all_attacks import Attack
from mimir.models import Model
from mimir.config import ExperimentConfig


class LOSSRankSkippedAttack(Attack):

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

    def _should_include_token(self, token_log_probs: torch.Tensor, actual_token_log_prob: float) -> bool:
        """
        Determine if a token should be included based on its rank.

        Args:
            token_log_probs: Tensor of log probabilities for all tokens at this position [vocab_size]
            actual_token_log_prob: The actual token's log probability (scalar float)

        Returns:
            bool: True if rank < K (should include), False if rank >= K (should skip)
        """
        # Sort to get ranks (descending order by probability)
        # sorted_log_probs: [vocab_size], sorted from highest to lowest
        # largest to smallest order
        sorted_log_probs = torch.sort(token_log_probs, descending=True)[0]

        # Find where the actual token ranks by comparing its log prob to sorted values
        # rank: int, 0-indexed position in sorted list
        rank = (sorted_log_probs >= actual_token_log_prob).sum().item()

        # Include token only if rank < K (well-predicted tokens)
        return rank < self.k

    @torch.no_grad()
    def _attack(self, document: str, probs: List[float], tokens: Optional[np.ndarray] = None, **kwargs) -> float:
        """
        LOSS-score with rank-based skipping.

        For each token:
        - If rank >= K: skip (don't include in mean)
        - If rank < K: include actual token's log probability

        Args:
            document: The text document (str)
            probs: List of log probabilities for actual tokens [num_tokens]
            tokens: Optional pre-tokenized tokens (not used, for API compatibility)
            **kwargs: Must include 'all_probs' - Tensor of shape [num_tokens, vocab_size]

        Returns:
            float: Negative mean of included log probabilities, or np.inf if no tokens included
        """
        # Validate and extract all_probs: [num_tokens, vocab_size]
        all_probs = self._validate_inputs(probs, kwargs)

        # Process each token position, only including those with rank < K
        included_log_probs = []  # Will contain floats for tokens with rank < K
        for actual_log_prob, token_all_log_probs in zip(probs, all_probs):
            # actual_log_prob: float
            # token_all_log_probs: [vocab_size]
            if self._should_include_token(token_all_log_probs, actual_log_prob):
                included_log_probs.append(actual_log_prob)

        # Handle case where all tokens were filtered out
        if len(included_log_probs) == 0:
            return np.inf  # Infinitely bad score when no tokens qualify

        # Return negative mean (consistent with LOSS attack)
        return -np.mean(included_log_probs)

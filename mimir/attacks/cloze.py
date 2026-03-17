"""
Cloze-style attack based on token prediction with instruction prompting.

For each token position n:
- Prepends instruction: "Fill in the blank with the most likely next token ___"
- Uses tokens [0:n] as the context after instruction
- Predicts token n

Key difference from LOSS attack:
- LOSS: uses sliding window approach (get_ll method)
- CLOZE: uses explicit instruction + prefix [0:n] to predict token n
"""
import torch as ch
import numpy as np
from mimir.attacks.all_attacks import Attack
from mimir.models import Model
from mimir.config import ExperimentConfig


class ClozeAttack(Attack):

    def __init__(self, config: ExperimentConfig, model: Model):
        super().__init__(config, model, ref_model=None)
        # Instruction to prepend before each cloze prediction
        self.instruction_prefix = "Fill in the most likely next token. "
        self.instruction_postfix = " _"
        # Tokenize the instruction parts once and cache them
        self.instruction_prefix_token_ids = self.target_model.tokenizer.encode(
            self.instruction_prefix, add_special_tokens=False
        )
        self.instruction_postfix_token_ids = self.target_model.tokenizer.encode(
            self.instruction_postfix, add_special_tokens=False
        )

    def _get_cloze_logprob(self, prefix_token_ids, target_token_id):
        """
        Get log probability of target token given instruction + prefix + postfix.

        Args:
            prefix_token_ids: List of token IDs for prefix [0:n]
            target_token_id: Token ID to predict at position n

        Returns:
            float: Log probability of target token, or None if error
        """
        try:
            # Build: instruction_prefix + prefix + instruction_postfix
            full_input = self.instruction_prefix_token_ids + prefix_token_ids

            # Convert to tensor
            input_ids = ch.tensor([full_input]).to(self.target_model.device)

            # Get model output (full logits for all vocab)
            with ch.no_grad():
                outputs = self.target_model.model(input_ids)
                logits = outputs.logits[0, -1, :]  # Last token position, all vocab

            # Convert to log probabilities
            log_probs = ch.nn.functional.log_softmax(logits, dim=-1)

            # Get log prob of target token
            target_logprob = log_probs[target_token_id].item()

            return target_logprob

        except Exception:
            return None

    @ch.no_grad()
    def _attack(self, document, probs, tokens=None, **kwargs):
        """
        Cloze-style attack using prefix-based prediction.

        For each token position n:
        - Uses tokens [0:n] as prefix
        - Gets log probability of token n
        - Averages across all positions

        Args:
            document: The text document (str)
            probs: List of log probabilities (not used, for API compatibility)
            tokens: Optional pre-tokenized tokens
            **kwargs: Additional parameters

        Returns:
            float: Negative mean log probability (consistent with LOSS attack convention)
        """
        # Tokenize the document
        if tokens is not None:
            raise ValueError("Expected untokenized")
        else:
            inputs = self.target_model.tokenizer(document, return_tensors="pt", add_special_tokens=False)
            token_ids = inputs["input_ids"][0].tolist()

        if len(token_ids) < 2:
            return 0.0

        # For each position n (starting from 1), use [0:n] to predict token n
        log_probs = []
        for n in range(1, len(token_ids)):
            prefix_tokens = token_ids[0:n]
            target_token = token_ids[n]

            logprob = self._get_cloze_logprob(prefix_tokens, target_token)

            if logprob is not None:
                log_probs.append(logprob)

        if len(log_probs) == 0:
            return 0.0

        # Return negative mean (consistent with LOSS attack)
        return -np.mean(log_probs)

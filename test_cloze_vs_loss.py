#!/usr/bin/env python3
"""
Standalone script to test CLOZE vs LOSS attacks locally on M1.

Usage:
    python test_cloze_vs_loss.py
"""
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

from mimir.config import ExperimentConfig, EnvironmentConfig
from mimir.models import LanguageModel
from mimir.attacks.loss import LOSSAttack
from mimir.attacks.cloze import ClozeAttack


def create_minimal_config():
    """Create a minimal config for testing."""
    env_config = EnvironmentConfig(
        device="mps" if torch.backends.mps.is_available() else "cpu",
        cache_dir="/tmp/cache",
    )

    config = ExperimentConfig(
        experiment_name="test_cloze_vs_loss",
        base_model="gpt2",  # Tiny model for M1
        dataset_member="dummy",  # Not used in this test
        dataset_nonmember="dummy",  # Not used in this test
        env_config=env_config,
        random_seed=42,
    )

    return config


def main():
    print("=" * 60)
    print("Testing CLOZE vs LOSS Attack")
    print("=" * 60)

    # Create config
    config = create_minimal_config()
    print(f"\nDevice: {config.env_config.device}")
    print(f"Model: {config.base_model}")

    # Load model
    print(f"\nLoading model...")
    model = LanguageModel(config)
    model.load()

    # Initialize attacks
    loss_attack = LOSSAttack(config, model)
    cloze_attack = ClozeAttack(config, model)

    # Test examples
    test_examples = [
        "The quick brown fox jumps over the lazy dog.",
        "To be or not to be, that is the question.",
        "During last weekend’s snowy cold spell, when temperatures fell as low as 5 degrees at Denver International Airport, a disturbing number of Denverites were out and about in shorts. Maybe those scantily bedecked Denverites were just prepared for temperatures to rise to what will likely be record highs for March, as highs hit 73 degrees on Tuesday and steadily climb to 85 or higher by the weekend.  The record high for March, set in 1971, is 84 degrees. The National Weather Service predicts the city’s temperature may top that mark multiple times this week. “Thursday, Friday and Saturday, actually all three days, we could set monthly record heat,” said National Weather Service meteorologist Robert Koopmeiners. “So pretty substantial. Temperatures could be as much as 30 degrees above the normals for those dates.” A large ridge of high pressure over the southwestern U.S. is driving the heat wave. “It's real strong for this time of year,” Koopmeiners said. “It has some summer characteristics.”  The heat wave follows a dry, warm winter. Snowpack is at record lows in the mountains, raising serious questions about water usage over the summer. Denver Water is already considering watering restrictions for this summer, such as limiting customers to irrigating on two specific days each week, Axios reported. Dry conditions also lead to higher wildfire risks.  “As far as the drought and heat goes, it's not the greatest,” Koopmeiners said.  Even so, bask while you can. Summer could be brutal. ",
    ]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    loss_scores = []
    cloze_scores = []

    for i, text in enumerate(test_examples, 1):
        print(f"\nExample {i}: {text[:50]}...")

        # Get probabilities (needed for both attacks)
        probs = model.get_probabilities(text)

        # Run LOSS attack (uses sliding window)
        loss_score = loss_attack._attack(text, probs=probs)

        # Run CLOZE attack (prefix-based)
        cloze_score = cloze_attack._attack(text, probs=probs)

        loss_scores.append(loss_score)
        cloze_scores.append(cloze_score)

        print(f"  LOSS score:  {loss_score:.4f}")
        print(f"  CLOZE score: {cloze_score:.4f}")
        print(f"  Difference:  {abs(loss_score - cloze_score):.4f}")

    # Compute correlations
    if len(loss_scores) >= 3:
        pearson_corr, pearson_p = pearsonr(loss_scores, cloze_scores)
        spearman_corr, spearman_p = spearmanr(loss_scores, cloze_scores)

        print("\n" + "=" * 60)
        print("CORRELATION ANALYSIS")
        print("=" * 60)
        print(f"Pearson correlation:  {pearson_corr:.4f} (p={pearson_p:.4f})")
        print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4f})")

    print("\n" + "=" * 60)
    print("EXPLANATION")
    print("=" * 60)
    print("""
LOSS Attack:
- Uses sliding window approach (get_ll method)
- More efficient for long sequences
- May see different context for same token

CLOZE Attack:
- For each position n, uses tokens [0:n] as prefix
- True cloze evaluation (like gpt_oss_tokens.py)
- Every token sees all preceding context
- Can be slower for long sequences

Differences should be small for short sequences but may diverge
for longer sequences due to the sliding window in LOSS.
    """)


if __name__ == "__main__":
    main()

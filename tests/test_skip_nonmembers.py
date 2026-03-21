"""
Test that the skip-nonmembers optimization triggers correctly
based on whether dataset_member == dataset_nonmember.
"""
from pathlib import Path

from mimir.config import ExperimentConfig


FIXTURES = Path(__file__).parent / "fixtures"


def _should_skip_nonmembers(config: ExperimentConfig) -> bool:
    return config.dataset_member == config.dataset_nonmember


class TestSkipNonmembers:
    def test_same_datasets_skips(self):
        """When member == nonmember, we should skip the nonmember pass."""
        config = ExperimentConfig.load(
            FIXTURES.parent.parent / "configs" / "bothbins.blocks.cloze.json",
            drop_extra_fields=False,
        )
        assert _should_skip_nonmembers(config), (
            f"Expected skip when datasets match: "
            f"member={config.dataset_member}, nonmember={config.dataset_nonmember}"
        )

    def test_diff_datasets_does_not_skip(self):
        """When member != nonmember, we must run both passes."""
        config = ExperimentConfig.load(
            FIXTURES / "demo_diff_sets.json",
            drop_extra_fields=False,
        )
        assert not _should_skip_nonmembers(config), (
            f"Expected no skip when datasets differ: "
            f"member={config.dataset_member}, nonmember={config.dataset_nonmember}"
        )

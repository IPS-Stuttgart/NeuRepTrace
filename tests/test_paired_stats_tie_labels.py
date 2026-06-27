from __future__ import annotations

import pandas as pd

from neureptrace.paired_stats import paired_decoder_statistics


def test_paired_decoder_statistics_marks_exact_mean_ties() -> None:
    subject_metrics = pd.DataFrame(
        {
            "decoder": ["a", "a", "b", "b"],
            "subject": ["s1", "s2", "s1", "s2"],
            "effect_accuracy": [0.6, 0.8, 0.6, 0.8],
            "effect_log_loss": [0.5, 0.7, 0.5, 0.7],
        }
    )

    stats = paired_decoder_statistics(
        subject_metrics,
        metrics=("effect_accuracy", "effect_log_loss"),
        n_permutations=10_000,
    )

    assert set(stats["better_decoder_by_mean"]) == {"tie"}
    assert set(stats["mean_difference_a_minus_b"]) == {0.0}

from __future__ import annotations

import pandas as pd

from neureptrace.paired_stats import paired_decoder_statistics


def test_paired_decoder_statistics_labels_exact_mean_ties() -> None:
    rows = []
    for subject in ("sub-01", "sub-02"):
        rows.extend(
            [
                {"emission_mode": "calibrated", "decoder": "lda", "subject": subject, "effect_accuracy": 0.5},
                {"emission_mode": "calibrated", "decoder": "logistic", "subject": subject, "effect_accuracy": 0.5},
            ]
        )

    stats = paired_decoder_statistics(pd.DataFrame(rows), metrics=("effect_accuracy",), n_permutations=10_000)

    assert stats.loc[0, "decoder_a_mean"] == stats.loc[0, "decoder_b_mean"]
    assert stats.loc[0, "better_decoder_by_mean"] == "tie"

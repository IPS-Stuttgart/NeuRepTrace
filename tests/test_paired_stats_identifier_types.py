from __future__ import annotations

import pandas as pd

from neureptrace.paired_stats import paired_decoder_statistics


def test_paired_statistics_preserves_type_distinct_subject_ids() -> None:
    rows = []
    for subject in (1, "1"):
        rows.extend(
            [
                {"decoder": "lda", "subject": subject, "effect_accuracy": 0.4},
                {"decoder": "logistic", "subject": subject, "effect_accuracy": 0.8},
            ]
        )

    statistics = paired_decoder_statistics(
        pd.DataFrame(rows),
        metrics=("effect_accuracy",),
    )

    assert statistics.loc[0, "n_subjects"] == 2
    assert statistics.loc[0, "better_decoder_by_mean"] == "logistic"


def test_paired_statistics_preserves_type_distinct_condition_ids() -> None:
    rows = []
    for condition, lda_accuracy, logistic_accuracy in ((1, 0.4, 0.8), ("1", 0.9, 0.5)):
        for subject in ("sub-01", "sub-02"):
            rows.extend(
                [
                    {
                        "pca_components": condition,
                        "decoder": "lda",
                        "subject": subject,
                        "effect_accuracy": lda_accuracy,
                    },
                    {
                        "pca_components": condition,
                        "decoder": "logistic",
                        "subject": subject,
                        "effect_accuracy": logistic_accuracy,
                    },
                ]
            )

    statistics = paired_decoder_statistics(
        pd.DataFrame(rows),
        metrics=("effect_accuracy",),
    ).set_index("pca_components")

    assert len(statistics) == 2
    assert statistics.loc[1, "better_decoder_by_mean"] == "logistic"
    assert statistics.loc["1", "better_decoder_by_mean"] == "lda"


def test_paired_statistics_preserves_type_distinct_decoder_ids() -> None:
    rows = []
    for subject in ("sub-01", "sub-02"):
        rows.extend(
            [
                {"decoder": 1, "subject": subject, "effect_accuracy": 0.4},
                {"decoder": "1", "subject": subject, "effect_accuracy": 0.8},
            ]
        )

    statistics = paired_decoder_statistics(
        pd.DataFrame(rows),
        metrics=("effect_accuracy",),
    )

    assert statistics.loc[0, "decoder_a"] == 1
    assert statistics.loc[0, "decoder_b"] == "1"
    assert statistics.loc[0, "better_decoder_by_mean"] == "1"

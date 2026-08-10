from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.results import _probability_ece_by_group, aggregate_time_decode_results


def test_exact_ece_supports_signed_probability_class_ids() -> None:
    results = pd.DataFrame(
        {
            "subject": ["s1"],
            "fold": [0],
            "time": [0.1],
            "accuracy": [1.0],
            "log_loss": [0.2],
            "brier": [0.1],
            "ece": [0.9],
            "n_test": [2],
        }
    )
    observations = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "fold": [0, 0],
            "time": [0.1, 0.1],
            "true_label": [-1, 2],
            "prob_class_-1": [0.8, 0.1],
            "prob_class_2": [0.2, 0.9],
        }
    )

    aggregated = aggregate_time_decode_results(results, observations=observations)

    assert aggregated["ece_mean"].tolist() == pytest.approx([0.15])


def test_exact_ece_rejects_duplicate_signed_probability_labels() -> None:
    observations = pd.DataFrame(
        {
            "subject": ["s1"],
            "time": [0.1],
            "true_label": [1],
            "prob_class_1": [0.6],
            "prob_class_+1": [0.4],
        }
    )

    with pytest.raises(ValueError, match=r"unique class labels.*duplicate label\(s\): \[1\]"):
        _probability_ece_by_group(observations, ["subject", "time"], n_bins=10)

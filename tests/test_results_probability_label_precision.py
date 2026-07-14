from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.results import _probability_ece_by_group


def test_probability_ece_preserves_adjacent_large_integer_labels() -> None:
    first_label = 2**53
    second_label = first_label + 1
    observations = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "time": [0.184, 0.184],
            "true_label": pd.Series([first_label, second_label], dtype=object),
            f"prob_class_{first_label}": [0.9, 0.1],
            f"prob_class_{second_label}": [0.1, 0.9],
        }
    )

    result = _probability_ece_by_group(observations, ["subject", "time"], n_bins=10)

    assert result["n_observations"].tolist() == [2]
    assert result["ece"].tolist() == pytest.approx([0.1])


def test_probability_ece_rejects_ambiguous_large_float_labels() -> None:
    label = 2**53 + 2
    observations = pd.DataFrame(
        {
            "subject": ["s1"],
            "time": [0.184],
            "true_label": [float(label)],
            f"prob_class_{label}": [1.0],
        }
    )

    with pytest.raises(ValueError, match="must be exact integer labels"):
        _probability_ece_by_group(observations, ["subject", "time"], n_bins=10)

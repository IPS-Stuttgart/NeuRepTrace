from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.paired_stats import paired_decoder_statistics, sign_flip_p_value


def test_sign_flip_p_value_rejects_nan_difference() -> None:
    with pytest.raises(ValueError, match="finite"):
        sign_flip_p_value(np.array([0.1, float("nan")], dtype=float))


def test_sign_flip_p_value_rejects_infinite_difference() -> None:
    with pytest.raises(ValueError, match="finite"):
        sign_flip_p_value(np.array([0.1, float("inf")], dtype=float))


def test_paired_decoder_statistics_rejects_non_finite_metric_values() -> None:
    subject_metrics = pd.DataFrame(
        {
            "decoder": ["a", "a", "b", "b"],
            "subject": ["s1", "s2", "s1", "s2"],
            "effect_accuracy": [0.8, float("inf"), 0.7, 0.6],
        }
    )

    with pytest.raises(ValueError, match="finite"):
        paired_decoder_statistics(subject_metrics, metrics=("effect_accuracy",))

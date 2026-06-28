from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from neureptrace._observation_ensemble_source_debias_bool_patch import normalize_source_baseline_debiasing
from neureptrace.observation_ensemble import ensemble_probability_observations


def _source_observations() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for decoder, baseline_probs, post_probs in (
        ("logistic", (0.70, 0.30), (0.20, 0.80)),
        ("linear_svm", (0.60, 0.40), (0.25, 0.75)),
    ):
        rows.append(
            {
                "decoder": decoder,
                "emission_mode": "calibrated",
                "subject": "sub-01",
                "fold": 0,
                "time": -0.10,
                "true_label": 0,
                "prob_class_0": baseline_probs[0],
                "prob_class_1": baseline_probs[1],
            }
        )
        rows.append(
            {
                "decoder": decoder,
                "emission_mode": "calibrated",
                "subject": "sub-01",
                "fold": 0,
                "time": 0.10,
                "true_label": 1,
                "prob_class_0": post_probs[0],
                "prob_class_1": post_probs[1],
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        (False, False),
        (True, True),
        (np.bool_(False), False),
        (0, False),
        (1, True),
        (0.0, False),
        (1.0, True),
        ("false", False),
        ("OFF", False),
        ("true", True),
        ("yes", True),
        (np.asarray(False), False),
    ],
)
def test_source_baseline_debiasing_bool_tokens_normalize(value, expected) -> None:
    assert normalize_source_baseline_debiasing(value) is expected


@pytest.mark.parametrize("value", ["maybe", "", 2, -1, 0.5, np.inf, np.asarray([False, True])])
def test_source_baseline_debiasing_bool_tokens_reject_ambiguous_values(value) -> None:
    with pytest.raises(ValueError, match="source_baseline_debiasing"):
        normalize_source_baseline_debiasing(value)


def test_observation_ensemble_string_false_does_not_enable_source_debiasing() -> None:
    explicit_false = ensemble_probability_observations(_source_observations(), source_baseline_debiasing=False)
    string_false = ensemble_probability_observations(_source_observations(), source_baseline_debiasing="false")

    assert string_false["source_baseline_debiasing"].eq(False).all()
    assert_frame_equal(
        explicit_false.loc[:, ["time", "prob_class_0", "prob_class_1", "source_baseline_debiasing"]],
        string_false.loc[:, ["time", "prob_class_0", "prob_class_1", "source_baseline_debiasing"]],
    )


def test_observation_ensemble_rejects_ambiguous_source_debiasing_value() -> None:
    with pytest.raises(ValueError, match="source_baseline_debiasing"):
        ensemble_probability_observations(_source_observations(), source_baseline_debiasing="maybe")

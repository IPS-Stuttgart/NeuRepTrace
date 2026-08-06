from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.observation_ensemble import ensemble_probability_observations


def _source_observations() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": "sub-01",
                "fold": 0,
                "decoder": decoder,
                "emission_mode": "calibrated",
                "time": 0.1,
                "sample_index": 0,
                "sequence_id": 0,
                "true_label": 0,
                "true_class": "zero",
                "class_0": "zero",
                "class_1": "one",
                "prob_class_0": probability_zero,
                "prob_class_1": 1.0 - probability_zero,
            }
            for decoder, probability_zero in (
                ("logistic", 0.8),
                ("linear_svm", 0.7),
            )
        ]
    )


@pytest.mark.parametrize(
    ("kwargs", "control_name"),
    [
        (
            {"weights": (np.complex128(0.5 + 2.0j), 0.5), "baseline_window": None},
            "weights",
        ),
        (
            {
                "source_temperatures": (
                    np.complex128(1.0 + 2.0j),
                    1.0,
                ),
                "baseline_window": None,
            },
            "source_temperatures",
        ),
        (
            {
                "probability_tolerance": np.complex128(1e-3 + 2.0j),
                "baseline_window": None,
            },
            "probability_tolerance",
        ),
        (
            {
                "baseline_window": (
                    np.complex128(0.0 + 2.0j),
                    0.2,
                )
            },
            "baseline_window",
        ),
    ],
)
def test_observation_ensemble_rejects_complex_numeric_controls(
    kwargs: dict[str, object],
    control_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{control_name} must contain only real-valued numbers",
    ):
        ensemble_probability_observations(_source_observations(), **kwargs)


def test_observation_ensemble_rejects_object_array_complex_weights() -> None:
    weights = np.asarray([np.complex128(0.5 + 1.0j), 0.5], dtype=object)

    with pytest.raises(
        ValueError,
        match="weights must contain only real-valued numbers",
    ):
        ensemble_probability_observations(
            _source_observations(),
            weights=weights,
            baseline_window=None,
        )

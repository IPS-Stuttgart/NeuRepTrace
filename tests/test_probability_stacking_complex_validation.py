from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.probability_stacking import (
    combine_probability_cube,
    fit_source_oof_stacking,
    fit_stacking_weights,
    stack_probability_observations,
    summarize_stacked_metrics,
)


def _probability_cube() -> np.ndarray:
    return np.asarray(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.6, 0.4], [0.4, 0.6]],
        ],
        dtype=float,
    )


def _observation_rows(*, subject: str, labels: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for decoder in ("weak", "strong"):
        for sample_index, true_label in enumerate(labels):
            if decoder == "strong":
                prob_0, prob_1 = ((0.9, 0.1) if true_label == 0 else (0.1, 0.9))
            else:
                prob_0, prob_1 = ((0.6, 0.4) if true_label == 0 else (0.4, 0.6))
            rows.append(
                {
                    "subject": subject,
                    "fold": sample_index % 2,
                    "decoder": decoder,
                    "emission_mode": "calibrated",
                    "time": 0.1,
                    "window_start": 0.05,
                    "window_stop": 0.15,
                    "sample_index": sample_index,
                    "true_label": int(true_label),
                    "true_class": "zero" if true_label == 0 else "one",
                    "class_0": "zero",
                    "class_1": "one",
                    "prob_class_0": prob_0,
                    "prob_class_1": prob_1,
                }
            )
    return pd.DataFrame(rows)


def _complex_probabilities(values: np.ndarray) -> np.ndarray:
    offsets = np.asarray([0.1j, -0.1j], dtype=np.complex128)
    return np.asarray(values, dtype=np.complex128) + offsets


def test_fit_stacking_weights_rejects_complex_probability_cube() -> None:
    cube = np.stack(
        [_complex_probabilities(candidate) for candidate in _probability_cube()],
        axis=0,
    )

    with pytest.raises(
        ValueError,
        match="Probability values must contain real-valued probabilities",
    ):
        fit_stacking_weights(cube, [0, 1])


def test_combine_probability_cube_rejects_complex_weights() -> None:
    with pytest.raises(ValueError, match="weights must be real-valued"):
        combine_probability_cube(
            _probability_cube(),
            np.asarray([1.0 + 0.5j, 1.0 - 0.5j]),
        )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("max_iter", np.complex128(20.0 + 1.0j), "max_iter must be a positive integer"),
        (
            "learning_rate",
            np.complex128(0.25 + 0.1j),
            "learning_rate must be positive and finite",
        ),
        (
            "min_probability",
            np.complex128(1.0e-6 + 1.0e-7j),
            r"min_probability must lie in \(0, 1\)",
        ),
    ],
)
def test_fit_stacking_weights_rejects_complex_scalar_controls(
    argument: str,
    value: object,
    message: str,
) -> None:
    kwargs = {argument: value}

    with pytest.raises(ValueError, match=message):
        fit_stacking_weights(_probability_cube(), [0, 1], **kwargs)


def test_fit_source_oof_stacking_rejects_complex_temperature() -> None:
    with pytest.raises(ValueError, match="temperature must be positive and finite"):
        fit_source_oof_stacking(
            _probability_cube(),
            [0, 1],
            candidates=("strong", "weak"),
            weighting="softmax",
            temperature=np.complex128(0.05 + 0.01j),
        )


def test_stack_probability_observations_rejects_complex_columns() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1])
    probability_columns = ["prob_class_0", "prob_class_1"]
    complex_values = _complex_probabilities(source.loc[:, probability_columns].to_numpy())
    for column_index, column in enumerate(probability_columns):
        source[column] = complex_values[:, column_index]

    with pytest.raises(
        ValueError,
        match="Probability values must contain real-valued probabilities",
    ):
        stack_probability_observations(
            source,
            target,
            weighting="stacked",
            max_iter=20,
        )


def test_summarize_stacked_metrics_rejects_complex_probabilities() -> None:
    observations = pd.DataFrame(
        {
            "true_label": [0, 1],
            "prob_class_0": [0.8 + 0.1j, 0.2 + 0.1j],
            "prob_class_1": [0.2 - 0.1j, 0.8 - 0.1j],
        }
    )

    with pytest.raises(
        ValueError,
        match="Probability values must contain real-valued probabilities",
    ):
        summarize_stacked_metrics(observations)

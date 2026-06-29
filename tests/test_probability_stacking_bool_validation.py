from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.probability_stacking import (
    fit_source_oof_stacking,
    fit_stacking_weights,
    stack_probability_observations,
    summarize_stacked_metrics,
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


def test_fit_source_oof_stacking_rejects_boolean_source_labels() -> None:
    cube = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.6, 0.4], [0.4, 0.6]],
        ]
    )

    with pytest.raises(ValueError, match="source_labels values must be numeric, not boolean"):
        fit_source_oof_stacking(cube, [False, True], candidates=("strong", "weak"))


def test_fit_stacking_weights_rejects_boolean_probability_cube() -> None:
    cube = np.array(
        [
            [[True, False], [False, True]],
            [[False, True], [True, False]],
        ]
    )

    with pytest.raises(ValueError, match="Probability values must be numeric, not boolean"):
        fit_stacking_weights(cube, [0, 1])


def test_fit_stacking_weights_rejects_boolean_labels() -> None:
    cube = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.6, 0.4], [0.4, 0.6]],
        ]
    )

    with pytest.raises(ValueError, match="labels values must be numeric, not boolean"):
        fit_stacking_weights(cube, [False, True])


def test_stack_probability_observations_rejects_boolean_probability_columns() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1])
    source[["prob_class_0", "prob_class_1"]] = source[["prob_class_0", "prob_class_1"]] > 0.5

    with pytest.raises(ValueError, match="Probability values must be numeric, not boolean"):
        stack_probability_observations(source, target, weighting="stacked", max_iter=20)


def test_summarize_stacked_metrics_rejects_boolean_true_labels() -> None:
    observations = pd.DataFrame(
        {
            "true_label": [False, True],
            "prob_class_0": [0.8, 0.2],
            "prob_class_1": [0.2, 0.8],
        }
    )

    with pytest.raises(ValueError, match="true_label values must be numeric, not boolean"):
        summarize_stacked_metrics(observations)


def test_summarize_stacked_metrics_rejects_boolean_probability_columns() -> None:
    observations = pd.DataFrame(
        {
            "true_label": [0, 1],
            "prob_class_0": [True, False],
            "prob_class_1": [False, True],
        }
    )

    with pytest.raises(ValueError, match="Probability values must be numeric, not boolean"):
        summarize_stacked_metrics(observations)

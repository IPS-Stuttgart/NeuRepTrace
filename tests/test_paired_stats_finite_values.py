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


@pytest.mark.parametrize(
    "differences",
    [
        np.array([0.1 + 0.2j, -0.2 + 0.0j]),
        np.array([np.asarray(0.1 + 0.2j), np.asarray(-0.2 + 0.0j)], dtype=object),
    ],
)
def test_sign_flip_p_value_rejects_complex_differences(differences: np.ndarray) -> None:
    with pytest.raises(ValueError, match="differences must contain only real values"):
        sign_flip_p_value(differences)


@pytest.mark.parametrize(
    "differences",
    [
        np.array([True, False], dtype=bool),
        np.array([np.bool_(True), 0.2], dtype=object),
        np.array([np.asarray(True), np.asarray(False)], dtype=object),
    ],
)
def test_sign_flip_p_value_rejects_boolean_differences(differences: np.ndarray) -> None:
    with pytest.raises(ValueError, match="differences must not contain boolean values"):
        sign_flip_p_value(differences)


def test_sign_flip_p_value_rejects_boolean_random_state() -> None:
    with pytest.raises(ValueError, match="random_state must be a non-negative integer seed"):
        sign_flip_p_value(np.array([0.1, -0.2, 0.3], dtype=float), n_permutations=2, random_state=True)

    with pytest.raises(ValueError, match="random_state must be a non-negative integer seed"):
        sign_flip_p_value(np.array([0.1, -0.2, 0.3], dtype=float), n_permutations=2, random_state=np.bool_(False))


def test_paired_decoder_statistics_rejects_boolean_random_state() -> None:
    subject_metrics = pd.DataFrame(
        {
            "decoder": ["a", "a", "b", "b"],
            "subject": ["s1", "s2", "s1", "s2"],
            "effect_accuracy": [0.8, 0.7, 0.6, 0.5],
        }
    )

    with pytest.raises(ValueError, match="random_state must be a non-negative integer seed"):
        paired_decoder_statistics(subject_metrics, metrics=("effect_accuracy",), n_permutations=2, random_state=True)


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


@pytest.mark.parametrize(
    "metric_values",
    [
        np.array([0.8 + 0.1j, 0.7 + 0.0j, 0.6 + 0.0j, 0.5 + 0.0j]),
        np.array(
            [
                np.asarray(0.8 + 0.1j),
                np.asarray(0.7 + 0.0j),
                np.asarray(0.6 + 0.0j),
                np.asarray(0.5 + 0.0j),
            ],
            dtype=object,
        ),
    ],
)
def test_paired_decoder_statistics_rejects_complex_metric_values(metric_values: np.ndarray) -> None:
    subject_metrics = pd.DataFrame(
        {
            "decoder": ["a", "a", "b", "b"],
            "subject": ["s1", "s2", "s1", "s2"],
            "effect_accuracy": metric_values,
        }
    )

    with pytest.raises(ValueError, match="complex values in metric 'effect_accuracy'"):
        paired_decoder_statistics(subject_metrics, metrics=("effect_accuracy",))


@pytest.mark.parametrize(
    "metric_values",
    [
        [True, False, True, False],
        np.array([0.8, np.bool_(False), 0.6, 0.5], dtype=object),
        pd.array([True, False, True, False], dtype="boolean"),
    ],
)
def test_paired_decoder_statistics_rejects_boolean_metric_values(metric_values: object) -> None:
    subject_metrics = pd.DataFrame(
        {
            "decoder": ["a", "a", "b", "b"],
            "subject": ["s1", "s2", "s1", "s2"],
            "effect_accuracy": metric_values,
        }
    )

    with pytest.raises(ValueError, match="boolean values in metric 'effect_accuracy'"):
        paired_decoder_statistics(subject_metrics, metrics=("effect_accuracy",))


def test_paired_decoder_statistics_accepts_numeric_zero_one_metrics() -> None:
    subject_metrics = pd.DataFrame(
        {
            "decoder": ["a", "a", "b", "b"],
            "subject": ["s1", "s2", "s1", "s2"],
            "effect_accuracy": [1.0, 0.0, 0.75, 0.25],
        }
    )

    statistics = paired_decoder_statistics(
        subject_metrics,
        metrics=("effect_accuracy",),
        n_permutations=4,
    )

    assert statistics.loc[0, "decoder_a_mean"] == pytest.approx(0.5)
    assert statistics.loc[0, "decoder_b_mean"] == pytest.approx(0.5)

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.bushmeg_covariance_loso import (
    CovarianceCandidateSpec,
    CovarianceWindow,
    _candidate_grid,
    _channel_subset_indices,
    _float_grid,
    _normalize_covariance_epsilon,
    _normalize_covariance_shrinkage,
    _shuffle_training_labels,
    covariance_feature_vector,
    normalize_covariance_feature_mode,
    run_covariance_loso_subjects,
)
from neureptrace.bushmeg_source_loso import SubjectEpochs


def test_covariance_feature_modes_are_finite_and_pymegdec_compatible():
    signal = np.array(
        [
            [1.0, 2.0, 4.0, 5.0],
            [2.0, 3.0, 5.0, 7.0],
            [7.0, 6.0, 4.0, 3.0],
        ],
        dtype=float,
    )

    variance = covariance_feature_vector(signal, "variance", shrinkage=0.1, epsilon=1e-6)
    covariance = covariance_feature_vector(signal, "covariance_upper", shrinkage=0.1, epsilon=1e-6)
    correlation = covariance_feature_vector(signal, "correlation_upper", shrinkage=0.1, epsilon=1e-6)
    logeuclidean = covariance_feature_vector(signal, "logeig-covariance", shrinkage=0.1, epsilon=1e-6)

    assert normalize_covariance_feature_mode("covariance") == "covariance_upper"
    assert variance.shape == (3,)
    assert covariance.shape == (6,)
    assert correlation.shape == (6,)
    assert logeuclidean.shape == (6,)
    assert np.all(np.isfinite(variance))
    assert np.all(np.isfinite(covariance))
    assert np.all(np.isfinite(correlation))
    assert np.all(np.isfinite(logeuclidean))
    np.testing.assert_allclose(correlation[[0, 3, 5]], np.ones(3))


def _toy_subject(subject_id: str, *, offset: float) -> SubjectEpochs:
    times = np.linspace(0.0, 0.30, 8)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=int)
    data = np.zeros((labels.size, 4, times.size), dtype=np.float32)
    phase = np.linspace(-1.0, 1.0, times.size)
    for trial_index, label in enumerate(labels):
        scale = -1.0 if label == 0 else 1.0
        amplitude = 1.0 + 0.2 * trial_index
        data[trial_index, 0] = offset + scale * amplitude * phase
        data[trial_index, 1] = offset - scale * 0.5 * amplitude * phase
        data[trial_index, 2] = offset + 0.25 * phase**2 + 0.01 * trial_index
        data[trial_index, 3] = offset + np.sin(np.linspace(0.0, np.pi, times.size)) * (1.0 + 0.1 * label)
    return SubjectEpochs(
        subject=subject_id,
        data=data,
        times=times,
        metadata=pd.DataFrame(
            {
                "participant": subject_id,
                "stimulus_class": labels,
                "condition": ["main"] * labels.size,
            }
        ),
        labels=labels,
    )


def _toy_candidate() -> CovarianceCandidateSpec:
    return CovarianceCandidateSpec(
        name="covariance_smoke",
        decoder="logistic",
        emission_mode="uncalibrated",
        feature_preprocessor="none",
        pca_components=None,
        classifier_param=1.0,
        window=CovarianceWindow(name="post", start=0.0, stop=0.30),
        covariance_feature_mode="correlation_upper",
        covariance_shrinkage=0.1,
        covariance_epsilon=1e-6,
        covariance_max_channels=4,
    )


def _toy_subjects() -> dict[str, SubjectEpochs]:
    return {
        "s1": _toy_subject("s1", offset=0.0),
        "s2": _toy_subject("s2", offset=0.1),
        "s3": _toy_subject("s3", offset=-0.1),
    }


def test_covariance_loso_runs_on_in_memory_subjects():
    subjects = _toy_subjects()
    candidate = _toy_candidate()

    summary, inner, predictions = run_covariance_loso_subjects(
        subjects,
        candidates=[candidate],
        class_names=["a", "b"],
        max_iter=300,
    )

    assert len(summary) == 3
    assert len(inner) == 6
    assert len(predictions) == 18
    assert set(summary["covariance_feature_mode"]) == {"correlation_upper"}
    assert np.all(np.isfinite(summary["balanced_accuracy"]))
    assert {"prob_class_0", "prob_class_1"}.issubset(predictions.columns)
    assert predictions["label_shuffle_control"].unique().tolist() == [False]
    assert predictions["label_shuffle_seed"].unique().tolist() == [0]


def test_label_shuffle_control_is_deterministic_and_count_preserving():
    labels = np.repeat(np.arange(4), 5)
    shuffled_a = _shuffle_training_labels(labels, seed=7, context=("outer", "inner", "candidate"))
    shuffled_b = _shuffle_training_labels(labels, seed=7, context=("outer", "inner", "candidate"))
    shuffled_c = _shuffle_training_labels(labels, seed=8, context=("outer", "inner", "candidate"))

    np.testing.assert_array_equal(shuffled_a, shuffled_b)
    assert sorted(shuffled_a.tolist()) == sorted(labels.tolist())
    assert not np.array_equal(shuffled_a, shuffled_c)


def test_covariance_numeric_controls_reject_booleans_and_fractional_counts():
    with pytest.raises(ValueError, match="covariance_shrinkage"):
        _normalize_covariance_shrinkage(True)

    with pytest.raises(ValueError, match="covariance_epsilon"):
        _normalize_covariance_epsilon(True)

    with pytest.raises(ValueError, match="covariance_shrinkages"):
        _float_grid([0.1, True], [0.1], name="covariance_shrinkages")

    with pytest.raises(ValueError, match="covariance_max_channels must be an integer"):
        _channel_subset_indices(4, 2.5)

    with pytest.raises(ValueError, match="covariance_max_channels must be an integer"):
        _channel_subset_indices(4, True)


def test_covariance_candidate_grid_rejects_invalid_channel_and_window_values():
    base_config = {
        "decoding": {
            "decoder": "logistic",
            "emission_mode": "uncalibrated",
            "feature_preprocessor": "none",
            "pca_components": None,
            "tuning_c_grid": "1.0",
        },
        "covariance_loso": {
            "candidate_grid": {
                "time_windows": [{"name": "post", "start": 0.0, "stop": 0.3}],
                "covariance_max_channels": [4],
            }
        },
    }

    bad_channels = {
        **base_config,
        "covariance_loso": {
            "candidate_grid": {
                "time_windows": [{"name": "post", "start": 0.0, "stop": 0.3}],
                "covariance_max_channels": [1.5],
            }
        },
    }
    with pytest.raises(ValueError, match="covariance_max_channels must be an integer"):
        _candidate_grid(bad_channels)

    bad_window = {
        **base_config,
        "covariance_loso": {
            "candidate_grid": {
                "time_windows": [{"name": "post", "start": False, "stop": 0.3}],
                "covariance_max_channels": [4],
            }
        },
    }
    with pytest.raises(ValueError, match="covariance_window_start"):
        _candidate_grid(bad_window)


def test_covariance_loso_rejects_invalid_runtime_controls():
    subjects = _toy_subjects()
    candidate = _toy_candidate()

    with pytest.raises(ValueError, match="max_iter must be an integer"):
        run_covariance_loso_subjects(subjects, candidates=[candidate], class_names=["a", "b"], max_iter=300.5)

    with pytest.raises(ValueError, match="max_iter must be an integer"):
        run_covariance_loso_subjects(subjects, candidates=[candidate], class_names=["a", "b"], max_iter=True)

    with pytest.raises(ValueError, match="label_shuffle_seed must be an integer"):
        run_covariance_loso_subjects(
            subjects,
            candidates=[candidate],
            class_names=["a", "b"],
            label_shuffle_control=True,
            label_shuffle_seed=13.5,
        )

    with pytest.raises(ValueError, match="label_shuffle_seed must be an integer"):
        _shuffle_training_labels(np.array([0, 1]), seed=True, context=("outer",))

    with pytest.raises(ValueError, match="at least two classes"):
        run_covariance_loso_subjects(subjects, candidates=[candidate], class_names=["a"])

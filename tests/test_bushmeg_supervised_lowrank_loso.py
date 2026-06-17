from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.bushmeg_source_loso import SubjectEpochs
from neureptrace.bushmeg_supervised_lowrank_loso import (
    EpochWindow,
    LowRankCandidateSpec,
    SupervisedPLSTransformer,
    _combine_probabilities,
    _epoch_bin_mean_features,
    run_supervised_lowrank_loso_subjects,
)


def _synthetic_subjects() -> dict[str, SubjectEpochs]:
    rng = np.random.default_rng(13)
    times = np.linspace(-0.10, 0.35, 46)
    n_subjects = 4
    n_classes = 3
    trials_per_class = 4
    n_channels = 5
    subjects = {}
    class_patterns = np.array(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0],
        ]
    )
    signal_mask = (times >= 0.05) & (times <= 0.25)
    for subject_index in range(n_subjects):
        labels = np.repeat(np.arange(n_classes), trials_per_class)
        data = rng.normal(scale=0.15, size=(labels.size, n_channels, times.size))
        subject_offset = rng.normal(scale=0.05, size=(n_channels, 1))
        data += subject_offset
        for trial_index, label in enumerate(labels):
            data[trial_index, :, signal_mask] += class_patterns[label][None, :]
        metadata = pd.DataFrame(
            {
                "participant": f"s{subject_index}",
                "stimulus_class": [f"class_{label}" for label in labels],
            }
        )
        subjects[f"s{subject_index}"] = SubjectEpochs(
            subject=f"s{subject_index}",
            data=data.astype(np.float32),
            times=times,
            metadata=metadata,
            labels=labels.astype(int),
        )
    return subjects


def test_supervised_pls_transformer_clips_to_feasible_rank():
    rng = np.random.default_rng(17)
    features = rng.normal(size=(6, 4))
    labels = np.array([0, 1, 2, 0, 1, 2])

    transformer = SupervisedPLSTransformer(n_components=50).fit(features, labels)
    projected = transformer.transform(features)

    assert transformer.n_components_ == 4
    assert projected.shape == (6, 4)


def test_epoch_bin_mean_features_can_append_deltas():
    data = np.arange(2 * 3 * 6, dtype=float).reshape(2, 3, 6)
    times = np.linspace(0.0, 0.5, 6)
    window = EpochWindow("toy", 0.0, 0.5)

    features_without_delta = _epoch_bin_mean_features(data, times, window, temporal_bins=3, include_deltas=False)
    features_with_delta = _epoch_bin_mean_features(data, times, window, temporal_bins=3, include_deltas=True)

    assert features_without_delta.shape == (2, 9)
    assert features_with_delta.shape == (2, 15)


def test_combine_probabilities_supports_log_mean():
    first = np.array([[0.9, 0.1], [0.2, 0.8]])
    second = np.array([[0.8, 0.2], [0.4, 0.6]])

    combined = _combine_probabilities([first, second], mode="log_mean")

    assert combined.shape == (2, 2)
    np.testing.assert_allclose(combined.sum(axis=1), 1.0)
    assert combined[0, 0] > combined[0, 1]
    assert combined[1, 1] > combined[1, 0]


def test_combine_probabilities_rejects_invalid_candidate_probabilities():
    first = np.array([[0.9, 0.1], [0.2, 0.8]])
    second = np.array([[0.4, 0.4], [0.4, 0.6]])

    with pytest.raises(ValueError, match="must sum to 1.0"):
        _combine_probabilities([first, second], mode="log_mean")


def test_combine_probabilities_rejects_invalid_min_probability():
    first = np.array([[0.9, 0.1], [0.2, 0.8]])

    with pytest.raises(ValueError, match="min_probability"):
        _combine_probabilities([first], mode="log_mean", min_probability=1.0)


def test_supervised_lowrank_loso_selects_and_predicts_on_synthetic_subjects():
    subjects = _synthetic_subjects()
    candidates = [
        LowRankCandidateSpec(
            name="post_signal",
            decoder="multinomial-logistic",
            classifier_param=1.0,
            window=EpochWindow("post", 0.00, 0.30),
            temporal_bins=6,
            pls_components=3,
        ),
        LowRankCandidateSpec(
            name="post_signal_delta",
            decoder="multinomial-logistic",
            classifier_param=1.0,
            window=EpochWindow("post", 0.00, 0.30),
            temporal_bins=6,
            pls_components=3,
            include_deltas=True,
        ),
    ]

    summary, inner, predictions = run_supervised_lowrank_loso_subjects(
        subjects,
        candidates=candidates,
        class_names=["class_0", "class_1", "class_2"],
        ensemble_size=2,
        max_iter=1000,
    )

    assert len(summary) == 4
    assert set(summary["outer_test_subject"]) == set(subjects)
    assert summary["ensemble_size"].tolist() == [2, 2, 2, 2]
    assert summary["balanced_accuracy"].min() > 0.9
    assert len(inner) == 4 * 2 * 3
    assert len(predictions) == sum(len(subject.labels) for subject in subjects.values())
    probability_columns = [column for column in predictions.columns if column.startswith("prob_class_")]
    assert len(probability_columns) == 3
    np.testing.assert_allclose(predictions[probability_columns].sum(axis=1), 1.0, atol=1e-8)

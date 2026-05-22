from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.decoding import make_decoder
from neureptrace.bushmeg_source_loso import (
    CandidateSpec,
    FeatureCache,
    SubjectEpochs,
    WindowSpec,
    _candidate_metrics,
    _select_candidate,
    _window_bin_mean_features,
)


def test_window_bin_mean_features_concatenates_channel_bins():
    data = np.array(
        [
            [[1.0, 2.0, 10.0, 14.0], [4.0, 6.0, 8.0, 10.0]],
            [[3.0, 5.0, 20.0, 24.0], [2.0, 4.0, 6.0, 8.0]],
        ],
        dtype=float,
    )
    times = np.array([0.10, 0.15, 0.20, 0.25])
    features = _window_bin_mean_features(
        data,
        times,
        WindowSpec(center=0.175, width=0.20),
        temporal_bins=2,
    )

    expected = np.array(
        [
            [1.5, 5.0, 12.0, 9.0],
            [4.0, 3.0, 22.0, 7.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(features, expected)


def test_candidate_metrics_report_multiclass_topk():
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.4, 0.3, 0.3],
            [0.1, 0.3, 0.6],
        ]
    )
    labels = np.array([0, 1, 2])

    metrics = _candidate_metrics(probabilities, labels, n_classes=3)

    assert metrics["accuracy"] == 2 / 3
    assert metrics["balanced_accuracy"] == 2 / 3
    assert metrics["top2_accuracy"] == 1.0
    assert metrics["top3_accuracy"] == 1.0


def test_select_candidate_uses_only_source_subjects_for_inner_loso():
    # The first feature dimension carries the class identity with a subject offset.
    # Both candidates are valid, but the one with temporal binning captures the
    # class signal after averaging across two samples.
    subjects = {}
    times = np.array([0.10, 0.20])
    for subject_idx in range(4):
        labels = np.array([0, 0, 1, 1])
        data = np.zeros((4, 1, 2), dtype=np.float32)
        data[:, 0, 0] = labels + subject_idx * 0.01
        data[:, 0, 1] = labels + subject_idx * 0.01
        subjects[str(subject_idx)] = SubjectEpochs(
            subject=str(subject_idx),
            data=data,
            times=times,
            metadata=pd.DataFrame(),
            labels=labels,
        )

    candidates = [
        CandidateSpec(
            name="mean_bin",
            decoder="logistic",
            emission_mode="uncalibrated",
            feature_preprocessor="none",
            pca_components=None,
            classifier_param=1.0,
            temporal_bins=1,
            windows=(WindowSpec(center=0.15, width=0.20),),
        )
    ]
    selected, rows, summary = _select_candidate(
        subjects=subjects,
        cache=FeatureCache(subjects),
        candidates=candidates,
        outer_test_subject="3",
        n_classes=2,
        max_iter=200,
        selection_metric="balanced_accuracy",
    )

    assert selected.name == "mean_bin"
    assert {row["inner_test_subject"] for row in rows} == {"0", "1", "2"}
    assert all(row["outer_test_subject"] == "3" for row in rows)
    assert summary["inner_n_folds"] == 3


def test_make_decoder_uses_classifier_param_for_logistic_c():
    model = make_decoder(
        "logistic",
        emission_mode="uncalibrated",
        classifier_param=0.25,
    )

    assert model.named_steps["logisticregression"].C == 0.25


def test_make_decoder_uses_classifier_param_for_linear_svm_c():
    model = make_decoder(
        "linear_svm",
        emission_mode="uncalibrated",
        classifier_param=0.5,
    )

    assert model.named_steps["linearsvc"].C == 0.5

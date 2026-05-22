from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from neureptrace.decoding import make_decoder, normalize_decoder_name
from neureptrace.bushmeg_source_loso import (
    CandidateSpec,
    FeatureCache,
    SubjectEpochs,
    WindowSpec,
    _apply_class_bias,
    _candidate_metrics,
    _fit_candidate_model,
    _fit_class_bias,
    _prepare_window_train_test_features,
    _sample_weights_for_training,
    _select_candidate,
    _window_features,
    _window_bin_mean_features,
    normalize_source_feature_family,
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


def test_template_similarity_features_are_source_subject_prototype_scores():
    subjects = {}
    times = np.array([0.15])
    labels = np.array([0, 0, 1, 1])
    for subject_idx in range(3):
        offset = subject_idx * 0.25
        data = np.zeros((4, 2, 1), dtype=np.float32)
        data[labels == 0, :, 0] = np.array([2.0 + offset, -2.0 + offset])
        data[labels == 1, :, 0] = np.array([-2.0 + offset, 2.0 + offset])
        subjects[str(subject_idx)] = SubjectEpochs(
            subject=str(subject_idx),
            data=data,
            times=times,
            metadata=pd.DataFrame(),
            labels=labels,
        )

    candidate = CandidateSpec(
        name="template",
        decoder="logistic",
        emission_mode="uncalibrated",
        feature_preprocessor="none",
        pca_components=None,
        classifier_param=1.0,
        temporal_bins=1,
        windows=(WindowSpec(center=0.15, width=0.05),),
        feature_family="template_similarity",
    )

    train_features, test_features = _prepare_window_train_test_features(
        subjects=subjects,
        cache=FeatureCache(subjects),
        candidate=candidate,
        train_subjects=["0", "1"],
        test_subject="2",
        window=WindowSpec(center=0.15, width=0.05),
        n_classes=2,
    )

    assert train_features.shape == (8, 2)
    assert test_features.shape == (4, 2)
    assert np.all(test_features[:2, 0] > test_features[:2, 1])
    assert np.all(test_features[2:, 1] > test_features[2:, 0])


def test_normalize_source_feature_family_aliases():
    assert normalize_source_feature_family("template-corr") == "template_similarity"
    assert normalize_source_feature_family("templates-plus-bin-means") == "template_similarity_plus_bin_means"


def test_window_feature_kinds_add_logvar_and_covariance_branches():
    data = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [4.0, 5.0, 6.0, 7.0], [7.0, 8.0, 9.0, 10.0]],
            [[2.0, 4.0, 6.0, 8.0], [1.0, 3.0, 5.0, 7.0], [8.0, 6.0, 4.0, 2.0]],
        ],
        dtype=np.float32,
    )
    times = np.array([0.10, 0.15, 0.20, 0.25])
    window = WindowSpec(center=0.175, width=0.20)

    evoked = _window_features(data, times, window, temporal_bins=2, feature_kind="evoked")
    logvar = _window_features(data, times, window, temporal_bins=2, feature_kind="logvar")
    evoked_logvar = _window_features(data, times, window, temporal_bins=2, feature_kind="evoked_logvar")
    covariance = _window_features(data, times, window, temporal_bins=2, feature_kind="covariance", covariance_max_channels=2)
    evoked_covariance = _window_features(data, times, window, temporal_bins=2, feature_kind="evoked_covariance", covariance_max_channels=2)

    assert evoked.shape == (2, 6)
    assert logvar.shape == (2, 6)
    assert evoked_logvar.shape == (2, 12)
    assert covariance.shape == (2, 3)
    assert evoked_covariance.shape == (2, 9)
    assert np.all(np.isfinite(logvar))
    assert np.all(np.isfinite(covariance))


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


def test_subject_class_balanced_sample_weights_equalize_observed_cells():
    times = np.array([0.10])
    subjects = {
        "s1": SubjectEpochs(
            subject="s1",
            data=np.zeros((3, 1, 1), dtype=np.float32),
            times=times,
            metadata=pd.DataFrame(),
            labels=np.array([0, 0, 1]),
        ),
        "s2": SubjectEpochs(
            subject="s2",
            data=np.zeros((4, 1, 1), dtype=np.float32),
            times=times,
            metadata=pd.DataFrame(),
            labels=np.array([0, 1, 1, 1]),
        ),
    }
    labels = np.concatenate([subjects["s1"].labels, subjects["s2"].labels])

    weights = _sample_weights_for_training(subjects, ["s1", "s2"], labels, "subject_class_balanced")

    assert weights is not None
    assert np.isclose(weights.mean(), 1.0)
    cell_sums = []
    cursor = 0
    for subject in ["s1", "s2"]:
        subject_labels = subjects[subject].labels
        subject_weights = weights[cursor : cursor + len(subject_labels)]
        cursor += len(subject_labels)
        for class_label in np.unique(subject_labels):
            cell_sums.append(float(subject_weights[subject_labels == class_label].sum()))
    np.testing.assert_allclose(cell_sums, np.full(len(cell_sums), cell_sums[0]))


def test_balanced_accuracy_class_bias_can_adjust_overpredicted_class():
    probabilities = np.array(
        [
            [0.49, 0.51],
            [0.45, 0.55],
            [0.40, 0.60],
            [0.35, 0.65],
        ]
    )
    labels = np.array([0, 0, 1, 1])
    baseline_score = balanced_accuracy_score(labels, probabilities.argmax(axis=1))

    bias = _fit_class_bias(probabilities, labels, n_classes=2, mode="balanced_accuracy")
    adjusted = _apply_class_bias(probabilities, bias)

    assert balanced_accuracy_score(labels, adjusted.argmax(axis=1)) > baseline_score


def test_fit_candidate_model_routes_sample_weight_to_registry_decoder():
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    weights = np.array([1.0, 2.0, 1.0, 2.0])
    model = make_decoder(
        "multinomial-logistic",
        emission_mode="uncalibrated",
        feature_preprocessor="none",
        classifier_param=1.0,
    )

    fitted = _fit_candidate_model(model, features, labels, sample_weight=weights)

    assert fitted.predict(features).shape == labels.shape


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


def test_make_decoder_recognizes_ovo_and_ecoc_linear_svm():
    ovo = make_decoder("onevsone-linear-svm", emission_mode="uncalibrated", classifier_param=0.5)
    ecoc = make_decoder("ecoc-svm", emission_mode="uncalibrated", classifier_param=0.5)

    assert normalize_decoder_name("onevsone-linear-svm") == "ovo_linear_svm"
    assert normalize_decoder_name("output-code-linear-svm") == "ecoc_linear_svm"
    assert "onevsoneclassifier" in ovo.named_steps
    assert "ecoclinearsvc" in ecoc.named_steps
    assert ovo.named_steps["onevsoneclassifier"].estimator.C == 0.5
    assert ecoc.named_steps["ecoclinearsvc"].C == 0.5


def test_make_decoder_constructs_torch_mlp_without_importing_torch():
    model = make_decoder("shallow-torch-mlp", emission_mode="uncalibrated", classifier_param=1e-4, max_iter=3)

    assert normalize_decoder_name("shallow-torch-mlp") == "torch_mlp"
    assert "torchmlpclassifier" in model.named_steps
    assert model.named_steps["torchmlpclassifier"].weight_decay == 1e-4
    assert model.named_steps["torchmlpclassifier"].max_iter == 3

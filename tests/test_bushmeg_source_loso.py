from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from neureptrace.decoding.source_alignment import source_alignment_config
from neureptrace.decoding import make_decoder, normalize_decoder_name
from neureptrace.bushmeg_source_loso import (
    CandidateSpec,
    FeatureCache,
    _baseline_channel_whitener,
    SubjectEpochs,
    WindowSpec,
    _apply_class_bias,
    _candidate_metrics,
    _combine_window_probabilities,
    _class_prototype_similarity_features,
    _candidate_grid,
    _fit_candidate_model,
    _fit_class_bias,
    _predict_candidate,
    _preprocessing_normalization_name,
    _prepare_window_train_test_features,
    run_bushmeg_source_loso,
    _sample_weights_for_training,
    _select_candidate,
    _window_features,
    _window_bin_mean_features,
    _window_evoked_baseline_contrast_features,
    _window_evoked_slope_features,
    _window_evoked_dct_features,
    _window_evoked_stat_features,
    normalize_source_feature_family,
    normalize_source_feature_kind,
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


def test_window_evoked_baseline_contrast_features_subtract_trial_baseline():
    data = np.array(
        [
            [[1.0, 3.0, 10.0, 14.0], [20.0, 20.0, 25.0, 21.0]],
            [[5.0, 7.0, 9.0, 11.0], [2.0, 4.0, 5.0, 1.0]],
        ],
        dtype=np.float32,
    )
    times = np.array([-0.30, -0.10, 0.10, 0.20])

    features = _window_evoked_baseline_contrast_features(
        data,
        times,
        WindowSpec(center=0.15, width=0.10),
        temporal_bins=2,
    )

    expected = np.array(
        [
            [8.0, 5.0, 12.0, 1.0],
            [3.0, 2.0, 5.0, -2.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(features, expected)


def test_window_evoked_slope_features_append_within_bin_linear_contrasts():
    data = np.array(
        [
            [[1.0, 2.0, 10.0, 14.0], [4.0, 6.0, 8.0, 10.0]],
            [[3.0, 5.0, 20.0, 24.0], [2.0, 4.0, 6.0, 8.0]],
        ],
        dtype=float,
    )
    times = np.array([0.10, 0.15, 0.20, 0.25])
    window = WindowSpec(center=0.175, width=0.20)

    evoked = _window_bin_mean_features(data, times, window, temporal_bins=2)
    features = _window_evoked_slope_features(data, times, window, temporal_bins=2)

    scale = np.sqrt(2.0)
    expected = np.array(
        [
            [1.5, 5.0, 12.0, 9.0, 1.0 / scale, 2.0 / scale, 4.0 / scale, 2.0 / scale],
            [4.0, 3.0, 22.0, 7.0, 2.0 / scale, 2.0 / scale, 4.0 / scale, 2.0 / scale],
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(features[:, : evoked.shape[1]], evoked)
    np.testing.assert_allclose(features, expected, rtol=1e-6)


def test_window_evoked_dct_features_capture_shape_beyond_the_mean():
    data = np.array(
        [
            [[1.0, 0.0, -1.0, 0.0]],
            [[-1.0, 0.0, 1.0, 0.0]],
        ],
        dtype=np.float32,
    )
    times = np.array([0.10, 0.20, 0.30, 0.40])

    features = _window_evoked_dct_features(
        data,
        times,
        WindowSpec(center=0.25, width=0.30),
        temporal_bins=2,
    )

    assert features.shape == (2, 2)
    np.testing.assert_allclose(features[:, 0], np.zeros(2), atol=1e-7)
    assert abs(float(features[0, 1])) > 1e-7
    np.testing.assert_allclose(features[0, 1], -features[1, 1], rtol=1e-6, atol=1e-7)


def test_window_evoked_stat_features_capture_extrema_and_trend():
    data = np.array(
        [
            [[1.0, 2.0, 3.0]],
            [[3.0, 2.0, 1.0]],
        ],
        dtype=np.float32,
    )
    times = np.array([0.10, 0.20, 0.30])

    features = _window_evoked_stat_features(data, times, WindowSpec(center=0.20, width=0.20), temporal_bins=1)

    expected = np.array([[2.0, 1.0, 1.0, 3.0, 1.0], [2.0, 1.0, 1.0, 3.0, -1.0]], dtype=np.float32)
    assert features.shape == (2, 5)
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


def test_candidate_grid_supports_range_and_full_epoch_window_sets():
    config = {
        "preprocessing": {"window_size": 0.100, "tmin": -0.35, "tmax": 0.70},
        "decoding": {
            "classifier": "multinomial-logistic",
            "emission_mode": "uncalibrated",
            "feature_preprocessor": "none",
            "pca_components": None,
            "tuning_c_grid": "1.0",
        },
        "source_loso": {
            "candidate_grid": {
                "decoders": ["logistic"],
                "emission_modes": ["uncalibrated"],
                "feature_preprocessors": ["none"],
                "pca_components": [None],
                "temporal_bins": [2],
                "c_grid": [1.0],
                "window_sets": [
                    {"name": "late", "start": 0.300, "stop": 0.400, "step": 0.050, "window_size": 0.100},
                    {"name": "full", "full_epoch": True, "start": 0.000, "stop": 0.650},
                ],
            }
        },
    }

    window_specs = {tuple((round(window.center, 3), round(window.width, 3)) for window in candidate.windows) for candidate in _candidate_grid(config)}

    assert ((0.3, 0.1), (0.35, 0.1), (0.4, 0.1)) in window_specs
    assert ((0.325, 0.65),) in window_specs


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
    evoked_slope = _window_features(data, times, window, temporal_bins=2, feature_kind="evoked_slope")
    evoked_dct = _window_features(data, times, window, temporal_bins=2, feature_kind="evoked_dct")
    evoked_stats = _window_features(data, times, window, temporal_bins=2, feature_kind="evoked_stats")
    logvar = _window_features(data, times, window, temporal_bins=2, feature_kind="logvar")
    evoked_logvar = _window_features(data, times, window, temporal_bins=2, feature_kind="evoked_logvar")
    covariance = _window_features(data, times, window, temporal_bins=2, feature_kind="covariance", covariance_max_channels=2)
    evoked_covariance = _window_features(data, times, window, temporal_bins=2, feature_kind="evoked_covariance", covariance_max_channels=2)

    assert evoked.shape == (2, 6)
    assert evoked_slope.shape == (2, 12)
    assert evoked_dct.shape == (2, 6)
    assert evoked_stats.shape == (2, 30)
    assert logvar.shape == (2, 6)
    assert evoked_logvar.shape == (2, 12)
    assert covariance.shape == (2, 3)
    assert evoked_covariance.shape == (2, 9)
    assert np.all(np.isfinite(logvar))
    assert np.all(np.isfinite(evoked_slope))
    assert np.all(np.isfinite(evoked_stats))
    assert np.all(np.isfinite(evoked_dct))
    assert np.all(np.isfinite(covariance))


def test_mnn_feature_kinds_noise_normalize_before_feature_extraction():
    data = np.array(
        [
            [[1.0, 2.0, 5.0, 6.0], [2.0, 4.0, 4.0, 8.0]],
            [[2.0, 3.0, 6.0, 8.0], [4.0, 6.0, 8.0, 10.0]],
            [[-1.0, -2.0, -5.0, -6.0], [-2.0, -4.0, -4.0, -8.0]],
        ],
        dtype=np.float32,
    )
    times = np.array([-0.30, -0.10, 0.15, 0.20])
    window = WindowSpec(center=0.175, width=0.10)

    whitener = _baseline_channel_whitener(data, times)
    evoked = _window_features(data, times, window, temporal_bins=1, feature_kind="mnn_evoked")
    evoked_logvar = _window_features(data, times, window, temporal_bins=1, feature_kind="mnn_evoked_logvar")

    assert whitener.shape == (2, 2)
    assert evoked.shape == (3, 2)
    assert evoked_logvar.shape == (3, 4)
    assert np.all(np.isfinite(evoked_logvar))


def test_window_features_support_baseline_contrast_alias_and_mnn_variant():
    data = np.array(
        [
            [[1.0, 2.0, 5.0, 6.0], [2.0, 4.0, 4.0, 8.0]],
            [[2.0, 3.0, 6.0, 8.0], [4.0, 6.0, 8.0, 10.0]],
            [[-1.0, -2.0, -5.0, -6.0], [-2.0, -4.0, -4.0, -8.0]],
        ],
        dtype=np.float32,
    )
    times = np.array([-0.30, -0.10, 0.15, 0.20])
    window = WindowSpec(center=0.175, width=0.10)

    assert normalize_source_feature_kind("baseline-corrected-evoked") == "evoked_baseline_contrast"
    baseline_contrast = _window_features(data, times, window, temporal_bins=1, feature_kind="trial-baseline-contrast")
    mnn_baseline_contrast = _window_features(data, times, window, temporal_bins=1, feature_kind="mnn_evoked_baseline_contrast")

    assert baseline_contrast.shape == (3, 2)
    assert mnn_baseline_contrast.shape == (3, 2)
    assert np.all(np.isfinite(mnn_baseline_contrast))


def test_preprocessing_normalization_accepts_epoch_normalization_alias():
    assert _preprocessing_normalization_name({"epoch_normalization": "subject_baseline_whiten"}) == "subject_baseline_whiten"
    assert _preprocessing_normalization_name({"normalization": "subject_z", "epoch_normalization": "subject_baseline_whiten"}) == "subject_z"


def test_class_prototype_similarity_features_separates_matched_classes():
    train_features = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [-1.0, 0.0],
            [-0.9, -0.1],
        ],
        dtype=np.float32,
    )
    test_features = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
    labels = np.array([0, 0, 1, 1])

    train_proto, test_proto = _class_prototype_similarity_features(
        train_features,
        test_features,
        labels,
        n_classes=2,
    )

    assert train_proto.shape == (4, 4)
    assert test_proto.shape == (2, 4)
    assert test_proto[0, 0] > test_proto[0, 1]  # cosine to class 0 beats class 1
    assert test_proto[0, 2] > test_proto[0, 3]  # distance score to class 0 beats class 1
    assert np.all(np.isfinite(train_proto))


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


def test_log_probability_window_combine_uses_geometric_mean():
    probabilities_a = np.array([[0.9, 0.1], [0.4, 0.6]], dtype=float)
    probabilities_b = np.array([[0.5, 0.5], [0.8, 0.2]], dtype=float)
    probability_mean = _combine_window_probabilities(probabilities_a + probabilities_b, 2, "probability_mean")
    log_probability_mean = _combine_window_probabilities(
        np.log(probabilities_a) + np.log(probabilities_b),
        2,
        "log_probability_mean",
    )

    expected = np.sqrt(probabilities_a * probabilities_b)
    expected /= expected.sum(axis=1, keepdims=True)

    np.testing.assert_allclose(log_probability_mean, expected)
    assert not np.allclose(log_probability_mean, probability_mean)


def test_candidate_grid_expands_window_combine_modes():
    config = {
        "preprocessing": {"window_size": 0.100, "tmin": -0.35, "tmax": 0.35},
        "decoding": {
            "classifier": "multinomial-logistic",
            "emission_mode": "uncalibrated",
            "feature_preprocessor": "none",
            "pca_components": None,
            "tuning_c_grid": "1.0",
        },
        "source_loso": {
            "candidate_grid": {
                "decoders": ["logistic"],
                "emission_modes": ["uncalibrated"],
                "feature_preprocessors": ["none"],
                "pca_components": [None],
                "temporal_bins": [1],
                "c_grid": [1.0],
                "window_combines": ["log_probability_mean"],
                "window_sets": [
                    {"name": "response_window_c", "centers": [0.088, 0.136, 0.184, 0.232, 0.280], "window_size": 0.100},
                ],
            }
        },
    }

    candidates = _candidate_grid(config)

    assert {candidate.window_combine for candidate in candidates} == {"log_probability_mean"}
    assert candidates[0].window_centers == (0.088, 0.136, 0.184, 0.232, 0.28)


def test_predict_candidate_train_label_shuffle_is_train_only(monkeypatch):
    subjects = {}
    times = np.array([0.15])
    labels = np.array([0, 0, 1, 1])
    for subject_idx in range(3):
        data = np.zeros((4, 1, 1), dtype=np.float32)
        data[:, 0, 0] = labels * 4.0 + subject_idx * 0.01
        subjects[str(subject_idx)] = SubjectEpochs(
            subject=str(subject_idx),
            data=data,
            times=times,
            metadata=pd.DataFrame(),
            labels=labels,
        )
    candidate = CandidateSpec(
        name="single",
        decoder="logistic",
        emission_mode="uncalibrated",
        feature_preprocessor="none",
        pca_components=None,
        classifier_param=1.0,
        temporal_bins=1,
        windows=(WindowSpec(center=0.15, width=0.05),),
    )

    real_probabilities = _predict_candidate(
        subjects=subjects,
        cache=FeatureCache(subjects),
        candidate=candidate,
        train_subjects=["0", "1"],
        test_subject="2",
        n_classes=2,
        max_iter=200,
    )

    def flip_binary_labels(labels, *, seed, context):
        del seed, context
        return 1 - np.asarray(labels, dtype=int)

    monkeypatch.setattr("neureptrace.bushmeg_source_loso._base._shuffle_training_labels", flip_binary_labels)
    shuffled_probabilities = _predict_candidate(
        subjects=subjects,
        cache=FeatureCache(subjects),
        candidate=candidate,
        train_subjects=["0", "1"],
        test_subject="2",
        n_classes=2,
        max_iter=200,
        label_shuffle_control=True,
        label_shuffle_seed=13,
        shuffle_context=("test",),
    )

    assert balanced_accuracy_score(labels, real_probabilities.argmax(axis=1)) == 1.0
    assert balanced_accuracy_score(labels, shuffled_probabilities.argmax(axis=1)) == 0.0


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


def test_select_candidate_carries_strict_alignment_metadata():
    subjects = {}
    times = np.array([0.10, 0.20])
    for subject_idx in range(4):
        labels = np.array([0, 0, 1, 1])
        data = np.zeros((4, 2, 2), dtype=np.float32)
        data[:, 0, :] = labels[:, None]
        data[:, 1, :] = subject_idx * 0.01
        subjects[str(subject_idx)] = SubjectEpochs(
            subject=str(subject_idx),
            data=data,
            times=times,
            metadata=pd.DataFrame(),
            labels=labels,
        )

    candidate = CandidateSpec(
        name="aligned_mean_bin",
        decoder="logistic",
        emission_mode="uncalibrated",
        feature_preprocessor="none",
        pca_components=None,
        classifier_param=1.0,
        temporal_bins=1,
        windows=(WindowSpec(center=0.15, width=0.20),),
    )
    alignment_diagnostic_rows: list[dict[str, object]] = []
    selected, rows, _summary = _select_candidate(
        subjects=subjects,
        cache=FeatureCache(subjects),
        candidates=[candidate],
        outer_test_subject="3",
        n_classes=2,
        max_iter=200,
        selection_metric="balanced_accuracy",
        alignment_config=source_alignment_config(method="procrustes", components=1),
        alignment_diagnostic_rows=alignment_diagnostic_rows,
    )

    assert selected.name == "aligned_mean_bin"
    assert {row["alignment_method"] for row in rows} == {"procrustes"}
    assert {row["alignment_target_projection"] for row in rows} == {"group_projection"}
    diagnostics = pd.DataFrame(alignment_diagnostic_rows)
    assert diagnostics["dataset"].unique().tolist() == ["BUSH-MEG"]
    assert set(diagnostics["test_subject"]) == {"0", "1", "2"}
    assert diagnostics["alignment_method"].unique().tolist() == ["procrustes"]
    assert diagnostics["actual_components"].unique().tolist() == [1]
    assert diagnostics["feature_dim"].unique().tolist() == [2]
    assert diagnostics["decode_feature_dim"].unique().tolist() == [1]
    assert "source_inner_raw_balanced_accuracy" in diagnostics.columns
    assert diagnostics["source_inner_validation_type"].unique().tolist() == [
        "strict_source_loso_nearest_centroid_group_projection"
    ]
    assert diagnostics["target_transform_type"].unique().tolist() == ["source_group_projection"]


def test_run_bushmeg_source_loso_oracle_sidecar_marks_target_labels(tmp_path, monkeypatch):
    subjects = {}
    times = np.array([0.10, 0.20])
    for subject_idx in range(4):
        labels = np.array([0, 0, 1, 1])
        data = np.zeros((4, 2, 2), dtype=np.float32)
        data[:, 0, :] = labels[:, None]
        data[:, 1, :] = subject_idx * 0.02
        subjects[str(subject_idx)] = SubjectEpochs(
            subject=str(subject_idx),
            data=data,
            times=times,
            metadata=pd.DataFrame({"participant": [str(subject_idx)] * 4}),
            labels=labels,
        )
    encoder = LabelEncoder().fit(["face", "scrambled"])
    monkeypatch.setattr(
        "neureptrace.bushmeg_source_loso._load_subjects_from_config",
        lambda *_args, **_kwargs: (subjects, encoder),
    )

    config_path = tmp_path / "bush.yml"
    config_path.write_text(
        """
preprocessing:
  window_size: 0.20
  epoch_normalization: none
decoding:
  max_iter: 200
source_loso:
  selection_metric: balanced_accuracy
  alignment_method: procrustes
  alignment_components: 1
  alignment_times: same_decode_window
  alignment_target_projection: oracle_target_calibrated_alignment
  candidate_grid:
    decoders: [logistic]
    emission_modes: [uncalibrated]
    feature_preprocessors: [none]
    pca_components: [none]
    temporal_bins: [1]
    c_grid: [1.0]
    window_sets:
      - name: single
        centers: [0.15]
        window_size: 0.20
""",
        encoding="utf-8",
    )
    out = tmp_path / "summary.csv"

    run_bushmeg_source_loso(config_path, out_path=out)

    sidecar = json.loads((tmp_path / "summary.csv.provenance.json").read_text(encoding="utf-8"))
    assert sidecar["alignment_target_labels_used"] is True
    assert sidecar["alignment_target_anchor_values_used"] is False
    assert sidecar["source_alignment"]["alignment_target_projection"] == "oracle_target_calibrated_alignment"
    assert sidecar["source_alignment"]["alignment_valid_for_benchmark"] is False


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


def test_sample_weights_can_apply_cue_subject_multipliers():
    times = np.array([0.10])
    subjects = {
        "s1": SubjectEpochs(
            subject="s1",
            data=np.zeros((2, 1, 1), dtype=np.float32),
            times=times,
            metadata=pd.DataFrame(),
            labels=np.array([0, 1]),
        ),
        "s2": SubjectEpochs(
            subject="s2",
            data=np.zeros((2, 1, 1), dtype=np.float32),
            times=times,
            metadata=pd.DataFrame(),
            labels=np.array([0, 1]),
        ),
    }
    labels = np.concatenate([subjects["s1"].labels, subjects["s2"].labels])

    weights = _sample_weights_for_training(subjects, ["s1", "s2"], labels, "none", subject_weight_multipliers={"s1": 2.0, "s2": 0.5})

    assert weights is not None
    assert np.isclose(weights.mean(), 1.0)
    assert float(weights[:2].mean()) > float(weights[2:].mean())


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

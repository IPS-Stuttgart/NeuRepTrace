import numpy as np
import pytest

from neureptrace.decoding.group_generalization import (
    DecoderCandidate,
    make_decoder_candidate_grid,
    normalize_group_generalization_metric,
    run_group_generalization_benchmark,
    summarize_group_generalization,
)


def _linearly_decodable_group_data():
    rows = []
    labels = []
    groups = []
    for group in range(4):
        for trial in range(10):
            label = trial % 2
            labels.append(label)
            groups.append(group)
            rows.append([2.0 * label + 0.05 * group, float(trial % 5), -0.1 * group])
    return np.asarray(rows), np.asarray(labels), np.asarray(groups)


def test_nested_group_generalization_selects_candidate_and_scores_outer_groups():
    features, labels, groups = _linearly_decodable_group_data()
    candidates = (
        DecoderCandidate(name="dummy", decoder="mostFrequentDummy", emission_mode="uncalibrated"),
        DecoderCandidate(name="signal", decoder="logistic", emission_mode="uncalibrated", max_iter=2000),
    )

    result = run_group_generalization_benchmark(features, labels, groups, candidates, selection_metric="balanced-accuracy")

    assert result.outer.shape[0] == 4
    assert result.inner_validation.shape[0] == 24
    assert result.selected["candidate_name"].tolist() == ["signal"] * 4
    assert result.outer["balanced_accuracy"].min() == pytest.approx(1.0)
    assert result.predictions.shape[0] == features.shape[0]
    assert result.predictions["correct"].all()
    assert result.predictions["top2_correct"].all()


def test_group_generalization_can_score_requested_outer_subset_without_loader_assumptions():
    features, labels, groups = _linearly_decodable_group_data()

    result = run_group_generalization_benchmark(
        features,
        labels,
        groups,
        [DecoderCandidate(name="signal", decoder="logistic", emission_mode="uncalibrated", max_iter=2000)],
        outer_groups=[1, 3],
        sample_ids=[f"sample-{index}" for index in range(features.shape[0])],
    )

    assert result.outer["outer_group"].tolist() == [1, 3]
    assert set(result.predictions["test_group"]) == {1, 3}
    assert result.predictions["sample_id"].str.startswith("sample-").all()


def test_group_generalization_summary_reports_outer_fold_mean():
    features, labels, groups = _linearly_decodable_group_data()
    result = run_group_generalization_benchmark(features, labels, groups, ["logistic"])

    summary = summarize_group_generalization(result)

    assert summary.loc[0, "metric"] == "balanced_accuracy"
    assert summary.loc[0, "n_outer_folds"] == 4
    assert summary.loc[0, "balanced_accuracy_mean"] == pytest.approx(1.0)


def test_make_decoder_candidate_grid_expands_neutral_decoder_grid():
    candidates = make_decoder_candidate_grid(decoders=("logistic", "linear-svm"), emission_modes="uncalibrated", feature_preprocessors="none")

    assert [candidate.decoder for candidate in candidates] == ["logistic", "linear_svm"]
    assert [candidate.emission_mode for candidate in candidates] == ["uncalibrated", "uncalibrated"]
    assert all(candidate.feature_preprocessor == "none" for candidate in candidates)


def test_group_generalization_validates_row_alignment_and_metric_names():
    features, labels, groups = _linearly_decodable_group_data()

    with pytest.raises(ValueError, match="labels"):
        run_group_generalization_benchmark(features, labels[:-1], groups, ["logistic"])
    with pytest.raises(ValueError, match="Unknown group-generalization metric"):
        normalize_group_generalization_metric("roc_auc")

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.probability_stacking import (
    DEFAULT_OUTPUT_DECODER,
    _top_k_accuracy,
    class_balanced_sample_weights,
    fit_stacking_weights,
    fit_source_oof_stacking,
    main,
    stack_probability_observations,
    summarize_stacked_metrics,
)


def _probabilities_for(label: int, *, candidate: str) -> tuple[float, float]:
    if candidate == "strong":
        return (0.90, 0.10) if label == 0 else (0.10, 0.90)
    if candidate == "weak":
        return (0.40, 0.60) if label == 0 else (0.60, 0.40)
    raise AssertionError(candidate)


def _observation_rows(*, subject: str, labels: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for decoder in ("weak", "strong"):
        for sample_index, true_label in enumerate(labels):
            prob_0, prob_1 = _probabilities_for(true_label, candidate=decoder)
            rows.append(
                {
                    "subject": subject,
                    "fold": sample_index % 2,
                    "split_id": "split-0",
                    "seed": 7,
                    "decoder": decoder,
                    "backend": "sklearn",
                    "emission_mode": "calibrated",
                    "train_time": 0.10,
                    "test_time": 0.10,
                    "time": 0.10,
                    "window_start": 0.05,
                    "window_stop": 0.15,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": int(true_label),
                    "true_class": "zero" if true_label == 0 else "one",
                    "class_0": "zero",
                    "class_1": "one",
                    "prob_class_0": prob_0,
                    "prob_class_1": prob_1,
                }
            )
    return pd.DataFrame(rows)


def _three_class_observation_rows(*, subject: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    labels = [0, 1, 2, 0, 1, 2]
    for decoder in ("left_confound", "right_confound"):
        for sample_index, true_label in enumerate(labels):
            probabilities = np.full(3, 0.1)
            probabilities[true_label] = 0.6
            if decoder == "left_confound":
                probabilities[(true_label + 1) % 3] = 0.3
            else:
                probabilities[(true_label + 2) % 3] = 0.3
            rows.append(
                {
                    "subject": subject,
                    "fold": sample_index % 2,
                    "split_id": "split-0",
                    "seed": 7,
                    "decoder": decoder,
                    "backend": "sklearn",
                    "emission_mode": "calibrated",
                    "train_time": 0.10,
                    "test_time": 0.10,
                    "time": 0.10,
                    "window_start": 0.05,
                    "window_stop": 0.15,
                    "sample_index": sample_index,
                    "sequence_id": sample_index,
                    "true_label": int(true_label),
                    "true_class": str(true_label),
                    "class_0": "zero",
                    "class_1": "one",
                    "class_2": "two",
                    "prob_class_0": float(probabilities[0]),
                    "prob_class_1": float(probabilities[1]),
                    "prob_class_2": float(probabilities[2]),
                }
            )
    return pd.DataFrame(rows)


def _weights_from_output(stacked: pd.DataFrame) -> list[float]:
    return [float(value) for value in str(stacked["source_oof_weights"].iloc[0]).split("|")]


def test_fit_source_oof_stacking_prefers_better_source_candidate() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    weak = source.loc[source["decoder"] == "weak", ["prob_class_0", "prob_class_1"]].to_numpy(dtype=float)
    strong = source.loc[source["decoder"] == "strong", ["prob_class_0", "prob_class_1"]].to_numpy(dtype=float)
    labels = source.loc[source["decoder"] == "weak", "true_label"].to_numpy(dtype=int)

    fit = fit_source_oof_stacking(
        np.stack([weak, strong], axis=0),
        labels,
        candidates=("weak", "strong"),
        weighting="stacked",
        max_iter=120,
    )

    assert fit.candidates == ("weak", "strong")
    assert fit.pooling == "linear"
    assert np.isclose(sum(fit.weights), 1.0)
    assert fit.weights[1] > 0.80
    assert fit.source_oof_balanced_accuracy == 1.0


def test_fit_source_oof_stacking_auto_selects_log_pooling_when_source_loss_improves() -> None:
    source = _three_class_observation_rows(subject="source")
    left = source.loc[source["decoder"] == "left_confound", ["prob_class_0", "prob_class_1", "prob_class_2"]].to_numpy(dtype=float)
    right = source.loc[source["decoder"] == "right_confound", ["prob_class_0", "prob_class_1", "prob_class_2"]].to_numpy(dtype=float)
    labels = source.loc[source["decoder"] == "left_confound", "true_label"].to_numpy(dtype=int)

    linear = fit_source_oof_stacking(np.stack([left, right], axis=0), labels, candidates=("left_confound", "right_confound"), weighting="uniform", pooling="linear")
    log = fit_source_oof_stacking(np.stack([left, right], axis=0), labels, candidates=("left_confound", "right_confound"), weighting="uniform", pooling="log")
    auto = fit_source_oof_stacking(np.stack([left, right], axis=0), labels, candidates=("left_confound", "right_confound"), weighting="uniform", pooling="auto")

    assert log.source_oof_log_loss < linear.source_oof_log_loss
    assert auto.pooling == "log"
    assert np.isclose(auto.source_oof_log_loss, log.source_oof_log_loss)


def test_fit_source_oof_stacking_rejects_fractional_source_labels() -> None:
    cube = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.6, 0.4], [0.4, 0.6]],
        ]
    )

    with pytest.raises(ValueError, match="source_labels values must be integer-valued"):
        fit_source_oof_stacking(cube, [0.0, 0.5], candidates=("strong", "weak"))


def test_stacked_top_k_accuracy_rejects_fractional_k() -> None:
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6]])
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="k must be a positive integer"):
        _top_k_accuracy(probabilities, labels, k=1.5)

    with pytest.raises(ValueError, match="k must be a positive integer"):
        _top_k_accuracy(probabilities, labels, k=True)


def test_fit_stacking_weights_rejects_invalid_max_iter() -> None:
    cube = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.6, 0.4], [0.4, 0.6]],
        ]
    )
    labels = np.array([0, 1])

    for value in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="max_iter must be a positive integer"):
            fit_stacking_weights(cube, labels, max_iter=value)


def test_class_balanced_sample_weights_rejects_fractional_n_classes() -> None:
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="n_classes must be a positive integer"):
        class_balanced_sample_weights(labels, n_classes=2.5)

    with pytest.raises(ValueError, match="n_classes must be a positive integer"):
        class_balanced_sample_weights(labels, n_classes=True)


def test_fit_stacking_weights_rejects_bool_learning_rate() -> None:
    cube = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.6, 0.4], [0.4, 0.6]],
        ]
    )
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="learning_rate must be positive and finite"):
        fit_stacking_weights(cube, labels, learning_rate=True)


def test_fit_source_oof_stacking_rejects_bool_softmax_temperature() -> None:
    cube = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.6, 0.4], [0.4, 0.6]],
        ]
    )
    labels = np.array([0, 1])

    with pytest.raises(ValueError, match="temperature must be positive and finite"):
        fit_source_oof_stacking(cube, labels, candidates=("strong", "weak"), weighting="softmax", temperature=True)


def test_stack_probability_observations_applies_source_weights_to_target() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0])

    stacked = stack_probability_observations(source, target, weighting="stacked", max_iter=120)

    assert stacked["decoder"].unique().tolist() == [DEFAULT_OUTPUT_DECODER]
    assert stacked["backend"].unique().tolist() == ["source_oof_stacking"]
    assert stacked["source_oof_candidates"].unique().tolist() == ["weak|strong"]
    assert stacked["source_oof_pooling"].unique().tolist() == ["linear"]
    assert np.allclose(stacked[["prob_class_0", "prob_class_1"]].sum(axis=1), 1.0)
    assert _weights_from_output(stacked)[1] > 0.80
    assert stacked["predicted_label"].tolist() == [0, 1, 0]
    assert stacked["is_correct"].tolist() == [True, True, True]


def test_stack_probability_observations_records_auto_selected_pooling() -> None:
    source = _three_class_observation_rows(subject="source")
    target = _three_class_observation_rows(subject="target")

    stacked = stack_probability_observations(source, target, weighting="uniform", pooling="auto")
    metrics = summarize_stacked_metrics(stacked)

    assert stacked["source_oof_pooling"].unique().tolist() == ["log"]
    assert set(metrics["source_oof_pooling"]) == {"log"}
    assert {"top2_accuracy", "top3_accuracy", "brier", "ece"}.issubset(metrics.columns)
    assert metrics["top2_accuracy"].tolist() == [1.0] * len(metrics)
    assert metrics["top3_accuracy"].tolist() == [1.0] * len(metrics)
    assert np.all(np.isfinite(metrics["brier"]))
    assert np.all(np.isfinite(metrics["ece"]))
    assert np.allclose(stacked[["prob_class_0", "prob_class_1", "prob_class_2"]].sum(axis=1), 1.0)


def test_stack_probability_observations_allows_unlabeled_targets() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0]).drop(columns=["true_label", "true_class"])

    stacked = stack_probability_observations(source, target, weighting="stacked", max_iter=120)

    assert stacked["decoder"].unique().tolist() == [DEFAULT_OUTPUT_DECODER]
    assert np.allclose(stacked[["prob_class_0", "prob_class_1"]].sum(axis=1), 1.0)
    assert _weights_from_output(stacked)[1] > 0.80
    assert stacked["predicted_label"].tolist() == [0, 1, 0]
    assert stacked["predicted_class"].tolist() == ["zero", "one", "zero"]
    assert stacked["true_label"].astype(str).tolist() == ["", "", ""]
    assert stacked["probability_true_class"].astype(str).tolist() == ["", "", ""]
    assert stacked["is_correct"].astype(str).tolist() == ["", "", ""]


def test_stack_probability_observations_rejects_fractional_target_labels() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0])
    target["true_label"] = target["true_label"].astype(float)
    target.loc[target["sample_index"] == 1, "true_label"] = 0.5

    with pytest.raises(ValueError, match="target true_label values must be integer-valued"):
        stack_probability_observations(source, target, weighting="stacked", max_iter=120)


def test_stack_probability_observations_rejects_negative_probability_values() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0])
    source.loc[0, "prob_class_0"] = -0.1

    with pytest.raises(ValueError, match="non-negative"):
        stack_probability_observations(source, target, weighting="stacked", max_iter=120)


def test_stack_probability_observations_rejects_probability_values_above_one() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0])
    source.loc[0, "prob_class_0"] = 1.2

    with pytest.raises(ValueError, match="must not exceed 1.0"):
        stack_probability_observations(source, target, weighting="stacked", max_iter=120)


def test_stack_probability_observations_rejects_unnormalized_probability_rows() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0])
    source.loc[0, ["prob_class_0", "prob_class_1"]] = [0.2, 0.2]

    with pytest.raises(ValueError, match="must sum to 1.0"):
        stack_probability_observations(source, target, weighting="stacked", max_iter=120)


def test_stack_probability_observations_rejects_source_label_mismatch_with_custom_alignment_keys() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0])
    mask = (source["decoder"] == "strong") & (source["sample_index"] == 1)
    source.loc[mask, "true_label"] = 0
    source.loc[mask, "true_class"] = "zero"

    with pytest.raises(ValueError, match="inconsistent 'true_label'"):
        stack_probability_observations(
            source,
            target,
            alignment_columns=["subject", "sample_index"],
            weighting="stacked",
            max_iter=120,
        )


def test_stack_probability_observations_rejects_target_label_mismatch_with_custom_alignment_keys() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0])
    mask = (target["decoder"] == "strong") & (target["sample_index"] == 1)
    target.loc[mask, "true_label"] = 0
    target.loc[mask, "true_class"] = "zero"

    with pytest.raises(ValueError, match="inconsistent 'true_label'"):
        stack_probability_observations(
            source,
            target,
            alignment_columns=["subject", "sample_index"],
            weighting="stacked",
            max_iter=120,
        )


def test_summarize_stacked_metrics_rejects_fractional_true_labels() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0])
    stacked = stack_probability_observations(source, target, weighting="stacked", max_iter=120)
    stacked["true_label"] = stacked["true_label"].astype(float)
    stacked.loc[0, "true_label"] = 0.5

    with pytest.raises(ValueError, match="true_label values must be integer-valued"):
        summarize_stacked_metrics(stacked)


def test_target_labels_do_not_affect_fitted_source_weights() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0])
    flipped_target = target.copy()
    flipped_target["true_label"] = 1 - flipped_target["true_label"].astype(int)
    flipped_target["true_class"] = flipped_target["true_label"].map({0: "zero", 1: "one"})

    stacked = stack_probability_observations(source, target, weighting="stacked", max_iter=120)
    flipped = stack_probability_observations(source, flipped_target, weighting="stacked", max_iter=120)

    assert stacked["source_oof_weights"].unique().tolist() == flipped["source_oof_weights"].unique().tolist()
    assert stacked["model_hash"].unique().tolist() == flipped["model_hash"].unique().tolist()


def test_stack_probability_observations_rejects_misaligned_candidates() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0])
    drop_index = target.loc[(target["decoder"] == "strong") & (target["sample_index"] == 2)].index[0]
    misaligned = target.drop(index=drop_index)

    with pytest.raises(ValueError, match="align one-to-one"):
        stack_probability_observations(source, misaligned)


def test_probability_stacking_cli_writes_observations_and_metrics(tmp_path: Path) -> None:
    source_path = tmp_path / "source.csv"
    target_path = tmp_path / "target.csv"
    stacked_path = tmp_path / "stacked.csv"
    metrics_path = tmp_path / "metrics.csv"
    _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1]).to_csv(source_path, index=False)
    _observation_rows(subject="target", labels=[0, 1, 0]).to_csv(target_path, index=False)

    exit_code = main(
        [
            "--source-oof",
            str(source_path),
            "--target",
            str(target_path),
            "--out",
            str(stacked_path),
            "--metrics-out",
            str(metrics_path),
            "--max-iter",
            "120",
        ]
    )

    assert exit_code == 0
    stacked = pd.read_csv(stacked_path)
    metrics = pd.read_csv(metrics_path)
    assert stacked["decoder"].unique().tolist() == [DEFAULT_OUTPUT_DECODER]
    assert _weights_from_output(stacked)[1] > 0.80
    assert not metrics.empty
    assert "source_oof_weights" in metrics.columns
    assert {"top2_accuracy", "top3_accuracy", "brier", "ece"}.issubset(metrics.columns)

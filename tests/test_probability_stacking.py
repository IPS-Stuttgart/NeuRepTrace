from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.probability_stacking import (
    DEFAULT_OUTPUT_DECODER,
    fit_source_oof_stacking,
    main,
    stack_probability_observations,
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
    assert np.isclose(sum(fit.weights), 1.0)
    assert fit.weights[1] > 0.80
    assert fit.source_oof_balanced_accuracy == 1.0


def test_stack_probability_observations_applies_source_weights_to_target() -> None:
    source = _observation_rows(subject="source", labels=[0, 1, 0, 1, 0, 1])
    target = _observation_rows(subject="target", labels=[0, 1, 0])

    stacked = stack_probability_observations(source, target, weighting="stacked", max_iter=120)

    assert stacked["decoder"].unique().tolist() == [DEFAULT_OUTPUT_DECODER]
    assert stacked["backend"].unique().tolist() == ["source_oof_stacking"]
    assert stacked["source_oof_candidates"].unique().tolist() == ["weak|strong"]
    assert np.allclose(stacked[["prob_class_0", "prob_class_1"]].sum(axis=1), 1.0)
    assert _weights_from_output(stacked)[1] > 0.80
    assert stacked["predicted_label"].tolist() == [0, 1, 0]
    assert stacked["is_correct"].tolist() == [True, True, True]


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

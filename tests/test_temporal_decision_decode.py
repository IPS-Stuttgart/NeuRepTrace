from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.io.dataset import EpochDataset
from neureptrace.temporal_decision_decode import _combine, _decoders, run_temporal_decision_decode_dataset, run_temporal_decision_decode_from_config


def _synthetic_grouped_dataset() -> EpochDataset:
    rng = np.random.default_rng(13)
    times = np.arange(-0.1, 0.31, 0.05)
    n_groups = 3
    n_classes = 3
    trials_per_class_and_group = 2
    n_channels = 4
    rows = []
    labels = []
    for group in range(n_groups):
        for class_label in range(n_classes):
            for _trial in range(trials_per_class_and_group):
                rows.append({"stimulus": class_label, "participant": f"p{group}"})
                labels.append(class_label)
    data = rng.normal(scale=0.05, size=(len(rows), n_channels, len(times)))
    decision_mask = (times >= 0.05) & (times <= 0.2)
    patterns = np.eye(n_classes, n_channels)
    for trial_index, class_label in enumerate(labels):
        data[trial_index, :, decision_mask] += patterns[class_label] * 1.5
    return EpochDataset(
        data=data,
        times=times,
        channel_names=[f"MEG{channel}" for channel in range(n_channels)],
        metadata=pd.DataFrame(rows),
        name="synthetic_bush_like",
    )


def _with_metadata(dataset: EpochDataset, metadata: pd.DataFrame) -> EpochDataset:
    return EpochDataset(
        data=dataset.data,
        times=dataset.times,
        channel_names=dataset.channel_names,
        metadata=metadata,
        name=dataset.name,
    )


def test_decoders_supports_logistic_svm_ensemble_alias():
    assert _decoders("logistic-svm-ensemble") == ("multinomial-logistic", "linear_svm")


def test_temporal_decision_combine_rejects_invalid_source_probabilities():
    valid = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=float)
    invalid = np.asarray([[1.2, -0.2], [0.1, 0.1]], dtype=float)

    with pytest.raises(ValueError, match="must be non-negative"):
        _combine([valid, invalid], mode="log_mean", min_probability=1e-12)


def test_temporal_decision_combine_rejects_unnormalized_source_probabilities():
    valid = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=float)
    invalid = np.asarray([[0.4, 0.4], [0.1, 0.1]], dtype=float)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        _combine([valid, invalid], mode="probability", min_probability=1e-12)


def test_temporal_decision_combine_rejects_invalid_min_probability():
    valid = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=float)

    with pytest.raises(ValueError, match="min_probability"):
        _combine([valid], mode="log_mean", min_probability=0.0)


def test_temporal_decision_combine_rejects_unknown_aggregation():
    valid = np.asarray([[0.7, 0.3], [0.2, 0.8]], dtype=float)

    with pytest.raises(ValueError, match="aggregation"):
        _combine([valid], mode="logmean_typo", min_probability=1e-12)


def test_temporal_decision_decode_runs_leave_one_group_out(tmp_path):
    dataset = _synthetic_grouped_dataset()
    summary_out = tmp_path / "summary.csv"
    observations_out = tmp_path / "observations.csv"

    results = run_temporal_decision_decode_dataset(
        dataset,
        label_column="stimulus",
        group_column="participant",
        out_path=summary_out,
        observation_out_path=observations_out,
        decoders=["logistic"],
        window_ms=100.0,
        step_ms=50.0,
        test_window=(0.075, 0.175),
        emission_mode="uncalibrated",
        max_iter=1000,
    )

    assert summary_out.exists()
    assert observations_out.exists()
    assert len(results) == 3
    assert set(results["heldout_group"]) == {"p0", "p1", "p2"}
    assert results["split_id"].unique().tolist() == ["leave-one-group-out"]
    assert results["n_test"].tolist() == [6, 6, 6]
    assert results["n_test_windows"].tolist() == [3, 3, 3]
    assert results["balanced_accuracy"].min() > 0.9

    observations = pd.read_csv(observations_out)
    assert len(observations) == 18
    probability_columns = [column for column in observations.columns if column.startswith("prob_class_")]
    assert len(probability_columns) == 3
    np.testing.assert_allclose(observations[probability_columns].sum(axis=1), 1.0, atol=1e-8)


def test_temporal_decision_decode_rejects_empty_labeled_grouped_subset(tmp_path: Path) -> None:
    dataset = _synthetic_grouped_dataset()
    metadata = dataset.metadata.copy()
    metadata["stimulus"] = np.nan
    dataset = _with_metadata(dataset, metadata)

    with pytest.raises(ValueError, match="no rows with non-missing"):
        run_temporal_decision_decode_dataset(
            dataset,
            label_column="stimulus",
            group_column="participant",
            out_path=tmp_path / "summary.csv",
            emission_mode="uncalibrated",
        )


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("stimulus", 0, "at least two classes"),
        ("participant", "p0", "at least two groups"),
    ],
)
def test_temporal_decision_decode_rejects_degenerate_loso_inputs(tmp_path: Path, column: str, value: object, message: str) -> None:
    dataset = _synthetic_grouped_dataset()
    metadata = dataset.metadata.copy()
    metadata[column] = value
    dataset = _with_metadata(dataset, metadata)

    with pytest.raises(ValueError, match=message):
        run_temporal_decision_decode_dataset(
            dataset,
            label_column="stimulus",
            group_column="participant",
            out_path=tmp_path / "summary.csv",
            emission_mode="uncalibrated",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_ms": True}, "window_ms"),
        ({"step_ms": np.inf}, "step_ms"),
        ({"tmin": True}, "tmin"),
        ({"tmax": np.inf}, "tmax"),
        ({"tmin": 0.2, "tmax": 0.1}, "tmax"),
        ({"test_window": [True, 0.1]}, "test_window"),
        ({"test_window": (0.2, 0.1)}, "test_window"),
        ({"max_iter": True}, "max_iter"),
        ({"max_iter": 100.5}, "max_iter"),
        ({"aggregation": "logmean_typo"}, "aggregation"),
        ({"min_probability": True}, "min_probability"),
        ({"calibration_bins": 0}, "calibration_bins"),
    ],
)
def test_temporal_decision_decode_rejects_malformed_dataset_controls(tmp_path: Path, kwargs: dict, message: str):
    dataset = _synthetic_grouped_dataset()

    with pytest.raises(ValueError, match=message):
        run_temporal_decision_decode_dataset(
            dataset,
            label_column="stimulus",
            group_column="participant",
            out_path=tmp_path / "summary.csv",
            emission_mode="uncalibrated",
            **kwargs,
        )


def _temporal_decision_config(
    tmp_path: Path,
    *,
    preprocessing: dict | None = None,
    decoding: dict | None = None,
    temporal: dict | None = None,
) -> dict:
    return {
        "dataset": {"name": "synthetic_temporal_decision"},
        "preprocessing": {"window_ms": 100.0, "step_ms": 50.0, **(preprocessing or {})},
        "decoding": {
            "label_column": "stimulus",
            "group_column": "participant",
            "emission_mode": "uncalibrated",
            **(decoding or {}),
        },
        "temporal_decision": temporal or {},
        "outputs": {"summary_csv": str(tmp_path / "summary.csv"), "provenance": False},
    }


@pytest.mark.parametrize(
    ("preprocessing", "decoding", "temporal", "message"),
    [
        ({}, {}, {"max_iter": True}, "temporal_decision.max_iter"),
        ({}, {"max_iter": 100.5}, {}, "temporal_decision.max_iter"),
        ({}, {}, {"min_probability": True}, "min_probability"),
        ({}, {}, {"calibration_bins": 1.5}, "temporal_decision.calibration_bins"),
        ({}, {"calibration_bins": 0}, {}, "temporal_decision.calibration_bins"),
        ({}, {}, {"test_window": [True, 0.1]}, "temporal_decision.test_window"),
        ({"window_ms": True}, {}, {}, "preprocessing.window_ms"),
        ({"step_ms": 0}, {}, {}, "preprocessing.step_ms"),
        ({"tmin": True}, {}, {}, "tmin"),
        ({"tmax": np.inf}, {}, {}, "tmax"),
    ],
)
def test_temporal_decision_decode_from_config_rejects_malformed_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    preprocessing: dict,
    decoding: dict,
    temporal: dict,
    message: str,
) -> None:
    config = _temporal_decision_config(tmp_path, preprocessing=preprocessing, decoding=decoding, temporal=temporal)
    monkeypatch.setattr("neureptrace.temporal_decision_decode.load_config", lambda _path: config)
    monkeypatch.setattr("neureptrace.temporal_decision_decode.load_epoch_dataset_from_config", lambda *_args, **_kwargs: _synthetic_grouped_dataset())

    with pytest.raises(ValueError, match=message):
        run_temporal_decision_decode_from_config(tmp_path / "config.yml")


def test_temporal_decision_decode_from_config_rejects_malformed_write_provenance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _temporal_decision_config(tmp_path)
    monkeypatch.setattr("neureptrace.temporal_decision_decode.load_config", lambda _path: config)

    with pytest.raises(ValueError, match="write_provenance"):
        run_temporal_decision_decode_from_config(tmp_path / "config.yml", write_provenance="maybe")

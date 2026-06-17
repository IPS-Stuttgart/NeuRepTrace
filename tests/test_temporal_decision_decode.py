from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.io.dataset import EpochDataset
import pytest

from neureptrace.temporal_decision_decode import _combine, _decoders, run_temporal_decision_decode_dataset


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

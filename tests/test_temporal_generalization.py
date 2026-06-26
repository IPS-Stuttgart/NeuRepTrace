import numpy as np
import pandas as pd
import pytest

from neureptrace.decoding import make_decoder
from neureptrace.decoding.temporal_generalization import (
    TemporalFeatureWindow,
    compute_temporal_generalization_matrix,
    summarize_temporal_generalization_matrix,
)


def _window(center, values, labels):
    return TemporalFeatureWindow(
        center=center,
        start=center - 0.05,
        stop=center + 0.05,
        features=np.asarray(values, dtype=float).reshape(-1, 1),
        labels=np.asarray(labels, dtype=int),
    )


def test_temporal_generalization_matrix_scores_all_train_test_pairs():
    train_windows = [
        _window(0.1, [-2.0, -1.0, 1.0, 2.0], [0, 0, 1, 1]),
        _window(0.0, [-2.0, -1.0, 1.0, 2.0], [1, 1, 0, 0]),
    ]
    test_windows = [
        _window(0.0, [-1.5, 1.5], [1, 0]),
        _window(0.1, [-1.5, 1.5], [0, 1]),
    ]

    rows = compute_temporal_generalization_matrix(
        train_windows,
        test_windows,
        fit_model=lambda window: make_decoder("logistic", max_iter=2000).fit(window.features, window.labels),
        predict_labels=lambda model, window: model.predict(window.features),
        chance_accuracy=0.5,
        metadata={"participant": "S01", "decoder": "logistic"},
    )

    assert rows["train_window_center_s"].tolist() == [0.0, 0.0, 0.1, 0.1]
    assert rows["test_window_center_s"].tolist() == [0.0, 0.1, 0.0, 0.1]
    assert rows["is_diagonal"].tolist() == [True, False, False, True]
    assert rows["participant"].tolist() == ["S01"] * 4
    assert rows["decoder"].tolist() == ["logistic"] * 4
    assert rows["chance_accuracy"].tolist() == [0.5] * 4
    assert rows.loc[rows["is_diagonal"], "accuracy"].tolist() == [1.0, 1.0]
    assert rows.loc[~rows["is_diagonal"], "accuracy"].tolist() == [0.0, 0.0]


def test_temporal_generalization_preserves_composite_tuple_labels():
    train_window = TemporalFeatureWindow(
        center=0.0,
        features=np.zeros((3, 1)),
        labels=[("left", 1), ("right", 2), ("left", 1)],
    )
    test_window = TemporalFeatureWindow(
        center=0.0,
        features=np.zeros((3, 1)),
        labels=[("left", 1), ("right", 2), ("left", 1)],
    )

    rows = compute_temporal_generalization_matrix(
        [train_window],
        [test_window],
        fit_model=lambda window: window,
        predict_labels=lambda _model, _window: np.asarray([("left", 1), ("right", 2), ("left", 1)], dtype=object),
    )

    assert rows.loc[0, "accuracy"] == 1.0
    assert rows.loc[0, "chance_accuracy"] == 0.5
    assert rows.loc[0, "n_train_trials"] == 3
    assert rows.loc[0, "n_validation_trials"] == 3
    assert rows.loc[0, "n_train_classes"] == 2
    assert rows.loc[0, "n_validation_classes"] == 2


def test_temporal_generalization_rejects_matrix_label_arrays():
    invalid_train_window = TemporalFeatureWindow(
        center=0.0,
        features=np.zeros((2, 1)),
        labels=np.asarray([[0, 1], [1, 0]]),
    )

    with pytest.raises(ValueError, match="train window labels must be one-dimensional"):
        compute_temporal_generalization_matrix(
            [invalid_train_window],
            [_window(0.0, [-1.0, 1.0], [0, 1])],
            fit_model=lambda window: window,
            predict_labels=lambda _model, window: window.labels,
        )


def test_temporal_generalization_includes_model_metadata():
    train_windows = [_window(0.0, [-1.0, 1.0], [0, 1])]
    test_windows = [_window(0.0, [-1.0, 1.0], [0, 1])]

    rows = compute_temporal_generalization_matrix(
        train_windows,
        test_windows,
        fit_model=lambda window: {"classes": len(np.unique(window.labels))},
        predict_labels=lambda _model, window: window.labels,
        model_metadata=lambda model: {"n_model_classes": model["classes"]},
    )

    assert rows.loc[0, "n_model_classes"] == 2
    assert rows.loc[0, "accuracy"] == 1.0


def test_summarize_temporal_generalization_matrix_groups_rows():
    rows = compute_temporal_generalization_matrix(
        [
            _window(0.0, [-1.0, 1.0], [0, 1]),
            _window(0.1, [-1.0, 1.0], [0, 1]),
        ],
        [_window(0.0, [-1.0, 1.0], [0, 1])],
        fit_model=lambda window: window,
        predict_labels=lambda _model, window: window.labels,
        chance_accuracy=0.5,
        metadata={"decoder": "toy"},
    )

    summary = summarize_temporal_generalization_matrix(
        rows,
        group_columns=("decoder", "test_window_center_s"),
    )

    assert summary.to_dict("records") == [
        {
            "decoder": "toy",
            "test_window_center_s": 0.0,
            "n_rows": 2,
            "accuracy_mean": 1.0,
            "accuracy_median": 1.0,
            "accuracy_std": 0.0,
            "accuracy_sem": 0.0,
            "percent_mean": 100.0,
            "percent_median": 100.0,
            "percent_std": 0.0,
            "percent_sem": 0.0,
            "chance_accuracy": 0.5,
            "chance_percent": 50.0,
            "above_chance_count": 2,
            "is_diagonal": False,
        }
    ]


def test_summarize_temporal_generalization_matrix_accepts_single_group_column_string():
    rows = pd.DataFrame(
        {
            "decoder": ["toy", "toy", "other"],
            "accuracy": [1.0, 0.5, 0.0],
            "chance_accuracy": [0.5, 0.5, 0.5],
        }
    )

    summary = summarize_temporal_generalization_matrix(rows, group_columns="decoder")

    assert summary["decoder"].tolist() == ["other", "toy"]
    assert summary["n_rows"].tolist() == [1, 2]
    assert summary["accuracy_mean"].tolist() == [0.0, 0.75]
    assert summary["above_chance_count"].tolist() == [0, 1]


@pytest.mark.parametrize("invalid", [True, np.bool_(False), 1.5, np.nan])
def test_temporal_generalization_rejects_invalid_center_decimals(invalid):
    with pytest.raises(ValueError, match="center_decimals"):
        compute_temporal_generalization_matrix(
            [_window(0.0, [-1.0, 1.0], [0, 1])],
            [_window(0.0, [-1.0, 1.0], [0, 1])],
            fit_model=lambda window: window,
            predict_labels=lambda _model, window: window.labels,
            center_decimals=invalid,
        )


@pytest.mark.parametrize("invalid", [True, np.bool_(False), np.nan, -0.1, 1.1])
def test_temporal_generalization_rejects_invalid_chance_accuracy(invalid):
    with pytest.raises(ValueError, match="chance_accuracy"):
        compute_temporal_generalization_matrix(
            [_window(0.0, [-1.0, 1.0], [0, 1])],
            [_window(0.0, [-1.0, 1.0], [0, 1])],
            fit_model=lambda window: window,
            predict_labels=lambda _model, window: window.labels,
            chance_accuracy=invalid,
        )

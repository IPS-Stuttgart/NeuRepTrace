from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.svm import LinearSVC

from neureptrace.decoding.precomputed_foundation import (
    PrecomputedFoundationFeatureTable,
    align_precomputed_foundation_features,
    fit_precomputed_foundation_probe,
    load_precomputed_foundation_features,
    make_precomputed_foundation_feature_table,
    normalize_feature_fit_scope,
)


def test_npz_loader_aligns_requested_row_order(tmp_path) -> None:
    path = tmp_path / "features.npz"
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]], dtype=float)
    row_ids = np.asarray(["trial-c", "trial-a", "trial-b"])
    np.savez(path, features=features, row_ids=row_ids, feature_names=np.asarray(["f0", "f1"]))

    table = load_precomputed_foundation_features(path, source_model="BENDR")
    aligned = align_precomputed_foundation_features(table, ["trial-a", "trial-b", "trial-c"])

    assert table.n_rows == 3
    assert table.n_features == 2
    assert table.feature_names == ("f0", "f1")
    assert table.metadata["precomputed_foundation_source_model"] == "BENDR"
    assert table.metadata["precomputed_foundation_protocol_category"] == 1
    assert np.allclose(aligned, np.asarray([[0.0, 1.0], [2.0, 2.0], [1.0, 0.0]]))


def test_npy_loader_uses_sequential_row_ids(tmp_path) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float))

    table = load_precomputed_foundation_features(path)

    assert table.row_ids == (0, 1)
    assert np.allclose(align_precomputed_foundation_features(table, [1, 0]), np.asarray([[3.0, 4.0], [1.0, 2.0]]))


def test_csv_loader_selects_numeric_feature_columns(tmp_path) -> None:
    path = tmp_path / "features.csv"
    frame = pd.DataFrame(
        {
            "row_id": ["r1", "r2", "r3"],
            "label_like_text": ["a", "b", "c"],
            "feat_0": [1.0, 2.0, 3.0],
            "feat_1": [0.5, 0.25, 0.125],
        }
    )
    frame.to_csv(path, index=False)

    table = load_precomputed_foundation_features(path, feature_prefix="feat_", feature_fit_scope="unlabeled_target")

    assert table.row_ids == ("r1", "r2", "r3")
    assert table.feature_names == ("feat_0", "feat_1")
    assert table.metadata["precomputed_foundation_protocol_category"] == 2
    assert table.metadata["precomputed_foundation_uses_target_features_for_feature_fit"] is True
    assert table.metadata["precomputed_foundation_uses_target_labels_for_feature_fit"] is False


def test_duplicate_row_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="row_ids must be unique"):
        make_precomputed_foundation_feature_table([[0.0], [1.0]], row_ids=["dup", "dup"])


def test_missing_requested_row_id_is_rejected() -> None:
    table = make_precomputed_foundation_feature_table([[0.0], [1.0]], row_ids=["a", "b"])

    with pytest.raises(KeyError, match="missing"):
        align_precomputed_foundation_features(table, ["a", "missing"])


def test_fit_precomputed_foundation_probe_uses_only_train_labels() -> None:
    features = np.asarray(
        [
            [-2.0, 0.0],
            [-1.5, 0.2],
            [2.0, 0.0],
            [1.6, -0.2],
            [-1.8, 0.1],
            [1.8, -0.1],
        ],
        dtype=float,
    )
    row_ids = ["s0", "s1", "s2", "s3", "t0", "t1"]
    table = make_precomputed_foundation_feature_table(features, row_ids=row_ids, source_model="LaBraM")

    result = fit_precomputed_foundation_probe(
        feature_table=table,
        train_row_ids=["s0", "s1", "s2", "s3"],
        train_labels=["left", "left", "right", "right"],
        test_row_ids=["t0", "t1"],
    )

    assert result.predictions.tolist() == ["left", "right"]
    assert result.probabilities is not None
    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert result.metadata["precomputed_foundation_probe_uses_train_labels"] is True
    assert result.metadata["precomputed_foundation_probe_uses_target_labels"] is False
    assert result.metadata["precomputed_foundation_valid_for_strict_source_only"] is True


def test_fit_precomputed_foundation_probe_supports_decision_function_classifier() -> None:
    table = make_precomputed_foundation_feature_table(
        [[-2.0], [-1.0], [2.0], [1.0], [-1.5], [1.5]],
        row_ids=["a", "b", "c", "d", "e", "f"],
    )

    result = fit_precomputed_foundation_probe(
        feature_table=table,
        train_row_ids=["a", "b", "c", "d"],
        train_labels=[0, 0, 1, 1],
        test_row_ids=["e", "f"],
        classifier=LinearSVC(random_state=0),
    )

    assert result.predictions.tolist() == [0, 1]
    assert result.probabilities is not None
    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    margins = np.asarray(result.classifier.decision_function(result.test_features), dtype=float)
    expected_positive = 1.0 / (1.0 + np.exp(-margins))
    np.testing.assert_allclose(result.probabilities[:, 1], expected_positive)


def test_fit_precomputed_foundation_probe_rejects_target_labels_argument() -> None:
    table = make_precomputed_foundation_feature_table([[0.0], [1.0], [2.0]], row_ids=["a", "b", "c"])

    with pytest.raises(TypeError):
        fit_precomputed_foundation_probe(
            feature_table=table,
            train_row_ids=["a", "b"],
            train_labels=[0, 1],
            test_row_ids=["c"],
            target_labels=[1],  # type: ignore[call-arg]
        )


def test_feature_fit_scope_aliases_and_oracle_metadata() -> None:
    assert normalize_feature_fit_scope("category-2") == "source_plus_unlabeled_target"
    assert normalize_feature_fit_scope("oracle") == "oracle_target_included"

    table = make_precomputed_foundation_feature_table([[0.0], [1.0]], row_ids=["a", "b"], feature_fit_scope="oracle")
    assert table.metadata["precomputed_foundation_protocol_category"] == 4
    assert table.metadata["precomputed_foundation_debug_upper_bound"] is True


def test_table_constructor_validates_feature_names() -> None:
    with pytest.raises(ValueError, match="feature_names"):
        PrecomputedFoundationFeatureTable(features=np.asarray([[1.0, 2.0]]), row_ids=("r",), feature_names=("only_one",))

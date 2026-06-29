from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_quantile import (
    SOURCE_QUANTILE_CATEGORY,
    apply_source_quantile_bins,
    apply_source_quantile_clip,
    apply_source_quantile_rank,
    source_feature_quantiles,
    source_quantile_bins,
    source_quantile_clip,
    source_quantile_rank,
)


def test_source_feature_quantiles_are_source_only() -> None:
    features = np.asarray([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [100.0, 20.0]], dtype=float)

    lower, upper = source_feature_quantiles(features, lower=0.25, upper=0.75)

    assert SOURCE_QUANTILE_CATEGORY == "1_strict_source_only"
    assert np.allclose(lower, np.quantile(features, 0.25, axis=0))
    assert np.allclose(upper, np.quantile(features, 0.75, axis=0))


def test_source_quantile_clip_uses_source_bounds_for_test_rows() -> None:
    source = np.asarray([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]], dtype=float)
    test = np.asarray([[-5.0, 10.5], [10.0, 20.0]], dtype=float)

    result = source_quantile_clip(source_features=source, test_features=test, lower=0.25, upper=0.75)

    assert result.train_features.shape == source.shape
    assert result.test_features.shape == test.shape
    assert np.all(result.test_features >= result.lower)
    assert np.all(result.test_features <= result.upper)
    assert result.metadata["source_quantile_clip_protocol_category"] == SOURCE_QUANTILE_CATEGORY
    assert result.metadata["source_quantile_clip_uses_source_features"] is True
    assert result.metadata["source_quantile_clip_uses_test_features_for_fitting"] is False
    assert result.metadata["source_quantile_clip_uses_test_labels"] is False
    assert result.metadata["source_quantile_clip_test_values_clipped"] == int(np.count_nonzero(result.test_clipped_mask))


def test_source_quantile_rank_uses_source_reference_for_test_rows() -> None:
    source = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    test = np.asarray([[-1.0], [0.5], [2.5], [4.0]], dtype=float)

    result = source_quantile_rank(source_features=source, test_features=test, epsilon=0.01)

    assert result.train_features.shape == source.shape
    assert result.test_features.shape == test.shape
    assert np.all(result.test_features >= 0.01)
    assert np.all(result.test_features <= 0.99)
    assert np.allclose(result.sorted_source_values.ravel(), source.ravel())
    assert result.metadata["source_quantile_rank_protocol_category"] == SOURCE_QUANTILE_CATEGORY
    assert result.metadata["source_quantile_rank_uses_source_features"] is True
    assert result.metadata["source_quantile_rank_uses_test_features_for_fitting"] is False
    assert result.metadata["source_quantile_rank_uses_test_labels"] is False


def test_source_quantile_rank_centered_output_is_bounded() -> None:
    source = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    test = np.asarray([[0.0], [3.0]], dtype=float)

    result = source_quantile_rank(source_features=source, test_features=test, centered=True, epsilon=0.01)

    assert np.all(result.test_features >= -0.98)
    assert np.all(result.test_features <= 0.98)
    assert result.metadata["source_quantile_rank_centered"] is True


def test_source_quantile_bins_use_source_edges_for_test_rows() -> None:
    source = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    test = np.asarray([[-1.0], [0.5], [1.5], [4.0]], dtype=float)

    result = source_quantile_bins(source_features=source, test_features=test, n_bins=4)

    assert result.train_features.shape == source.shape
    assert result.test_features.shape == test.shape
    assert result.bin_edges.shape == (3, 1)
    assert result.train_features.dtype == np.int16
    assert result.test_features.tolist() == [[0], [0], [2], [3]]
    assert result.metadata["source_quantile_bins_protocol_category"] == SOURCE_QUANTILE_CATEGORY
    assert result.metadata["source_quantile_bins_uses_source_features"] is True
    assert result.metadata["source_quantile_bins_uses_test_features_for_fitting"] is False
    assert result.metadata["source_quantile_bins_uses_test_labels"] is False


def test_apply_source_quantile_bins_with_explicit_edges() -> None:
    edges = np.asarray([[1.0, 10.0], [2.0, 20.0]], dtype=float)

    bins = apply_source_quantile_bins([[0.0, 9.0], [1.0, 10.0], [3.0, 30.0]], bin_edges=edges)

    assert bins.tolist() == [[0, 0], [1, 1], [2, 2]]


def test_apply_source_quantile_bins_preserves_large_bin_indices() -> None:
    n_edges = np.iinfo(np.int16).max + 1
    edges = np.arange(n_edges, dtype=float).reshape(-1, 1)

    bins = apply_source_quantile_bins([[float(n_edges)]], bin_edges=edges)

    assert bins.dtype == np.int32
    assert bins.tolist() == [[n_edges]]


def test_apply_source_quantile_rank_handles_ties() -> None:
    sorted_values = np.asarray([[0.0], [1.0], [1.0], [2.0]], dtype=float)

    ranks = apply_source_quantile_rank([[1.0]], sorted_values=sorted_values, epsilon=0.01)

    assert np.allclose(ranks, np.asarray([[0.5]]))


def test_apply_source_quantile_clip_returns_mask() -> None:
    clipped, mask = apply_source_quantile_clip([[0.0, 5.0], [10.0, -5.0]], lower=[1.0, 0.0], upper=[9.0, 4.0])

    assert np.allclose(clipped, np.asarray([[1.0, 4.0], [9.0, 0.0]]))
    assert mask.tolist() == [[True, True], [True, True]]


def test_source_quantile_helpers_reject_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        source_quantile_clip(source_features=[[0.0, 1.0]], test_features=[[0.0]])
    with pytest.raises(ValueError, match="same feature width"):
        source_quantile_rank(source_features=[[0.0, 1.0]], test_features=[[0.0]])
    with pytest.raises(ValueError, match="same feature width"):
        source_quantile_bins(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_source_feature_quantiles_validate_bounds() -> None:
    with pytest.raises(ValueError, match="lower"):
        source_feature_quantiles([[0.0], [1.0]], lower=0.9, upper=0.1)


def test_source_quantile_rank_validates_epsilon() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        source_quantile_rank(source_features=[[0.0], [1.0]], test_features=[[0.5]], epsilon=0.5)


def test_source_quantile_bins_validate_n_bins() -> None:
    with pytest.raises(ValueError, match="n_bins"):
        source_quantile_bins(source_features=[[0.0], [1.0]], test_features=[[0.5]], n_bins=0)
    with pytest.raises(ValueError, match="bin_edges"):
        apply_source_quantile_bins([[0.5]], bin_edges=[0.0, 1.0])


def test_source_feature_quantiles_reject_boolean_bounds() -> None:
    with pytest.raises(ValueError, match="not boolean"):
        source_feature_quantiles([[0.0], [1.0]], lower=False, upper=True)
    with pytest.raises(ValueError, match="not boolean"):
        source_quantile_clip(source_features=[[0.0], [1.0]], test_features=[[0.5]], lower=0.0, upper=np.bool_(True))


def test_source_feature_quantiles_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        source_feature_quantiles([[0.0], [float("nan")]])


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        source_feature_quantiles([[0.0], [1.0]], heldout_features=[[0.5]])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        source_quantile_clip(source_features=[[0.0], [1.0]], test_features=[[0.5]], heldout_labels=[0])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        source_quantile_rank(source_features=[[0.0], [1.0]], test_features=[[0.5]], heldout_labels=[0])  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        source_quantile_bins(source_features=[[0.0], [1.0]], test_features=[[0.5]], heldout_labels=[0])  # type: ignore[call-arg]

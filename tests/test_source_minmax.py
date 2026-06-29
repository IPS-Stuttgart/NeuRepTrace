from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_minmax import (
    SOURCE_MINMAX_CATEGORY,
    apply_source_minmax_transform,
    fit_source_minmax_reference,
    fit_source_minmax_transform,
)


def test_source_minmax_transform_uses_source_bounds_only() -> None:
    source = np.asarray([[0.0, 10.0], [2.0, 20.0]], dtype=float)
    test = np.asarray([[1.0, 15.0], [4.0, 30.0]], dtype=float)

    result = fit_source_minmax_transform(source_features=source, test_features=test)

    assert np.allclose(result.train_features, np.asarray([[0.0, 0.0], [1.0, 1.0]]))
    assert np.allclose(result.test_features, np.asarray([[0.5, 0.5], [2.0, 2.0]]))
    assert result.metadata["source_minmax_protocol_category"] == SOURCE_MINMAX_CATEGORY
    assert result.metadata["source_minmax_uses_source_features"] is True
    assert result.metadata["source_minmax_uses_test_features_for_fitting"] is False
    assert result.metadata["source_minmax_uses_test_labels"] is False
    assert result.metadata["source_minmax_valid_for_strict_source_only"] is True


def test_source_minmax_custom_range() -> None:
    result = fit_source_minmax_transform(
        source_features=[[0.0], [2.0]],
        test_features=[[1.0]],
        feature_range=(-1.0, 1.0),
    )

    assert np.allclose(result.train_features.ravel(), np.asarray([-1.0, 1.0]))
    assert np.allclose(result.test_features.ravel(), np.asarray([0.0]))


def test_source_minmax_reference_can_be_reused() -> None:
    reference = fit_source_minmax_reference([[0.0], [4.0]])
    transformed = apply_source_minmax_transform([[2.0]], reference)

    assert np.allclose(transformed.ravel(), np.asarray([0.5]))


def test_source_minmax_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_minmax_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_source_minmax_rejects_bad_range() -> None:
    with pytest.raises(ValueError, match="feature_range"):
        fit_source_minmax_reference([[0.0], [1.0]], feature_range=(1.0, 0.0))


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_minmax_transform(
            source_features=[[0.0], [1.0]],
            test_features=[[0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )

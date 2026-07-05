from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_ecdf import (
    SOURCE_ECDF_CATEGORY,
    apply_source_ecdf_transform,
    fit_source_ecdf_reference,
    fit_source_ecdf_transform,
    normalize_ecdf_output,
    source_ecdf_config,
)


def test_source_ecdf_uniform_transform_shapes_and_metadata() -> None:
    source = np.asarray([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]], dtype=float)
    test = np.asarray([[1.5, 11.5], [4.0, 9.0]], dtype=float)

    result = fit_source_ecdf_transform(source_features=source, test_features=test, config={"output": "uniform"})

    assert result.train_features.shape == source.shape
    assert result.test_features.shape == test.shape
    assert np.all(result.train_features > 0.0)
    assert np.all(result.train_features < 1.0)
    assert result.metadata["source_ecdf_protocol_category"] == SOURCE_ECDF_CATEGORY
    assert result.metadata["source_ecdf_uses_source_features"] is True
    assert result.metadata["source_ecdf_uses_test_features_for_fitting"] is False
    assert result.metadata["source_ecdf_uses_test_labels"] is False
    assert result.metadata["source_ecdf_valid_for_strict_source_only"] is True


def test_source_ecdf_accepts_one_pass_feature_iterables() -> None:
    source_rows = ([float(row), float(row + 10)] for row in range(4))
    test_rows = ([1.5, 11.5], [4.0, 9.0])

    result = fit_source_ecdf_transform(source_features=source_rows, test_features=(row for row in test_rows), config={"output": "rank"})

    assert result.train_features.shape == (4, 2)
    assert result.test_features.shape == (2, 2)
    assert result.test_features.tolist() == [[2.0, 2.0], [4.0, 0.0]]


def test_source_ecdf_reference_can_be_reused() -> None:
    source = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    test = np.asarray([[1.5], [3.0]], dtype=float)
    reference = fit_source_ecdf_reference(source)

    direct = apply_source_ecdf_transform(test, reference)
    via_fit = fit_source_ecdf_transform(source_features=source, test_features=test)

    assert np.allclose(direct, via_fit.test_features)


def test_source_ecdf_apply_accepts_one_pass_iterables() -> None:
    reference = fit_source_ecdf_reference(([float(value)] for value in range(3)))

    transformed = apply_source_ecdf_transform(([1.5], [3.0]), reference)

    assert transformed.shape == (2, 1)
    assert np.allclose(transformed.ravel(), [0.625, 0.875])


def test_source_ecdf_rank_output_returns_counts() -> None:
    source = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    test = np.asarray([[-1.0], [0.0], [1.5], [3.0]], dtype=float)

    result = fit_source_ecdf_transform(source_features=source, test_features=test, config={"output": "rank"})

    assert result.test_features.ravel().tolist() == [0.0, 1.0, 2.0, 3.0]


def test_source_ecdf_normal_output_is_finite() -> None:
    source = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    test = np.asarray([[-100.0], [1.5], [100.0]], dtype=float)

    result = fit_source_ecdf_transform(source_features=source, test_features=test, config={"output": "normal", "epsilon": 1e-4})

    assert result.test_features.shape == (3, 1)
    assert np.all(np.isfinite(result.test_features))
    assert result.test_features[0, 0] < result.test_features[1, 0] < result.test_features[2, 0]


def test_source_ecdf_aliases_and_validation() -> None:
    assert normalize_ecdf_output("cdf") == "uniform"
    assert normalize_ecdf_output("normal-score") == "normal"
    assert source_ecdf_config(epsilon="1e-4").epsilon == 1e-4

    with pytest.raises(ValueError, match="output mode"):
        normalize_ecdf_output("bad")

    with pytest.raises(ValueError, match="epsilon"):
        source_ecdf_config(epsilon=0.75)


def test_source_ecdf_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_ecdf_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_ecdf_transform(source_features=[[0.0], [1.0]], test_features=[[0.5]], heldout_labels=[0])  # type: ignore[call-arg]

from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_random_projection import (
    SOURCE_RANDOM_PROJECTION_CATEGORY,
    apply_source_random_projection,
    fit_source_random_projection_reference,
    fit_source_random_projection_transform,
    normalize_projection_distribution,
    source_random_projection_config,
)


def test_gaussian_random_projection_shapes_and_metadata() -> None:
    source = np.arange(20, dtype=float).reshape(5, 4)
    test = np.arange(8, dtype=float).reshape(2, 4)

    result = fit_source_random_projection_transform(
        source_features=source,
        test_features=test,
        config={"n_components": 3, "random_state": 7},
    )

    assert result.train_features.shape == (5, 3)
    assert result.test_features.shape == (2, 3)
    assert result.reference.projection.shape == (4, 3)
    assert result.metadata["source_random_projection_protocol_category"] == SOURCE_RANDOM_PROJECTION_CATEGORY
    assert result.metadata["source_random_projection_uses_source_values"] is False
    assert result.metadata["source_random_projection_uses_test_features_for_fitting"] is False
    assert result.metadata["source_random_projection_uses_test_labels"] is False
    assert result.metadata["source_random_projection_valid_for_strict_source_only"] is True


def test_random_projection_is_reproducible_with_fixed_seed() -> None:
    first = fit_source_random_projection_reference(5, config={"n_components": 2, "random_state": 42})
    second = fit_source_random_projection_reference(5, config={"n_components": 2, "random_state": 42})

    assert np.allclose(first.projection, second.projection)


def test_sparse_projection_contains_zeros() -> None:
    reference = fit_source_random_projection_reference(16, config={"n_components": 8, "distribution": "sparse", "density": 0.25, "random_state": 1})

    assert reference.projection.shape == (16, 8)
    assert np.count_nonzero(reference.projection == 0.0) > 0


def test_projection_reference_can_be_reused() -> None:
    source = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    reference = fit_source_random_projection_reference(2, config={"n_components": 1, "random_state": 3})

    direct = apply_source_random_projection(source, reference)
    expected = source @ reference.projection

    assert np.allclose(direct, expected)


def test_aliases_and_validation() -> None:
    assert normalize_projection_distribution("normal") == "gaussian"
    assert normalize_projection_distribution("achlioptas") == "sparse"
    cfg = source_random_projection_config(n_components="all", distribution="dense", density="auto")
    assert cfg.n_components == "all"
    assert cfg.distribution == "gaussian"

    with pytest.raises(ValueError, match="distribution"):
        normalize_projection_distribution("bad")

    with pytest.raises(ValueError, match="n_components"):
        fit_source_random_projection_reference(4, config={"n_components": 0})

    with pytest.raises(ValueError, match="density"):
        source_random_projection_config(distribution="sparse", density=0.0)


def test_random_projection_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_random_projection_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_random_projection_transform(
            source_features=[[0.0], [1.0]],
            test_features=[[0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )

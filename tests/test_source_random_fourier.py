from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_random_fourier import (
    SOURCE_RANDOM_FOURIER_CATEGORY,
    apply_source_random_fourier,
    fit_source_random_fourier_reference,
    fit_source_random_fourier_transform,
    source_auto_rbf_gamma,
    source_random_fourier_config,
)


def test_random_fourier_shapes_and_metadata() -> None:
    source = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    test = np.asarray([[0.5, 0.5], [2.0, 0.0]], dtype=float)

    result = fit_source_random_fourier_transform(
        source_features=source,
        test_features=test,
        config={"n_components": 6, "gamma": 0.5, "random_state": 7},
    )

    assert result.train_features.shape == (4, 6)
    assert result.test_features.shape == (2, 6)
    assert result.reference.weights.shape == (2, 6)
    assert result.metadata["source_random_fourier_protocol_category"] == SOURCE_RANDOM_FOURIER_CATEGORY
    assert result.metadata["source_random_fourier_uses_source_features"] is True
    assert result.metadata["source_random_fourier_uses_test_features_for_fitting"] is False
    assert result.metadata["source_random_fourier_uses_test_labels"] is False
    assert result.metadata["source_random_fourier_valid_for_strict_source_only"] is True


def test_random_fourier_reference_is_reproducible() -> None:
    source = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    first = fit_source_random_fourier_reference(source, config={"n_components": 4, "gamma": 1.0, "random_state": 11})
    second = fit_source_random_fourier_reference(source, config={"n_components": 4, "gamma": 1.0, "random_state": 11})

    assert np.allclose(first.weights, second.weights)
    assert np.allclose(first.offsets, second.offsets)


def test_random_fourier_include_original_appends_input_features() -> None:
    source = np.asarray([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    test = np.asarray([[1.0, 2.0]], dtype=float)

    result = fit_source_random_fourier_transform(
        source_features=source,
        test_features=test,
        config={"n_components": 3, "include_original": True, "random_state": 5},
    )

    assert result.train_features.shape == (2, 5)
    assert result.test_features.shape == (1, 5)
    assert np.allclose(result.train_features[:, :2], source)
    assert result.metadata["source_random_fourier_include_original"] is True


def test_auto_gamma_uses_source_rows_only() -> None:
    source = np.asarray([[0.0], [2.0], [4.0]], dtype=float)
    gamma = source_auto_rbf_gamma(source)

    assert gamma > 0.0
    reference = fit_source_random_fourier_reference(source, config={"n_components": 2, "gamma": "auto"})
    assert np.isclose(reference.gamma, gamma)


def test_random_fourier_reference_can_be_reused() -> None:
    source = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=float)
    reference = fit_source_random_fourier_reference(source, config={"n_components": 3, "gamma": 0.5})

    direct = apply_source_random_fourier(source, reference)
    via_fit = fit_source_random_fourier_transform(source_features=source, test_features=source, config={"n_components": 3, "gamma": 0.5})

    assert direct.shape == via_fit.test_features.shape


def test_random_fourier_config_validation() -> None:
    cfg = source_random_fourier_config(n_components="5", gamma="0.25", include_original="true")
    assert cfg.n_components == "5"
    assert cfg.gamma == 0.25
    assert cfg.include_original is True

    with pytest.raises(ValueError, match="n_components"):
        fit_source_random_fourier_reference([[0.0], [1.0]], config={"n_components": 0})

    with pytest.raises(ValueError, match="gamma"):
        source_random_fourier_config(gamma=0.0)


def test_random_fourier_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_random_fourier_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_random_fourier_transform(
            source_features=[[0.0], [1.0]],
            test_features=[[0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )

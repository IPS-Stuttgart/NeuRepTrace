from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_polynomial import (
    SOURCE_POLYNOMIAL_CATEGORY,
    SourcePolynomialConfig,
    apply_source_polynomial_transform,
    fit_source_polynomial_reference,
    fit_source_polynomial_transform,
    source_polynomial_config,
)


def test_polynomial_transform_shapes_and_metadata() -> None:
    source = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    test = np.asarray([[5.0, 6.0]], dtype=float)

    result = fit_source_polynomial_transform(source_features=source, test_features=test)

    assert result.train_features.shape == (2, 5)
    assert result.test_features.shape == (1, 5)
    assert result.reference.output_names == ("x0", "x1", "x0^2", "x1^2", "x0*x1")
    assert np.allclose(result.test_features[0], np.asarray([5.0, 6.0, 25.0, 36.0, 30.0]))
    assert result.metadata["source_polynomial_protocol_category"] == SOURCE_POLYNOMIAL_CATEGORY
    assert result.metadata["source_polynomial_uses_source_values"] is False
    assert result.metadata["source_polynomial_uses_test_features_for_fitting"] is False
    assert result.metadata["source_polynomial_uses_test_labels"] is False
    assert result.metadata["source_polynomial_valid_for_strict_source_only"] is True


def test_polynomial_accepts_one_pass_iterable_feature_rows() -> None:
    source_rows = (row for row in [[1.0, 2.0], [3.0, 4.0]])
    test_rows = (row for row in [[5.0, 6.0]])

    result = fit_source_polynomial_transform(source_features=source_rows, test_features=test_rows)
    reused = apply_source_polynomial_transform((row for row in [[7.0, 8.0]]), result.reference)

    assert result.train_features.shape == (2, 5)
    assert result.test_features.shape == (1, 5)
    assert np.allclose(result.test_features[0], np.asarray([5.0, 6.0, 25.0, 36.0, 30.0]))
    assert reused.shape == (1, 5)
    assert np.allclose(reused[0], np.asarray([7.0, 8.0, 49.0, 64.0, 56.0]))


def test_polynomial_reference_can_be_reused() -> None:
    reference = fit_source_polynomial_reference(3, config={"include_bias": True, "max_interactions": 1})
    rows = np.asarray([[2.0, 3.0, 4.0]], dtype=float)

    transformed = apply_source_polynomial_transform(rows, reference)

    assert transformed.shape == (1, 8)
    assert reference.output_names == ("bias", "x0", "x1", "x2", "x0^2", "x1^2", "x2^2", "x0*x1")
    assert np.allclose(transformed[0], np.asarray([1.0, 2.0, 3.0, 4.0, 4.0, 9.0, 16.0, 6.0]))


def test_polynomial_can_disable_blocks() -> None:
    source = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    test = np.asarray([[4.0, 5.0, 6.0]], dtype=float)

    result = fit_source_polynomial_transform(
        source_features=source,
        test_features=test,
        config={"include_original": False, "include_squares": False, "include_interactions": True, "max_interactions": 2},
    )

    assert result.reference.output_names == ("x0*x1", "x0*x2")
    assert np.allclose(result.test_features[0], np.asarray([20.0, 24.0]))


def test_polynomial_requires_at_least_one_output() -> None:
    with pytest.raises(ValueError, match="At least one"):
        fit_source_polynomial_reference(2, config={"include_original": False, "include_squares": False, "include_interactions": False, "include_bias": False})


def test_polynomial_config_validation() -> None:
    cfg = source_polynomial_config(include_bias="true", include_interactions="false")
    assert cfg.include_bias is True
    assert cfg.include_interactions is False

    with pytest.raises(ValueError, match="max_interactions"):
        fit_source_polynomial_reference(3, config={"max_interactions": -1})


def test_direct_polynomial_config_normalizes_boolean_strings() -> None:
    cfg = SourcePolynomialConfig(include_bias="true", include_original="false", include_squares="false", include_interactions="false")

    assert cfg.include_bias is True
    assert cfg.include_original is False
    assert cfg.include_squares is False
    assert cfg.include_interactions is False
    assert fit_source_polynomial_reference(2, config=cfg).output_names == ("bias",)


def test_direct_polynomial_config_normalizes_numpy_scalars() -> None:
    cfg = SourcePolynomialConfig(
        include_bias=np.asarray(True),
        include_original=np.bool_(False),
        include_squares=np.asarray(False),
        include_interactions=np.asarray(False),
        max_interactions=np.asarray(0),
    )

    assert cfg.include_bias is True
    assert cfg.include_original is False
    assert cfg.include_squares is False
    assert cfg.include_interactions is False
    assert cfg.max_interactions == 0
    assert fit_source_polynomial_reference(2, config=cfg).output_names == ("bias",)


def test_direct_polynomial_config_normalizes_max_interaction_alias() -> None:
    cfg = SourcePolynomialConfig(max_interactions="full")

    assert cfg.max_interactions == "all"
    assert len(fit_source_polynomial_reference(3, config=cfg).interaction_pairs) == 3


@pytest.mark.parametrize(
    ("field", "kwargs"),
    [
        ("include_bias", {"include_bias": np.asarray([True])}),
        ("include_original", {"include_original": np.asarray([False])}),
        ("include_squares", {"include_squares": np.asarray([False])}),
        ("include_interactions", {"include_interactions": np.asarray([False])}),
    ],
)
def test_direct_polynomial_config_rejects_vector_boolean_controls(field: str, kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match=field):
        SourcePolynomialConfig(**kwargs)


@pytest.mark.parametrize("n_features", [True, np.bool_(True)])
def test_polynomial_rejects_boolean_feature_width(n_features) -> None:
    with pytest.raises(ValueError, match="n_features must be a positive integer"):
        fit_source_polynomial_reference(n_features)


@pytest.mark.parametrize("max_interactions", [True, False, np.bool_(True), np.bool_(False)])
def test_polynomial_rejects_boolean_max_interactions(max_interactions) -> None:
    with pytest.raises(ValueError, match="max_interactions must be a non-negative integer"):
        fit_source_polynomial_reference(3, config={"max_interactions": max_interactions})


def test_polynomial_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_polynomial_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_polynomial_transform(
            source_features=[[0.0], [1.0]],
            test_features=[[0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )

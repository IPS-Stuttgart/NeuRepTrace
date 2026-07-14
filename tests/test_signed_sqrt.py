from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.signed_sqrt import (
    SIGNED_SQRT_CATEGORY,
    SignedSqrtConfig,
    signed_sqrt_config,
    signed_sqrt_transform,
    transform_train_test_signed_sqrt,
)


def test_signed_sqrt_transform_preserves_sign_and_compresses_magnitude() -> None:
    transformed = signed_sqrt_transform([[-4.0, -1.0, 0.0, 1.0, 9.0]])

    assert np.allclose(transformed, np.asarray([[-2.0, -1.0, 0.0, 1.0, 3.0]]))


def test_signed_sqrt_transform_accepts_one_pass_feature_iterables() -> None:
    values = (row for row in [[-4.0, -1.0, 0.0, 1.0, 9.0]])

    transformed = signed_sqrt_transform(values)

    assert np.allclose(transformed, np.asarray([[-2.0, -1.0, 0.0, 1.0, 3.0]]))


def test_signed_sqrt_transform_accepts_nested_one_pass_feature_iterables() -> None:
    values = ((value for value in row) for row in ([-4.0, 9.0], [16.0, -25.0]))

    transformed = signed_sqrt_transform(values)

    np.testing.assert_allclose(transformed, np.asarray([[-2.0, 3.0], [4.0, -5.0]]))


@pytest.mark.parametrize(
    "features",
    [
        [[True, False]],
        [[1.0, np.bool_(True)]],
        np.asarray([[True, False]], dtype=bool),
        np.asarray([[1.0, True]], dtype=object),
    ],
)
def test_signed_sqrt_rejects_boolean_feature_values(features: object) -> None:
    with pytest.raises(ValueError, match="boolean flags"):
        signed_sqrt_transform(features)  # type: ignore[arg-type]


def test_signed_sqrt_rejects_boolean_values_in_nested_one_pass_iterables() -> None:
    values = ((value for value in row) for row in ([1.0, True],))

    with pytest.raises(ValueError, match="boolean flags"):
        signed_sqrt_transform(values)


def test_signed_sqrt_scale_changes_output() -> None:
    transformed = signed_sqrt_transform([[4.0, -4.0]], scale=4.0)

    assert np.allclose(transformed, np.asarray([[1.0, -1.0]]))


def test_transform_train_test_signed_sqrt_metadata() -> None:
    result = transform_train_test_signed_sqrt(
        train_features=[[4.0, 9.0]],
        test_features=[[-1.0, 0.0]],
    )

    assert result.train_features.shape == (1, 2)
    assert result.test_features.shape == (1, 2)
    assert result.metadata["signed_sqrt_protocol_category"] == SIGNED_SQRT_CATEGORY
    assert result.metadata["signed_sqrt_has_fitted_parameters"] is False
    assert result.metadata["signed_sqrt_uses_labels"] is False
    assert result.metadata["signed_sqrt_valid_for_strict_source_only"] is True


def test_transform_train_test_signed_sqrt_accepts_one_pass_feature_iterables() -> None:
    result = transform_train_test_signed_sqrt(
        train_features=(row for row in [[4.0, 9.0], [16.0, 25.0]]),
        test_features=(row for row in [[-1.0, 0.0]]),
    )

    assert result.train_features.shape == (2, 2)
    assert result.test_features.shape == (1, 2)
    assert result.metadata["signed_sqrt_n_train_rows"] == 2
    assert result.metadata["signed_sqrt_n_test_rows"] == 1


def test_signed_sqrt_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        transform_train_test_signed_sqrt(train_features=[[1.0, 2.0]], test_features=[[1.0]])


def test_signed_sqrt_config_validation() -> None:
    assert signed_sqrt_config(scale="2.5").scale == 2.5
    with pytest.raises(ValueError, match="scale"):
        signed_sqrt_config(scale=0.0)


@pytest.mark.parametrize("scale", [True, np.asarray(2.0), np.asarray([2.0])])
def test_signed_sqrt_config_rejects_boolean_and_array_scales(scale: object) -> None:
    with pytest.raises(ValueError, match="scale"):
        signed_sqrt_config(scale=scale)  # type: ignore[arg-type]


@pytest.mark.parametrize("scale", [True, np.asarray(2.0), np.asarray([2.0])])
def test_signed_sqrt_revalidates_config_objects(scale: object) -> None:
    with pytest.raises(ValueError, match="scale"):
        transform_train_test_signed_sqrt(train_features=[[1.0]], test_features=[[1.0]], config=SignedSqrtConfig(scale=scale))  # type: ignore[arg-type]

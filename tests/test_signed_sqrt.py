from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.signed_sqrt import (
    SIGNED_SQRT_CATEGORY,
    signed_sqrt_config,
    signed_sqrt_transform,
    transform_train_test_signed_sqrt,
)


def test_signed_sqrt_transform_preserves_sign_and_compresses_magnitude() -> None:
    transformed = signed_sqrt_transform([[-4.0, -1.0, 0.0, 1.0, 9.0]])

    assert np.allclose(transformed, np.asarray([[-2.0, -1.0, 0.0, 1.0, 3.0]]))


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


def test_signed_sqrt_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        transform_train_test_signed_sqrt(train_features=[[1.0, 2.0]], test_features=[[1.0]])


def test_signed_sqrt_config_validation() -> None:
    assert signed_sqrt_config(scale="2.5").scale == 2.5
    with pytest.raises(ValueError, match="scale"):
        signed_sqrt_config(scale=0.0)

from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.signed_log import (
    SIGNED_LOG_CATEGORY,
    SignedLogConfig,
    signed_log_config,
    transform_signed_log,
    transform_train_test_signed_log,
)


def test_signed_log_transform_matches_formula() -> None:
    values = np.asarray([[-3.0, 0.0, 3.0]], dtype=float)

    transformed = transform_signed_log(values, scale=3.0)

    assert np.allclose(transformed, np.asarray([[-np.log1p(1.0), 0.0, np.log1p(1.0)]]))


def test_signed_log_transform_handles_extreme_finite_scale_ratios() -> None:
    maximum = np.finfo(np.float64).max
    minimum = np.nextafter(0.0, 1.0)
    values = np.asarray([[maximum, -maximum, 1.0, -1.0, minimum, -minimum, 0.0]])

    with np.errstate(over="raise", under="raise", divide="raise", invalid="raise"):
        transformed = transform_signed_log(values, scale=minimum)

    expected_maximum = np.log(maximum) - np.log(minimum)
    assert np.all(np.isfinite(transformed))
    assert transformed[0, 0] == pytest.approx(expected_maximum)
    assert transformed[0, 1] == pytest.approx(-expected_maximum)
    assert transformed[0, 2] == pytest.approx(-transformed[0, 3])
    assert transformed[0, 4] == pytest.approx(np.log1p(1.0))
    assert transformed[0, 5] == pytest.approx(-np.log1p(1.0))
    assert transformed[0, 6] == 0.0


def test_signed_log_accepts_one_pass_feature_iterables() -> None:
    values = (row for row in [[-3.0, 0.0, 3.0]])

    transformed = transform_signed_log(values, scale=3.0)

    assert np.allclose(transformed, np.asarray([[-np.log1p(1.0), 0.0, np.log1p(1.0)]]))


def test_train_test_signed_log_metadata_and_iterables() -> None:
    result = transform_train_test_signed_log(
        train_features=(row for row in [[-1.0, 0.0], [1.0, 2.0]]),
        test_features=(row for row in [[3.0, -3.0]]),
        config={"scale": 2.0},
    )

    assert result.train_features.shape == (2, 2)
    assert result.test_features.shape == (1, 2)
    assert result.metadata["signed_log_protocol_category"] == SIGNED_LOG_CATEGORY
    assert result.metadata["signed_log_has_fitted_parameters"] is False
    assert result.metadata["signed_log_uses_labels"] is False
    assert result.metadata["signed_log_valid_for_strict_source_only"] is True
    assert result.metadata["signed_log_n_train_rows"] == 2
    assert result.metadata["signed_log_n_test_rows"] == 1
    assert result.metadata["signed_log_scale"] == 2.0


def test_train_test_signed_log_revalidates_dataclass_config() -> None:
    with pytest.raises(ValueError, match="scale"):
        transform_train_test_signed_log(
            train_features=[[1.0]],
            test_features=[[2.0]],
            config=SignedLogConfig(scale=True),  # type: ignore[arg-type]
        )


def test_train_test_signed_log_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        transform_train_test_signed_log(train_features=[[1.0, 2.0]], test_features=[[1.0]])


def test_signed_log_config_validation() -> None:
    assert signed_log_config(scale="2.5").scale == 2.5

    with pytest.raises(ValueError, match="scale"):
        signed_log_config(scale=0.0)


@pytest.mark.parametrize("scale", [True, np.asarray(2.0), np.asarray([2.0])])
def test_signed_log_config_rejects_boolean_and_array_scales(scale: object) -> None:
    with pytest.raises(ValueError, match="scale"):
        signed_log_config(scale=scale)  # type: ignore[arg-type]

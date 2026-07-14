from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding import mekt


def _toy_covariances() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.stack(
        [
            np.diag([1.0, 2.0]),
            np.diag([1.1, 2.1]),
            np.diag([2.0, 1.0]),
            np.diag([2.1, 1.1]),
        ],
        axis=0,
    )
    target = np.stack([np.diag([1.2, 1.9]), np.diag([1.9, 1.2])], axis=0)
    labels = np.asarray([0, 0, 1, 1])
    return source, labels, target


@pytest.mark.parametrize(
    ("validator_name", "value", "parameter_name"),
    [
        ("_positive_int", np.asarray([1]), "n_iterations"),
        ("_positive_float", np.asarray([1.0]), "rho"),
        ("_nonnegative_float", np.asarray([0.0]), "alpha"),
    ],
)
def test_mekt_scalar_validators_reject_one_element_vectors(validator_name: str, value: np.ndarray, parameter_name: str) -> None:
    validator = getattr(mekt, validator_name)

    with pytest.raises(ValueError, match=parameter_name):
        validator(value, name=parameter_name)


@pytest.mark.parametrize("value", [np.inf, object()])
def test_mekt_positive_integer_conversion_failures_use_value_error(value: object) -> None:
    with pytest.raises(ValueError, match="n_iterations"):
        mekt._positive_int(value, name="n_iterations")


def test_mekt_scalar_validators_accept_zero_dimensional_arrays() -> None:
    assert mekt._positive_int(np.asarray(2), name="n_iterations") == 2
    assert mekt._positive_float(np.asarray(0.5), name="rho") == pytest.approx(0.5)
    assert mekt._nonnegative_float(np.asarray(0.0), name="alpha") == pytest.approx(0.0)


@pytest.mark.parametrize("value", [np.asarray([1]), np.inf, object()])
def test_mekt_transfer_rejects_invalid_iteration_controls_with_stable_error(value: object) -> None:
    source, labels, target = _toy_covariances()

    with pytest.raises(ValueError, match="n_iterations must be a positive integer"):
        mekt.mekt_transfer_features(source, labels, target, n_iterations=value)

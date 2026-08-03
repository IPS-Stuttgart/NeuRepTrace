from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.signed_power import signed_power_transform


def test_signed_power_accepts_generator_backed_rows() -> None:
    rows = (
        (value for value in row)
        for row in [
            [-4.0, -1.0, 0.0, 1.0, 9.0],
            [16.0, 25.0, 36.0, 49.0, 64.0],
        ]
    )

    transformed = signed_power_transform(rows, power=0.5)

    assert np.allclose(
        transformed,
        np.asarray(
            [
                [-2.0, -1.0, 0.0, 1.0, 3.0],
                [4.0, 5.0, 6.0, 7.0, 8.0],
            ]
        ),
    )


def test_signed_power_rejects_boolean_feature_values() -> None:
    with pytest.raises(ValueError, match="boolean"):
        signed_power_transform([[True, 1.0]])


def test_signed_power_rejects_boolean_values_in_generator_rows() -> None:
    rows = ((value for value in row) for row in [[True, 1.0]])

    with pytest.raises(ValueError, match="boolean"):
        signed_power_transform(rows)


def test_signed_power_rejects_complex_feature_values_clearly() -> None:
    with pytest.raises(ValueError, match="complex"):
        signed_power_transform([[1.0 + 2.0j]])

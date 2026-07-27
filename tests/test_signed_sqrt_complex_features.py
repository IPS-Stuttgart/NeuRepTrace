from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from neureptrace.decoding.signed_sqrt import signed_sqrt_transform, transform_train_test_signed_sqrt


@pytest.mark.parametrize(
    "feature_factory",
    [
        lambda: np.asarray([[1.0 + 2.0j, 4.0]], dtype=np.complex128),
        lambda: np.asarray([[1.0 + 2.0j, 4.0]], dtype=object),
        lambda: (iter(row) for row in ([1.0 + 2.0j, 4.0],)),
    ],
    ids=["native-array", "object-array", "nested-one-pass"],
)
def test_signed_sqrt_rejects_complex_feature_values(feature_factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="features.*real-valued.*complex"):
        signed_sqrt_transform(feature_factory())  # type: ignore[arg-type]


def test_train_test_signed_sqrt_rejects_complex_held_out_features() -> None:
    complex_test = np.asarray([[1.0 + 2.0j, 4.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="test_features.*real-valued.*complex"):
        transform_train_test_signed_sqrt(
            train_features=[[1.0, 4.0]],
            test_features=complex_test,
        )

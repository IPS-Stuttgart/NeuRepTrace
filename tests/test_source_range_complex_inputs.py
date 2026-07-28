from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_range import (
    apply_source_range_clip,
    source_feature_range,
    source_range_clip,
)


@pytest.mark.parametrize(
    "source_features",
    [
        [[1.0 + 2.0j, 0.0], [2.0, 1.0]],
        np.asarray([[1.0 + 2.0j, 0.0], [2.0, 1.0]], dtype=np.complex128),
        np.asarray([[1.0 + 2.0j, 0.0], [2.0, 1.0]], dtype=object),
        (iter(row) for row in ([1.0 + 2.0j, 0.0], [2.0, 1.0])),
    ],
)
def test_source_feature_range_rejects_complex_source_values(source_features) -> None:
    with pytest.raises(ValueError, match="source_features.*complex values"):
        source_feature_range(source_features)


def test_source_range_clip_rejects_complex_test_values() -> None:
    test_features = np.asarray([[1.0 + 2.0j, 0.0]], dtype=np.complex64)

    with pytest.raises(ValueError, match="test_features.*complex values"):
        source_range_clip(
            source_features=[[0.0, 1.0], [1.0, 2.0]],
            test_features=test_features,
        )


@pytest.mark.parametrize(
    ("bound_name", "lower", "upper"),
    [
        ("lower", np.asarray([0.0 + 1.0j, 0.0]), [1.0, 2.0]),
        ("upper", [0.0, 0.0], (value for value in [1.0, 2.0 + 1.0j])),
    ],
)
def test_apply_source_range_clip_rejects_complex_bounds(
    bound_name: str,
    lower,
    upper,
) -> None:
    with pytest.raises(ValueError, match=rf"{bound_name}.*complex values"):
        apply_source_range_clip([[0.5, 1.5]], lower=lower, upper=upper)

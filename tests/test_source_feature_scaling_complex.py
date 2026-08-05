from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_scaling import augment_source_with_feature_scaling


@pytest.mark.parametrize(
    "config",
    [
        None,
        {"synthetic_per_class": 1, "random_state": 7},
    ],
)
def test_source_feature_scaling_rejects_complex_numpy_features(config: object) -> None:
    features = np.asarray([[1.0 + 2.0j, 3.0], [4.0, 5.0 + 6.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="source_features must contain real-valued feature values"):
        augment_source_with_feature_scaling(features, [0, 1], config=config)  # type: ignore[arg-type]


def test_source_feature_scaling_rejects_complex_object_features() -> None:
    features = np.asarray([[1.0, np.complex128(2.0 + 3.0j)], [4.0, 5.0]], dtype=object)

    with pytest.raises(ValueError, match="source_features must contain real-valued feature values"):
        augment_source_with_feature_scaling(features, [0, 1])

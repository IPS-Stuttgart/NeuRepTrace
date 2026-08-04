from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_jitter import augment_source_with_feature_jitter


@pytest.mark.parametrize(
    "features",
    [
        np.asarray(
            [[1.0 + 2.0j, 2.0], [3.0, 4.0]],
            dtype=np.complex128,
        ),
        np.asarray(
            [[1.0 + 2.0j, 2.0], [3.0, 4.0]],
            dtype=object,
        ),
        [[np.complex64(1.0 + 2.0j), 2.0], [3.0, 4.0]],
    ],
)
def test_source_feature_jitter_rejects_complex_features(features: object) -> None:
    with pytest.raises(
        ValueError,
        match="source_features must contain only real values",
    ):
        augment_source_with_feature_jitter(features, [0, 1])


def test_source_feature_jitter_still_accepts_real_features() -> None:
    result = augment_source_with_feature_jitter(
        [[1.0, 2.0], [3.0, 4.0]],
        [0, 1],
    )

    assert result.features.shape == (2, 2)

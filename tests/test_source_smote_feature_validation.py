from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_smote import augment_source_with_smote


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
def test_source_smote_rejects_complex_source_features(features: object) -> None:
    with pytest.raises(
        ValueError,
        match="source_features must contain real-valued features, not complex values",
    ):
        augment_source_with_smote(features, [0, 1])


@pytest.mark.parametrize(
    "features",
    [
        np.asarray([[True, 2.0], [3.0, 4.0]], dtype=object),
        [[1.0, np.bool_(False)], [3.0, 4.0]],
    ],
)
def test_source_smote_rejects_boolean_source_features(features: object) -> None:
    with pytest.raises(
        ValueError,
        match="source_features must contain real numeric values, not booleans",
    ):
        augment_source_with_smote(features, [0, 1])


def test_source_smote_keeps_real_generator_features() -> None:
    features = ((value for value in row) for row in ([1.0, 2.0], [3.0, 4.0]))

    result = augment_source_with_smote(features, [0, 1])

    np.testing.assert_allclose(result.features, [[1.0, 2.0], [3.0, 4.0]])

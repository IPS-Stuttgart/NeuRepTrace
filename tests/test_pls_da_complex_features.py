from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding import PLSDiscriminantTransformer


def _real_features() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.1, 0.1],
            [0.1, 1.1],
        ]
    )


def test_pls_da_fit_rejects_complex_features() -> None:
    features = _real_features().astype(np.complex128)
    features[0, 0] += 2.0j

    with pytest.raises(ValueError, match="features.*real-valued.*complex"):
        PLSDiscriminantTransformer(n_components=1).fit(features, [0, 1, 0, 1])


def test_pls_da_transform_rejects_complex_features() -> None:
    transformer = PLSDiscriminantTransformer(n_components=1).fit(
        _real_features(),
        [0, 1, 0, 1],
    )
    features = _real_features().astype(object)
    features[0, 0] = 1.0 + 2.0j

    with pytest.raises(ValueError, match="features.*real-valued.*complex"):
        transformer.transform(features)

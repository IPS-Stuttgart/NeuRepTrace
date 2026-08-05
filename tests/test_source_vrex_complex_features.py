from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401 - importing the package installs runtime patches.
from neureptrace.decoding.source_vrex import TorchVRExClassifier


@pytest.mark.parametrize(
    "source_features",
    [
        np.asarray(
            [
                [1.0 + 0.0j, 2.0],
                [2.0, 3.0],
                [3.0, 4.0 + 1.0j],
                [4.0, 5.0],
            ],
            dtype=np.complex128,
        ),
        np.asarray(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0 + 1.0j],
                [4.0, 5.0],
            ],
            dtype=object,
        ),
    ],
    ids=["complex-array", "object-array"],
)
def test_torch_vrex_rejects_complex_source_features(source_features: np.ndarray) -> None:
    model = TorchVRExClassifier(max_epochs=1)
    labels = np.asarray([0, 1, 0, 1])
    domains = np.asarray(["s1", "s1", "s2", "s2"], dtype=object)

    with pytest.raises(ValueError, match="vrex source_features.*complex"):
        model.fit(source_features, labels, source_domains=domains)

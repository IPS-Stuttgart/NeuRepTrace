from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_knn import SourceKNNConfig


def test_source_knn_config_direct_construction_normalizes_values() -> None:
    cfg = SourceKNNConfig(
        k=np.asarray("2"),
        weights="inverse-distance",
        standardize=np.asarray(False),
        epsilon=np.asarray("1e-4"),
    )

    assert cfg.k == 2
    assert cfg.weights == "distance"
    assert cfg.standardize is False
    assert cfg.epsilon == pytest.approx(1e-4)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"k": np.asarray([1])}, "k"),
        ({"k": True}, "k"),
        ({"weights": "bad"}, "weight mode"),
        ({"standardize": np.asarray([False])}, "standardize"),
        ({"standardize": "sometimes"}, "standardize"),
        ({"epsilon": np.asarray([1e-4])}, "epsilon"),
        ({"epsilon": False}, "epsilon"),
    ],
)
def test_source_knn_config_direct_construction_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SourceKNNConfig(**kwargs)

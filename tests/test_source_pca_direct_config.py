from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_pca import SourcePCAConfig


def test_source_pca_config_direct_construction_normalizes_values() -> None:
    cfg = SourcePCAConfig(
        n_components=np.asarray("2"),
        center="false",
        scale="yes",
        whiten=np.asarray(1),
        epsilon="1e-9",
    )

    assert cfg.n_components == 2
    assert cfg.center is False
    assert cfg.scale is True
    assert cfg.whiten is True
    assert cfg.epsilon == pytest.approx(1e-9)

    full_cfg = SourcePCAConfig(n_components="FULL", center=np.asarray(False), scale=0.0, whiten="off")

    assert full_cfg.n_components == "full"
    assert full_cfg.center is False
    assert full_cfg.scale is False
    assert full_cfg.whiten is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_components": np.asarray([1])}, "n_components"),
        ({"n_components": True}, "n_components"),
        ({"center": np.asarray([False])}, "center"),
        ({"scale": "sometimes"}, "scale"),
        ({"whiten": 0.5}, "whiten"),
        ({"epsilon": np.asarray([1e-9])}, "epsilon"),
        ({"epsilon": False}, "epsilon"),
    ],
)
def test_source_pca_config_direct_construction_rejects_invalid_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        SourcePCAConfig(**kwargs)

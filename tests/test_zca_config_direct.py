from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_zca import SourceZCAConfig


def test_zca_config_direct_construction_normalizes_values() -> None:
    cfg = SourceZCAConfig(regularization=np.asarray(1e-4), center="false", recolor=np.asarray(True))  # type: ignore[arg-type]

    assert np.isclose(cfg.regularization, 1e-4)
    assert cfg.center is False
    assert cfg.recolor is True


def test_zca_config_direct_construction_rejects_invalid_regularization() -> None:
    with pytest.raises(ValueError, match="regularization"):
        SourceZCAConfig(regularization=np.asarray([1e-4]))  # type: ignore[arg-type]


def test_zca_config_direct_construction_rejects_invalid_boolean() -> None:
    with pytest.raises(ValueError, match="center"):
        SourceZCAConfig(center="maybe")  # type: ignore[arg-type]

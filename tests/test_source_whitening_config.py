from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_whitening import (
    SourceWhiteningConfig,
    fit_source_whitening_transform,
)


def test_direct_source_whitening_config_normalizes_values() -> None:
    cfg = SourceWhiteningConfig(
        mode="pca-whitening",
        regularization=np.asarray("1e-5"),
        center="false",
        epsilon=np.asarray("1e-9"),
    )

    assert cfg.mode == "pca"
    assert cfg.regularization == pytest.approx(1e-5)
    assert cfg.center is False
    assert cfg.epsilon == pytest.approx(1e-9)

    transform = fit_source_whitening_transform(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        config=cfg,
    )

    assert transform.mode == "pca"
    assert transform.regularization == pytest.approx(1e-5)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"regularization": True}, "regularization"),
        ({"regularization": np.asarray(True)}, "regularization"),
        ({"regularization": np.asarray([0.1])}, "regularization"),
        ({"epsilon": False}, "epsilon"),
        ({"epsilon": np.asarray(False)}, "epsilon"),
        ({"epsilon": np.asarray([1e-6])}, "epsilon"),
        ({"center": np.asarray([True])}, "center"),
    ],
)
def test_direct_source_whitening_config_rejects_malformed_values(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SourceWhiteningConfig(**kwargs)

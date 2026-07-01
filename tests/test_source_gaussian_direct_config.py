from __future__ import annotations

import pytest

from neureptrace.decoding.source_gaussian import SourceGaussianConfig


def test_direct_source_gaussian_config_normalizes_aliases_and_numeric_controls() -> None:
    config = SourceGaussianConfig(
        covariance_type="diag",
        prior="flat",
        variance_floor="1e-5",  # type: ignore[arg-type]
        temperature="2.0",  # type: ignore[arg-type]
    )

    assert config.covariance_type == "diagonal"
    assert config.prior == "uniform"
    assert config.variance_floor == pytest.approx(1e-5)
    assert config.temperature == pytest.approx(2.0)

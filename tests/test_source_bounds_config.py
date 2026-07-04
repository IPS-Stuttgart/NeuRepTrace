from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_bounds import SourceBoundsConfig, source_bounds_config


@pytest.mark.parametrize("value", [True, np.bool_(False), np.asarray(True)])
def test_source_bounds_config_rejects_boolean_quantiles(value) -> None:
    with pytest.raises(ValueError, match="lower_quantile must be a numeric quantile"):
        source_bounds_config(lower_quantile=value, upper_quantile=0.9)


def test_source_bounds_config_normalizes_symmetric_bool_strings() -> None:
    assert source_bounds_config(symmetric="false").symmetric is False
    assert source_bounds_config(symmetric="off").symmetric is False
    assert source_bounds_config(symmetric="yes").symmetric is True
    assert source_bounds_config(symmetric=np.asarray(False)).symmetric is False

    with pytest.raises(ValueError, match="symmetric must be a boolean"):
        source_bounds_config(symmetric="maybe")


def test_source_bounds_dataclass_validates_and_normalizes_fields() -> None:
    cfg = SourceBoundsConfig(
        lower_quantile="0.2",
        upper_quantile=np.asarray(0.8),
        symmetric="on",
        center="zero-center",
    )

    assert cfg.lower_quantile == pytest.approx(0.2)
    assert cfg.upper_quantile == pytest.approx(0.8)
    assert cfg.symmetric is True
    assert cfg.center == "zero"

    with pytest.raises(ValueError, match="upper_quantile must be a numeric quantile"):
        SourceBoundsConfig(lower_quantile=0.1, upper_quantile=np.asarray(False))

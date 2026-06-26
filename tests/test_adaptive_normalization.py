from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.adaptive_normalization import adaptive_normalization_config, fit_adaptive_feature_normalization


def test_adaptive_normalization_config_parses_boolean_strings() -> None:
    cfg = adaptive_normalization_config(center="false", scale="off", robust="yes")

    assert cfg.center is False
    assert cfg.scale is False
    assert cfg.robust is True


def test_adaptive_normalization_config_rejects_ambiguous_boolean_strings() -> None:
    with pytest.raises(ValueError, match="center"):
        adaptive_normalization_config(center="sometimes")


def test_adaptive_normalization_string_false_disables_transform() -> None:
    source = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    target = np.asarray([[10.0, 20.0]], dtype=float)

    result = fit_adaptive_feature_normalization(
        source_features=source,
        target_features=target,
        config={"mode": "source_only", "center": "false", "scale": "off"},
    )

    np.testing.assert_allclose(result.train_features, source.astype(np.float32))
    np.testing.assert_allclose(result.test_features, target.astype(np.float32))
    assert result.metadata["adaptive_feature_normalization_center"] is False
    assert result.metadata["adaptive_feature_normalization_scale"] is False

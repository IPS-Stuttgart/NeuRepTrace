from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_quantile import SOURCE_QUANTILE_CATEGORY, source_feature_quantiles


def test_source_feature_quantiles_are_source_only() -> None:
    features = np.asarray([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [100.0, 20.0]], dtype=float)

    lower, upper = source_feature_quantiles(features, lower=0.25, upper=0.75)

    assert SOURCE_QUANTILE_CATEGORY == "1_strict_source_only"
    assert np.allclose(lower, np.quantile(features, 0.25, axis=0))
    assert np.allclose(upper, np.quantile(features, 0.75, axis=0))


def test_source_feature_quantiles_validate_bounds() -> None:
    with pytest.raises(ValueError, match="lower"):
        source_feature_quantiles([[0.0], [1.0]], lower=0.9, upper=0.1)


def test_source_feature_quantiles_reject_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        source_feature_quantiles([[0.0], [float("nan")]])


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        source_feature_quantiles([[0.0], [1.0]], heldout_features=[[0.5]])  # type: ignore[call-arg]

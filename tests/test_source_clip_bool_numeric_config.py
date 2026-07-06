from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_clip import fit_source_clip_then_standardize, source_clip_config


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lower_quantile": False},
        {"upper_quantile": True},
        {"lower_quantile": np.asarray(False)},
        {"upper_quantile": np.asarray(True)},
    ],
)
def test_source_clip_rejects_boolean_numeric_quantiles(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="quantile"):
        source_clip_config(**kwargs)


@pytest.mark.parametrize("epsilon", [True, np.bool_(True), np.asarray(True)])
def test_clip_then_standardize_rejects_boolean_epsilon(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        fit_source_clip_then_standardize(source_features=[[0.0], [1.0]], test_features=[[0.5]], epsilon=epsilon)

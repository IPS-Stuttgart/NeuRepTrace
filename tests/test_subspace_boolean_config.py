from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.subspace_adaptation import fit_subspace_adaptation, subspace_adaptation_config


def _domains() -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray([[-1.0, 0.0], [-0.8, 0.1], [0.8, -0.1], [1.0, 0.0]], dtype=float)
    target = source + np.asarray([2.0, 0.25])
    return source, target


def test_subspace_boolean_strings_are_parsed() -> None:
    config = subspace_adaptation_config(
        standardize="false",
        class_balance_source="false",
        normalize_latent="false",
    )

    assert config.standardize is False
    assert config.class_balance_source is False
    assert config.normalize_latent is False

    source, target = _domains()
    result = fit_subspace_adaptation(
        source,
        target,
        n_components=1,
        standardize="false",
        class_balance_source="false",
        normalize_latent="false",
    )

    assert result.metadata["subspace_adaptation_standardize"] is False
    assert result.metadata["subspace_adaptation_class_balance_source"] is False
    assert result.metadata["subspace_adaptation_normalize_latent"] is False
    np.testing.assert_array_equal(result.feature_mean, np.zeros(source.shape[1]))
    np.testing.assert_array_equal(result.feature_scale, np.ones(source.shape[1]))


@pytest.mark.parametrize("field", ["standardize", "class_balance_source", "normalize_latent"])
def test_subspace_config_rejects_invalid_boolean_strings(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        subspace_adaptation_config(**{field: "sometimes"})

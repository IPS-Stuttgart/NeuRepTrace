from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.subspace_adaptation import fit_subspace_adaptation, subspace_adaptation_config


def _shifted_domains():
    source = np.asarray(
        [
            [-1.0, -0.1, 0.0],
            [-0.8, 0.1, 0.2],
            [1.0, 0.1, 0.0],
            [1.2, -0.1, -0.2],
        ],
        dtype=float,
    )
    target = source + np.asarray([4.0, 0.25, -0.15])
    return source, target


def test_subspace_config_parses_cli_style_boolean_strings() -> None:
    config = subspace_adaptation_config(
        standardize="false",
        class_balance_source="0",
        normalize_latent="off",
    )

    assert config.standardize is False
    assert config.class_balance_source is False
    assert config.normalize_latent is False


def test_balanced_tca_alias_still_enables_source_class_balance() -> None:
    config = subspace_adaptation_config(method="balanced-transfer-component-analysis", class_balance_source="false")

    assert config.method == "balanced_tca"
    assert config.class_balance_source is True


def test_subspace_config_rejects_ambiguous_boolean_strings() -> None:
    with pytest.raises(ValueError, match="standardize must be a boolean value"):
        subspace_adaptation_config(standardize="definitely")


def test_fit_subspace_adaptation_honors_false_string_config_booleans() -> None:
    source, target = _shifted_domains()

    result = fit_subspace_adaptation(
        source,
        target,
        config={
            "n_components": 1,
            "standardize": "false",
            "class_balance_source": "false",
            "normalize_latent": "false",
        },
    )

    assert result.metadata["subspace_adaptation_standardize"] is False
    assert result.metadata["subspace_adaptation_class_balance_source"] is False
    assert result.metadata["subspace_adaptation_normalize_latent"] is False

from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.subspace_adaptation import fit_subspace_adaptation, subspace_adaptation_config


def _toy_domains() -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(
        [
            [-1.0, -0.1, 0.0],
            [-0.8, 0.1, 0.2],
            [-1.2, 0.0, -0.1],
            [1.0, 0.1, 0.0],
            [1.2, -0.1, -0.2],
            [0.8, 0.0, 0.1],
        ],
        dtype=float,
    )
    target = source + np.asarray([4.0, 0.25, -0.15])
    return source, target


def test_subspace_config_string_false_flags_are_false() -> None:
    config = subspace_adaptation_config(
        standardize="false",
        class_balance_source="false",
        normalize_latent="false",
    )

    assert config.standardize is False
    assert config.class_balance_source is False
    assert config.normalize_latent is False

    source, target = _toy_domains()
    result = fit_subspace_adaptation(
        source,
        target,
        config={
            "standardize": "false",
            "class_balance_source": "false",
            "normalize_latent": "false",
        },
        n_components=1,
    )

    assert result.metadata["subspace_adaptation_standardize"] is False
    assert result.metadata["subspace_adaptation_class_balance_source"] is False
    assert result.metadata["subspace_adaptation_normalize_latent"] is False
    assert result.metadata["subspace_adaptation_uses_source_labels"] is False


def test_subspace_config_numeric_and_alias_false_values_are_false() -> None:
    config = subspace_adaptation_config(
        standardize=0,
        class_balance_source="off",
        normalize_latent=np.asarray(False),
    )

    assert config.standardize is False
    assert config.class_balance_source is False
    assert config.normalize_latent is False


def test_subspace_config_true_values_and_balanced_method() -> None:
    config = subspace_adaptation_config(
        standardize="yes",
        class_balance_source="1",
        normalize_latent=1.0,
    )

    assert config.standardize is True
    assert config.class_balance_source is True
    assert config.normalize_latent is True

    balanced = subspace_adaptation_config(method="balanced_tca", class_balance_source="false")
    assert balanced.class_balance_source is True


@pytest.mark.parametrize("option", ["standardize", "class_balance_source", "normalize_latent"])
def test_subspace_config_rejects_ambiguous_boolean_strings(option: str) -> None:
    with pytest.raises(ValueError, match=option):
        subspace_adaptation_config(**{option: "maybe"})


@pytest.mark.parametrize("option", ["standardize", "class_balance_source", "normalize_latent"])
def test_subspace_config_rejects_non_scalar_boolean_arrays(option: str) -> None:
    with pytest.raises(ValueError, match=option):
        subspace_adaptation_config(**{option: np.asarray([True, False])})

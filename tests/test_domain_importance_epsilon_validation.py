from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.domain_importance import (
    DomainImportanceConfig,
    domain_importance_config,
    fit_domain_classifier_importance_weights,
)


@pytest.mark.parametrize("epsilon", [0.0, -1e-9, 0.5, 1.0, np.inf, np.nan])
def test_domain_importance_config_rejects_out_of_range_epsilon(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        domain_importance_config(epsilon=epsilon)


@pytest.mark.parametrize(
    "epsilon",
    [
        True,
        False,
        np.bool_(True),
        np.asarray(True),
        np.asarray([1e-6]),
        [1e-6],
    ],
)
def test_domain_importance_config_rejects_boolean_or_nonscalar_epsilon(epsilon: object) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        domain_importance_config(epsilon=epsilon)


def test_domain_importance_config_accepts_numeric_scalar_array_epsilon() -> None:
    cfg = domain_importance_config(epsilon=np.asarray(1e-6))

    assert cfg.epsilon == pytest.approx(1e-6)


def test_fit_domain_classifier_importance_rejects_invalid_dataclass_epsilon() -> None:
    source = np.asarray([[0.0], [1.0], [2.0]], dtype=float)
    target = np.asarray([[0.5], [1.5], [2.5]], dtype=float)
    config = DomainImportanceConfig(epsilon=0.5)

    with pytest.raises(ValueError, match="epsilon"):
        fit_domain_classifier_importance_weights(source, target, config=config)

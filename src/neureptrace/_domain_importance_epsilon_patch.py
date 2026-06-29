"""Validate domain-importance probability clipping epsilon values."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_domain_importance_epsilon_patch_installed"
_EPSILON_MESSAGE = "epsilon must be a finite scalar probability clipping value in the open interval (0, 0.5)."


def _normalize_epsilon(value: Any) -> float:
    """Return a valid probability clipping epsilon for domain posteriors."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(_EPSILON_MESSAGE)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(_EPSILON_MESSAGE)
        if np.issubdtype(value.dtype, np.bool_):
            raise ValueError(_EPSILON_MESSAGE)
        value = value.item()
    try:
        epsilon = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_EPSILON_MESSAGE) from exc
    if not np.isfinite(epsilon) or epsilon <= 0.0 or epsilon >= 0.5:
        raise ValueError(_EPSILON_MESSAGE)
    return epsilon


def install() -> None:
    """Patch the public domain-importance API to validate epsilon as a probability bound."""

    from neureptrace.decoding import domain_importance

    if getattr(domain_importance, _PATCH_MARKER, False):
        return

    original_domain_importance_config = domain_importance.domain_importance_config
    original_fit_domain_classifier_importance_weights = domain_importance.fit_domain_classifier_importance_weights

    @wraps(original_domain_importance_config)
    def domain_importance_config(
        *,
        clip: object = domain_importance.DEFAULT_WEIGHT_CLIP,
        normalize: object = True,
        account_for_sample_priors: object = True,
        epsilon: object = domain_importance.DEFAULT_EPSILON,
    ):
        return original_domain_importance_config(
            clip=clip,
            normalize=normalize,
            account_for_sample_priors=account_for_sample_priors,
            epsilon=_normalize_epsilon(epsilon),
        )

    @wraps(original_fit_domain_classifier_importance_weights)
    def fit_domain_classifier_importance_weights(source_features, target_features, *, estimator=None, config=None):
        if isinstance(config, domain_importance.DomainImportanceConfig):
            _normalize_epsilon(config.epsilon)
        return original_fit_domain_classifier_importance_weights(
            source_features,
            target_features,
            estimator=estimator,
            config=config,
        )

    domain_importance.domain_importance_config = domain_importance_config
    domain_importance.fit_domain_classifier_importance_weights = fit_domain_classifier_importance_weights
    setattr(domain_importance, _PATCH_MARKER, True)


__all__ = ["install"]

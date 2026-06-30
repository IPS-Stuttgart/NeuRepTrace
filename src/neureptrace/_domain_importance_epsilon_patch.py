"""Validate domain-importance probability clipping epsilon values."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_domain_importance_epsilon_patch_installed"
_EPSILON_MESSAGE = "epsilon must be a finite scalar probability clipping value in the open interval (0, 0.5)."
_PROBA_METHOD = "predict" + "_proba"


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


def _checked_target_domain_probabilities(model, features: np.ndarray, *, epsilon: float) -> np.ndarray:
    """Return checked target-domain posteriors from an sklearn-like estimator."""

    probability_method = getattr(model, _PROBA_METHOD, None)
    if probability_method is None:
        raise TypeError("Domain importance weighting requires an estimator with a probability prediction method.")
    probabilities = np.asarray(probability_method(features), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1])).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[0] != features.shape[0]:
        raise ValueError("Domain classifier returned an invalid probability matrix.")
    if probabilities.shape[1] != classes.shape[0]:
        raise ValueError("Domain classifier classes_ length must match probability columns.")
    source_columns = np.flatnonzero(classes == 0)
    target_columns = np.flatnonzero(classes == 1)
    if probabilities.shape[1] != 2 or source_columns.size != 1 or target_columns.size != 1:
        raise ValueError("Domain classifier classes_ must contain exactly source-domain label 0 and target-domain label 1.")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("Domain classifier probabilities must be finite and non-negative.")
    row_sums = np.sum(probabilities, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-8):
        raise ValueError("Domain classifier probability rows must sum to 1.")
    target_probability = probabilities[:, int(target_columns[0])]
    return np.clip(target_probability, epsilon, 1.0 - epsilon)


def install() -> None:
    """Patch the public domain-importance API to validate probability clipping and estimator outputs."""

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
    domain_importance._target_domain_probabilities = _checked_target_domain_probabilities
    setattr(domain_importance, _PATCH_MARKER, True)


__all__ = ["install"]

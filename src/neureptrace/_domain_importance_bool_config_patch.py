from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

from neureptrace.decoding import domain_importance as _domain_importance

_PATCH_MARKER = "_neureptrace_domain_importance_bool_config_patch_installed"


def _bool_config(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be a boolean value.")
        return _bool_config(value.item(), name=name)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")


def _normalize_domain_importance_config(
    config: _domain_importance.DomainImportanceConfig,
) -> _domain_importance.DomainImportanceConfig:
    """Normalize direct dataclass configs through the public config validators."""

    return _domain_importance.DomainImportanceConfig(
        clip=_domain_importance._normalize_clip(config.clip),
        normalize=_bool_config(config.normalize, name="normalize"),
        account_for_sample_priors=_bool_config(
            config.account_for_sample_priors,
            name="account_for_sample_priors",
        ),
        epsilon=_domain_importance._probability_clipping_epsilon(config.epsilon),
    )


def _patched_domain_importance_config(
    *,
    clip: object = _domain_importance.DEFAULT_WEIGHT_CLIP,
    normalize: object = True,
    account_for_sample_priors: object = True,
    epsilon: object = _domain_importance.DEFAULT_EPSILON,
) -> _domain_importance.DomainImportanceConfig:
    return _domain_importance.DomainImportanceConfig(
        clip=_domain_importance._normalize_clip(clip),
        normalize=_bool_config(normalize, name="normalize"),
        account_for_sample_priors=_bool_config(account_for_sample_priors, name="account_for_sample_priors"),
        epsilon=_domain_importance._probability_clipping_epsilon(epsilon),
    )


def _patched_fit_domain_classifier_importance_weights(
    original_fit,
):
    @wraps(original_fit)
    def fit_domain_classifier_importance_weights(
        source_features,
        target_features,
        *,
        estimator=None,
        config=None,
    ):
        if isinstance(config, _domain_importance.DomainImportanceConfig):
            config = _normalize_domain_importance_config(config)
        return original_fit(source_features, target_features, estimator=estimator, config=config)

    setattr(fit_domain_classifier_importance_weights, _PATCH_MARKER, True)
    return fit_domain_classifier_importance_weights


def install() -> None:
    if not getattr(_domain_importance.domain_importance_config, _PATCH_MARKER, False):
        setattr(_patched_domain_importance_config, _PATCH_MARKER, True)
        _domain_importance.domain_importance_config = _patched_domain_importance_config

    if not getattr(_domain_importance.fit_domain_classifier_importance_weights, _PATCH_MARKER, False):
        _domain_importance.fit_domain_classifier_importance_weights = _patched_fit_domain_classifier_importance_weights(
            _domain_importance.fit_domain_classifier_importance_weights,
        )

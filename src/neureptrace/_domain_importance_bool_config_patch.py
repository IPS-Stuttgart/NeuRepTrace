from __future__ import annotations

from typing import Any

import numpy as np

from neureptrace.decoding import domain_importance as _domain_importance


def _bool_config(value: Any, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")


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
        epsilon=_domain_importance._positive_float(epsilon, name="epsilon"),
    )


def install() -> None:
    _domain_importance.domain_importance_config = _patched_domain_importance_config

"""Runtime patch for strict KMM numeric parameter validation."""

from __future__ import annotations

from functools import wraps
from typing import Any

import numpy as np

_GAMMA_ERROR = "KMM gamma must be positive and finite, or one of: median, auto, scale."
_EPSILON_ERROR = "KMM epsilon must be finite and non-negative, 'auto', or None."
_PATCH_MARKER = "_neureptrace_kmm_bool_validation_patched"


def _is_boolean(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def install() -> None:
    """Reject boolean KMM numeric parameters before Python coerces them to floats."""

    from neureptrace.decoding import kernel_mean_matching as kmm

    if getattr(kmm.resolve_kmm_gamma, _PATCH_MARKER, False):
        return

    original_resolve_kmm_gamma = kmm.resolve_kmm_gamma
    original_normalize_kmm_epsilon = kmm.normalize_kmm_epsilon
    original_kmm_config = kmm.kmm_config

    @wraps(original_resolve_kmm_gamma)
    def resolve_kmm_gamma(value: float | str, source_features: Any, target_features: Any) -> float:
        if _is_boolean(value):
            raise ValueError(_GAMMA_ERROR)
        return original_resolve_kmm_gamma(value, source_features, target_features)

    @wraps(original_normalize_kmm_epsilon)
    def normalize_kmm_epsilon(value: float | str | None, *, n_source: int) -> float | None:
        if _is_boolean(value):
            raise ValueError(_EPSILON_ERROR)
        return original_normalize_kmm_epsilon(value, n_source=n_source)

    @wraps(original_kmm_config)
    def kmm_config(
        *,
        kernel: str | None = "rbf",
        gamma: float | str = kmm.DEFAULT_KMM_GAMMA,
        max_weight: float | str = kmm.DEFAULT_KMM_MAX_WEIGHT,
        epsilon: float | str | None = kmm.DEFAULT_KMM_EPSILON,
        regularization: float | str = kmm.DEFAULT_KMM_REGULARIZATION,
        max_iter: int | str = kmm.DEFAULT_KMM_MAX_ITER,
        tol: float | str = kmm.DEFAULT_KMM_TOL,
        normalize: bool = True,
        class_balance: bool = False,
    ) -> kmm.KernelMeanMatchingConfig:
        if _is_boolean(gamma):
            raise ValueError(_GAMMA_ERROR)
        if _is_boolean(epsilon):
            raise ValueError(_EPSILON_ERROR)
        return original_kmm_config(
            kernel=kernel,
            gamma=gamma,
            max_weight=max_weight,
            epsilon=epsilon,
            regularization=regularization,
            max_iter=max_iter,
            tol=tol,
            normalize=normalize,
            class_balance=class_balance,
        )

    setattr(resolve_kmm_gamma, _PATCH_MARKER, True)
    setattr(normalize_kmm_epsilon, _PATCH_MARKER, True)
    setattr(kmm_config, _PATCH_MARKER, True)
    kmm.resolve_kmm_gamma = resolve_kmm_gamma
    kmm.normalize_kmm_epsilon = normalize_kmm_epsilon
    kmm.kmm_config = kmm_config


__all__ = ["install"]

"""Runtime patch for sign-flip scalar control validation."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PERMUTATION_COUNT_ERROR = "n_permutations must be a positive integer."
_RANDOM_STATE_ERROR = "random_state must be a non-negative integer seed."
_CLUSTER_ALPHA_ERROR = "cluster_alpha must be between 0 and 1."
_PATCH_MARKER = "_sign_flip_scalar_controls_patched"


def _scalar_float(value: object, error_message: str) -> float:
    """Return a float from a scalar-like numeric value, rejecting arrays and booleans."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(error_message)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error_message) from exc
    if array.ndim != 0:
        raise ValueError(error_message)
    scalar = array.item()
    if isinstance(scalar, (bool, np.bool_)):
        raise ValueError(error_message)
    try:
        return float(scalar)
    except (TypeError, ValueError) as exc:
        raise ValueError(error_message) from exc


def _validate_positive_permutation_count(n_permutations: int) -> int:
    numeric = _scalar_float(n_permutations, _PERMUTATION_COUNT_ERROR)
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 1.0:
        raise ValueError(_PERMUTATION_COUNT_ERROR)
    return int(numeric)


def _validate_random_state(random_state: int) -> int:
    numeric = _scalar_float(random_state, _RANDOM_STATE_ERROR)
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 0.0:
        raise ValueError(_RANDOM_STATE_ERROR)
    return int(numeric)


def _validate_cluster_alpha(cluster_alpha: float) -> float:
    numeric = _scalar_float(cluster_alpha, _CLUSTER_ALPHA_ERROR)
    if not np.isfinite(numeric) or not 0.0 < numeric < 1.0:
        raise ValueError(_CLUSTER_ALPHA_ERROR)
    return float(numeric)


def _patch_inference() -> None:
    import neureptrace.inference as inference

    if not getattr(inference._validate_positive_permutation_count, _PATCH_MARKER, False):
        _validate_positive_permutation_count._sign_flip_scalar_controls_patched = True  # type: ignore[attr-defined]
        inference._validate_positive_permutation_count = _validate_positive_permutation_count

    if not getattr(inference._sign_flip_t_statistics, _PATCH_MARKER, False):
        original_sign_flip_t_statistics = inference._sign_flip_t_statistics

        @wraps(original_sign_flip_t_statistics)
        def _sign_flip_t_statistics(effects: np.ndarray, *, n_permutations: int, random_state: int) -> np.ndarray:
            return original_sign_flip_t_statistics(
                effects,
                n_permutations=_validate_positive_permutation_count(n_permutations),
                random_state=_validate_random_state(random_state),
            )

        _sign_flip_t_statistics._sign_flip_scalar_controls_patched = True  # type: ignore[attr-defined]
        inference._sign_flip_t_statistics = _sign_flip_t_statistics

    if not getattr(inference.sign_flip_time_inference, _PATCH_MARKER, False):
        original_sign_flip_time_inference = inference.sign_flip_time_inference

        @wraps(original_sign_flip_time_inference)
        def sign_flip_time_inference(*args: object, **kwargs: object):
            validated_kwargs = dict(kwargs)
            if "n_permutations" in validated_kwargs:
                validated_kwargs["n_permutations"] = _validate_positive_permutation_count(validated_kwargs["n_permutations"])
            if "random_state" in validated_kwargs:
                validated_kwargs["random_state"] = _validate_random_state(validated_kwargs["random_state"])
            if "cluster_alpha" in validated_kwargs:
                validated_kwargs["cluster_alpha"] = _validate_cluster_alpha(validated_kwargs["cluster_alpha"])
            return original_sign_flip_time_inference(*args, **validated_kwargs)

        sign_flip_time_inference._sign_flip_scalar_controls_patched = True  # type: ignore[attr-defined]
        inference.sign_flip_time_inference = sign_flip_time_inference


def _patch_paired_stats() -> None:
    import neureptrace.paired_stats as paired_stats

    if not getattr(paired_stats._validate_positive_permutation_count, _PATCH_MARKER, False):
        _validate_positive_permutation_count._sign_flip_scalar_controls_patched = True  # type: ignore[attr-defined]
        paired_stats._validate_positive_permutation_count = _validate_positive_permutation_count
    if not getattr(paired_stats._validate_random_state, _PATCH_MARKER, False):
        _validate_random_state._sign_flip_scalar_controls_patched = True  # type: ignore[attr-defined]
        paired_stats._validate_random_state = _validate_random_state


def install() -> None:
    """Install strict scalar control validators for sign-flip inference APIs."""
    _patch_inference()
    _patch_paired_stats()


__all__ = ["install"]

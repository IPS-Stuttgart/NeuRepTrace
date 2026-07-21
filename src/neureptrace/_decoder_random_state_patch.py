"""Patch decoder random-state propagation and exact temporal integer controls."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_decoder_random_state_patch_installed"
_TEMPORAL_INTEGER_PATCH_MARKER = "_neureptrace_temporal_integer_patch_installed"


def _nested_random_state_updates(estimator: Any, random_state: int | None) -> dict[str, int | None]:
    """Return set_params updates for every nested random_state parameter."""

    if not (hasattr(estimator, "get_params") and hasattr(estimator, "set_params")):
        return {}
    try:
        params = estimator.get_params(deep=True)
    except Exception:  # pragma: no cover - third-party estimator defensive guard
        return {}
    return {
        name: random_state
        for name in params
        if name == "random_state" or name.endswith("__random_state")
    }


def _propagate_random_state(estimator: Any, random_state: int | None) -> Any:
    updates = _nested_random_state_updates(estimator, random_state)
    if updates:
        estimator.set_params(**updates)
    return estimator


def _patch_decoder_factories() -> None:
    from neureptrace import decoding

    if getattr(decoding, _PATCH_MARKER, False):
        return

    original_make_decoder = decoding.make_decoder
    original_make_tuned_decoder = decoding.make_tuned_decoder

    @wraps(original_make_decoder)
    def make_decoder(*args: Any, **kwargs: Any):
        random_state = kwargs.get("random_state", 13)
        estimator = original_make_decoder(*args, **kwargs)
        return _propagate_random_state(estimator, random_state)

    @wraps(original_make_tuned_decoder)
    def make_tuned_decoder(*args: Any, **kwargs: Any):
        random_state = kwargs.get("random_state", 13)
        estimator = original_make_tuned_decoder(*args, **kwargs)
        return _propagate_random_state(estimator, random_state)

    decoding.make_decoder = make_decoder
    decoding.make_tuned_decoder = make_tuned_decoder
    setattr(decoding, _PATCH_MARKER, True)


def _patch_temporal_integer_validation() -> None:
    from neureptrace import temporal_model

    original_validate_integer = temporal_model._validate_integer
    if getattr(original_validate_integer, _TEMPORAL_INTEGER_PATCH_MARKER, False):
        return

    @wraps(original_validate_integer)
    def _validate_integer(
        value: int | str,
        *,
        name: str,
        minimum: int | None = None,
    ) -> int:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be an integer.")

        if isinstance(value, (int, np.integer)):
            parsed = int(value)
        elif isinstance(value, str):
            try:
                numeric = Decimal(value.strip())
            except (InvalidOperation, ValueError) as exc:
                raise ValueError(f"{name} must be an integer.") from exc
            if not numeric.is_finite() or numeric != numeric.to_integral_value():
                raise ValueError(f"{name} must be an integer.")
            parsed = int(numeric)
        else:
            return original_validate_integer(value, name=name, minimum=minimum)

        if minimum is not None and parsed < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")
        return parsed

    setattr(_validate_integer, _TEMPORAL_INTEGER_PATCH_MARKER, True)
    temporal_model._validate_integer = _validate_integer


def install() -> None:
    """Install decoder propagation and exact temporal integer validation."""

    _patch_decoder_factories()
    _patch_temporal_integer_validation()


__all__ = ["install"]

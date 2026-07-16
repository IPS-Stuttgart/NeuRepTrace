"""Normalize source configuration aliases and preserve exact integer controls."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

import numpy as np

_PATCH_ATTR = "_neureptrace_source_numpy_string_alias_config_patch"
_INTEGER_PRECISION_MARKER = "_source_scaling_integer_precision_patched"
_ALIAS_VALUES = {"all", "full"}


def _scalar_or_original(value: Any) -> Any:
    """Return a NumPy scalar array's scalar value without accepting vectors."""

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            return value
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _alias_or_none(value: Any) -> str | None:
    scalar = _scalar_or_original(value)
    if not isinstance(scalar, str):
        return None
    alias = scalar.strip().lower()
    return alias if alias in _ALIAS_VALUES else None


def _integer(value: Any, *, name: str) -> int:
    """Normalize integer options without losing values above float precision."""

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be an integer.")
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        try:
            number = Decimal(value.strip())
        except InvalidOperation as exc:
            raise ValueError(f"{name} must be an integer.") from exc
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError(f"{name} must be an integer.")
        return int(number)
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(parsed)


def install() -> None:
    """Accept source aliases and preserve exact feature-scaling integers."""

    from neureptrace.decoding import source_knn, source_pca, source_polynomial, source_scaling

    current_integer = source_scaling._integer
    if not getattr(current_integer, _INTEGER_PRECISION_MARKER, False):
        setattr(_integer, _INTEGER_PRECISION_MARKER, True)
        _integer.__wrapped__ = current_integer
        source_scaling._integer = _integer

    original_component_request = source_pca._component_request
    if getattr(original_component_request, _PATCH_ATTR, False):
        return

    original_k_request = source_knn._normalize_k_request
    original_max_interactions = source_polynomial._max_interactions_value

    def _component_request(value: Any) -> int | str:
        alias = _alias_or_none(value)
        if alias is not None:
            return alias
        return original_component_request(value)

    def _normalize_k_request(value: Any) -> int | str:
        alias = _alias_or_none(value)
        if alias is not None:
            return alias
        return original_k_request(value)

    def _max_interactions_value(value: Any) -> int | str:
        alias = _alias_or_none(value)
        if alias is not None:
            return source_polynomial.DEFAULT_MAX_INTERACTIONS
        return original_max_interactions(value)

    for patched, original in (
        (_component_request, original_component_request),
        (_normalize_k_request, original_k_request),
        (_max_interactions_value, original_max_interactions),
    ):
        setattr(patched, _PATCH_ATTR, True)
        patched.__wrapped__ = original

    source_pca._component_request = _component_request
    source_knn._normalize_k_request = _normalize_k_request
    source_polynomial._max_interactions_value = _max_interactions_value


__all__ = ["install"]

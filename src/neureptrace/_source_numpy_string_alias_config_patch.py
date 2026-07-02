"""Normalize zero-dimensional NumPy string aliases for source config controls."""

from __future__ import annotations

from typing import Any

import numpy as np

_PATCH_ATTR = "_neureptrace_source_numpy_string_alias_config_patch"
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


def install() -> None:
    """Accept scalar NumPy string aliases where plain string aliases are valid."""

    from neureptrace.decoding import source_knn, source_pca, source_polynomial

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

"""Normalize source configuration aliases and preserve exact numeric controls."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from functools import wraps
from typing import Any

import numpy as np

_PATCH_ATTR = "_neureptrace_source_numpy_string_alias_config_patch"
_INTEGER_PRECISION_MARKER = "_source_scaling_integer_precision_patched"
_BALANCING_CONFIG_MARKER = "_source_class_balancing_config_normalized"
_THRESHOLD_OUTPUT_MARKER = "_source_threshold_output_precision_patched"
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


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when finite nonzero values survive conversion."""

    array = np.asarray(values)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return array
    if np.any((array != 0.0) & (compact == 0.0)):
        return array
    return compact


def _install_source_threshold_output_patch(source_threshold: Any) -> None:
    """Preserve accepted finite threshold output values outside float32 range."""

    current = source_threshold.fit_source_threshold_transform
    if getattr(current, _THRESHOLD_OUTPUT_MARKER, False):
        return

    @wraps(current)
    def fit_source_threshold_transform(
        *,
        source_features: Any,
        test_features: Any,
        config: Any = None,
    ) -> Any:
        cfg = source_threshold.source_threshold_config() if config is None else source_threshold._coerce_config(config)
        source = source_threshold._feature_matrix(source_features, name="source_features")
        test = source_threshold._feature_matrix(test_features, name="test_features")
        if source.shape[1] != test.shape[1]:
            raise ValueError(
                "source_features and test_features must have the same feature width: "
                f"{source.shape[1]} != {test.shape[1]}."
            )
        threshold_map = source_threshold.fit_source_threshold_map(source, config=cfg)
        train = source_threshold.apply_source_threshold_transform(source, threshold_map)
        test_out = source_threshold.apply_source_threshold_transform(test, threshold_map)
        return source_threshold.SourceThresholdResult(
            train_features=_compact_float32(train),
            test_features=_compact_float32(test_out),
            threshold_map=threshold_map,
            metadata=source_threshold._metadata(
                cfg,
                n_source_rows=source.shape[0],
                n_test_rows=test.shape[0],
                feature_dim=source.shape[1],
            ),
        )

    setattr(fit_source_threshold_transform, _THRESHOLD_OUTPUT_MARKER, True)
    fit_source_threshold_transform.__wrapped__ = current
    source_threshold.fit_source_threshold_transform = fit_source_threshold_transform


def install() -> None:
    """Accept source aliases and preserve exact numeric controls."""

    from neureptrace.decoding import (
        source_balancing,
        source_knn,
        source_pca,
        source_polynomial,
        source_scaling,
        source_threshold,
    )

    current_integer = source_scaling._integer
    if not getattr(current_integer, _INTEGER_PRECISION_MARKER, False):
        setattr(_integer, _INTEGER_PRECISION_MARKER, True)
        _integer.__wrapped__ = current_integer
        source_scaling._integer = _integer

    current_balancing_coercer = source_balancing._coerce_config
    if not getattr(current_balancing_coercer, _BALANCING_CONFIG_MARKER, False):

        def _coerce_balancing_config(config: Any):
            if isinstance(config, source_balancing.SourceClassBalancingConfig):
                return source_balancing.source_class_balancing_config(
                    mode=config.mode,
                    target_count=config.target_count,
                    random_state=config.random_state,
                    preserve_order=config.preserve_order,
                )
            return current_balancing_coercer(config)

        setattr(_coerce_balancing_config, _BALANCING_CONFIG_MARKER, True)
        _coerce_balancing_config.__wrapped__ = current_balancing_coercer
        source_balancing._coerce_config = _coerce_balancing_config

    _install_source_threshold_output_patch(source_threshold)

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

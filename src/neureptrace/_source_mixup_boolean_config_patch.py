"""Normalize Source MixUp boolean config values and composite row identifiers."""

from __future__ import annotations

import importlib
from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_mixup_boolean_config_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_LABEL_TOKEN_PREFIX = "__neureptrace_source_mixup_label_"


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _normalize_bool(value: Any, *, name: str) -> bool:
    """Return a real bool while rejecting ambiguous truthy/falsy objects."""

    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_STRINGS:
            return True
        if normalized in _FALSE_STRINGS:
            return False
        raise _bool_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _bool_error(name)
        return _normalize_bool(value.item(), name=name)
    if isinstance(value, (int, np.integer)):
        if int(value) in {0, 1}:
            return bool(value)
        raise _bool_error(name)
    if isinstance(value, (float, np.floating)):
        if np.isfinite(value) and float(value) in {0.0, 1.0}:
            return bool(value)
        raise _bool_error(name)
    raise _bool_error(name)


def _hashable_value(value: Any) -> Any:
    """Return a stable hashable representation for JSON/YAML-style IDs."""

    try:
        hash(value)
    except TypeError:
        if isinstance(value, np.ndarray):
            return tuple(_hashable_value(item) for item in value.tolist())
        if isinstance(value, list):
            return tuple(_hashable_value(item) for item in value)
        if isinstance(value, tuple):
            return tuple(_hashable_value(item) for item in value)
        if isinstance(value, dict):
            return tuple(
                sorted(
                    ((_hashable_value(key), _hashable_value(item)) for key, item in value.items()),
                    key=repr,
                )
            )
        return repr(value)
    return value


def _object_vector(items: list[Any]) -> np.ndarray:
    vector = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        vector[index] = item
    return vector


def _atomic_vector(values: Any, *, expected_length: int, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            items = [array.item()]
        elif array.ndim == 1:
            items = array.tolist()
        elif array.ndim == 2 and array.shape[0] == expected_length and array.shape[1] != 1:
            items = [tuple(row.tolist()) for row in array]
        elif array.ndim == 2 and (array.shape[1] == 1 or (array.shape[0] == 1 and array.shape[1] == expected_length)):
            items = array.reshape(-1).tolist()
        else:
            raise ValueError(f"{name} must be a vector or one composite row per feature row; got shape {array.shape}.")
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError:
            items = [values]
    if len(items) != expected_length:
        raise ValueError(f"{name} must contain one value per feature row: {len(items)} != {expected_length}.")
    return _object_vector(items)


def _values_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    try:
        return bool(np.all(result))
    except (TypeError, ValueError):
        return False


def _unique_values(values: np.ndarray) -> list[Any]:
    unique: list[Any] = []
    for value in values.tolist():
        if not any(_values_equal(value, existing) for existing in unique):
            unique.append(value)
    return unique


def _is_composite_label(value: Any) -> bool:
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray):
        return value.ndim != 0
    return isinstance(value, (tuple, list, dict))


def _tokenize_source_labels(source_labels: Any, *, expected_length: int) -> tuple[Any, dict[str, Any]]:
    labels = _atomic_vector(source_labels, expected_length=expected_length, name="source_labels")
    if not any(_is_composite_label(label) for label in labels.tolist()):
        return source_labels, {}
    unique = _unique_values(labels)
    token_to_label = {f"{_LABEL_TOKEN_PREFIX}{index}": label for index, label in enumerate(unique)}
    tokens = np.empty(labels.shape[0], dtype=object)
    for row, label in enumerate(labels.tolist()):
        match = next(index for index, known in enumerate(unique) if _values_equal(label, known))
        tokens[row] = f"{_LABEL_TOKEN_PREFIX}{match}"
    return tokens, token_to_label


def _restore_label_tokens(values: Any, token_to_label: dict[str, Any]) -> np.ndarray:
    vector = np.asarray(values, dtype=object).reshape(-1)
    if not token_to_label:
        return vector
    restored = np.empty(vector.shape[0], dtype=object)
    for index, value in enumerate(vector.tolist()):
        restored[index] = token_to_label.get(value, value)
    return restored


def _canonicalize_source_domains(source_domains: Any, *, expected_length: int | None = None) -> Any:
    """Keep row-level domain IDs hashable without flattening composite IDs."""

    if source_domains is None or isinstance(source_domains, (str, bytes)):
        return source_domains
    if isinstance(source_domains, np.ndarray) and expected_length is not None:
        return [_hashable_value(item) for item in _atomic_vector(source_domains, expected_length=expected_length, name="source_domains").tolist()]
    if isinstance(source_domains, np.ndarray):
        return source_domains
    try:
        items = list(source_domains)
    except TypeError:
        return source_domains
    return [_hashable_value(item) for item in items]


def install() -> None:
    """Install strict Source MixUp boolean option normalization."""

    source_mixup = importlib.import_module("neureptrace.decoding.source_mixup")

    original_config = source_mixup.source_mixup_config
    if not getattr(original_config, _PATCH_MARKER, False):

        @wraps(original_config)
        def source_mixup_config(
            *,
            synthetic_per_class: int | str = 0,
            alpha: float | str = source_mixup.DEFAULT_MIXUP_ALPHA,
            random_state: int | str | None = 13,
            same_class_partner: Any = True,
            cross_domain_partner: Any = True,
            hard_label_policy: str = "content",
            preserve_original: Any = True,
        ):
            return original_config(
                synthetic_per_class=synthetic_per_class,
                alpha=alpha,
                random_state=random_state,
                same_class_partner=_normalize_bool(same_class_partner, name="same_class_partner"),
                cross_domain_partner=_normalize_bool(cross_domain_partner, name="cross_domain_partner"),
                hard_label_policy=hard_label_policy,
                preserve_original=_normalize_bool(preserve_original, name="preserve_original"),
            )

        setattr(source_mixup_config, _PATCH_MARKER, True)
        source_mixup.source_mixup_config = source_mixup_config

    original_augment = source_mixup.augment_source_with_mixup
    if not getattr(original_augment, _PATCH_MARKER, False):

        def _coerce_config_with_bool(config: Any):
            if config is None:
                return None
            cfg = source_mixup._coerce_config(config)
            return replace(
                cfg,
                same_class_partner=_normalize_bool(cfg.same_class_partner, name="same_class_partner"),
                cross_domain_partner=_normalize_bool(cfg.cross_domain_partner, name="cross_domain_partner"),
                preserve_original=_normalize_bool(cfg.preserve_original, name="preserve_original"),
            )

        @wraps(original_augment)
        def augment_source_with_mixup(
            source_features,
            source_labels,
            *,
            source_domains=None,
            config: Any = None,
        ):
            n_rows = source_mixup._feature_matrix(source_features, name="source_features").shape[0]
            token_labels, token_to_label = _tokenize_source_labels(source_labels, expected_length=n_rows)
            result = original_augment(
                source_features,
                token_labels,
                source_domains=_canonicalize_source_domains(source_domains, expected_length=n_rows),
                config=_coerce_config_with_bool(config),
            )
            if not token_to_label:
                return result
            return replace(
                result,
                labels=_restore_label_tokens(result.labels, token_to_label),
                classes=_restore_label_tokens(result.classes, token_to_label),
            )

        setattr(augment_source_with_mixup, _PATCH_MARKER, True)
        source_mixup.augment_source_with_mixup = augment_source_with_mixup


__all__ = ["install"]

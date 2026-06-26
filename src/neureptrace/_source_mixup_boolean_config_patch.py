"""Normalize Source MixUp boolean config values from CLI/YAML-style inputs."""

from __future__ import annotations

import importlib
from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_mixup_boolean_config_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


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


def _canonicalize_source_domains(source_domains: Any) -> Any:
    """Keep row-level domain IDs hashable without flattening composite IDs."""

    if source_domains is None or isinstance(source_domains, (str, bytes, np.ndarray)):
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
            return original_augment(
                source_features,
                source_labels,
                source_domains=_canonicalize_source_domains(source_domains),
                config=_coerce_config_with_bool(config),
            )

        setattr(augment_source_with_mixup, _PATCH_MARKER, True)
        source_mixup.augment_source_with_mixup = augment_source_with_mixup


__all__ = ["install"]

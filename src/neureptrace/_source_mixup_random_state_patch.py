"""Validate Source MixUp numeric config and random-state values before RNG construction."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_mixup_random_state_patch_installed"
_NONE_STRINGS = {"", "none", "null"}


def _random_state_error() -> ValueError:
    return ValueError("random_state must be a non-negative integer or none.")


def _is_none_like_random_state(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _NONE_STRINGS
    return False


def _normalize_optional_random_state(value: Any, *, normalizer: Any) -> int | None:
    """Normalize optional random-state values accepted by NumPy SeedSequence."""

    if _is_none_like_random_state(value):
        return None
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _random_state_error()
        value = value.item()
        if _is_none_like_random_state(value):
            return None
    if isinstance(value, (list, tuple, dict, set)):
        raise _random_state_error()
    try:
        return normalizer(value, name="random_state")
    except ValueError as exc:
        raise _random_state_error() from exc


def _scalar_config_value(value: Any, *, name: str, expected: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(f"{name} must be {expected}.")
        value = value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise ValueError(f"{name} must be {expected}.")
    return value


def _numeric_scalar(value: Any, *, name: str, expected: str) -> float:
    scalar_value = _scalar_config_value(value, name=name, expected=expected)
    if isinstance(scalar_value, (bool, np.bool_)):
        raise ValueError(f"{name} must be {expected}.")
    try:
        numeric = float(scalar_value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be {expected}.") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be {expected}.")
    return numeric


def _normalize_integer(value: Any, *, name: str) -> int:
    numeric = _numeric_scalar(value, name=name, expected="an integer")
    if numeric % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(numeric)


def _normalize_nonnegative_int(value: Any, *, name: str) -> int:
    integer = _normalize_integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be non-negative.")
    return integer


def _positive_float(value: Any, *, name: str) -> float:
    numeric = _numeric_scalar(value, name=name, expected="positive and finite")
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return numeric


def _validate_mixup_numeric_values(*, synthetic_per_class: Any, alpha: Any) -> None:
    _normalize_nonnegative_int(synthetic_per_class, name="synthetic_per_class")
    _positive_float(alpha, name="alpha")


def _validate_mixup_config_numeric_values(source_mixup: Any, config: Any) -> None:
    if config is None:
        return
    if isinstance(config, source_mixup.SourceMixUpConfig):
        _validate_mixup_numeric_values(synthetic_per_class=config.synthetic_per_class, alpha=config.alpha)
        return
    if isinstance(config, Mapping):
        if "synthetic_per_class" in config:
            _normalize_nonnegative_int(config["synthetic_per_class"], name="synthetic_per_class")
        if "alpha" in config:
            _positive_float(config["alpha"], name="alpha")


def install() -> None:
    """Install early Source MixUp/SMOTE random-state and numeric-config validation."""

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
            seed = _normalize_optional_random_state(
                random_state,
                normalizer=source_mixup._normalize_nonnegative_int,
            )
            return original_config(
                synthetic_per_class=_normalize_nonnegative_int(synthetic_per_class, name="synthetic_per_class"),
                alpha=_positive_float(alpha, name="alpha"),
                random_state=seed,
                same_class_partner=same_class_partner,
                cross_domain_partner=cross_domain_partner,
                hard_label_policy=hard_label_policy,
                preserve_original=preserve_original,
            )

        setattr(source_mixup_config, _PATCH_MARKER, True)
        source_mixup.source_mixup_config = source_mixup_config

    original_augment = source_mixup.augment_source_with_mixup
    if not getattr(original_augment, _PATCH_MARKER, False):

        def _coerce_config_with_seed(config: Any):
            if config is None:
                return None
            _validate_mixup_config_numeric_values(source_mixup, config)
            cfg = source_mixup._coerce_config(config)
            seed = _normalize_optional_random_state(
                cfg.random_state,
                normalizer=source_mixup._normalize_nonnegative_int,
            )
            return replace(
                cfg,
                synthetic_per_class=_normalize_nonnegative_int(cfg.synthetic_per_class, name="synthetic_per_class"),
                alpha=_positive_float(cfg.alpha, name="alpha"),
                random_state=seed,
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
                source_domains=source_domains,
                config=_coerce_config_with_seed(config),
            )

        setattr(augment_source_with_mixup, _PATCH_MARKER, True)
        source_mixup.augment_source_with_mixup = augment_source_with_mixup

    source_smote = importlib.import_module("neureptrace.decoding.source_smote")

    original_smote_config = source_smote.source_smote_config
    if not getattr(original_smote_config, _PATCH_MARKER, False):

        @wraps(original_smote_config)
        def source_smote_config(
            *,
            synthetic_per_class: int | str = 0,
            cross_domain_partner: bool | int | str = True,
            preserve_original: bool | int | str = True,
            random_state: int | str | None = 13,
            jitter_std: float | str = 0.0,
        ):
            seed = _normalize_optional_random_state(
                random_state,
                normalizer=source_smote._nonnegative_int,
            )
            return original_smote_config(
                synthetic_per_class=synthetic_per_class,
                cross_domain_partner=cross_domain_partner,
                preserve_original=preserve_original,
                random_state=seed,
                jitter_std=jitter_std,
            )

        setattr(source_smote_config, _PATCH_MARKER, True)
        source_smote.source_smote_config = source_smote_config

    original_smote_augment = source_smote.augment_source_with_smote
    if not getattr(original_smote_augment, _PATCH_MARKER, False):

        def _coerce_smote_config_with_seed(config: Any):
            if config is None:
                return None
            cfg = source_smote._coerce_config(config)
            seed = _normalize_optional_random_state(
                cfg.random_state,
                normalizer=source_smote._nonnegative_int,
            )
            return replace(cfg, random_state=seed)

        @wraps(original_smote_augment)
        def augment_source_with_smote(
            source_features,
            source_labels,
            *,
            source_domains=None,
            config: Any = None,
        ):
            return original_smote_augment(
                source_features,
                source_labels,
                source_domains=source_domains,
                config=_coerce_smote_config_with_seed(config),
            )

        setattr(augment_source_with_smote, _PATCH_MARKER, True)
        source_smote.augment_source_with_smote = augment_source_with_smote


__all__ = ["install"]

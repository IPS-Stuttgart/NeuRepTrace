"""Normalize MixStyle scalar and boolean config values from CLI/YAML-style inputs."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_mixstyle_boolean_config_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}


def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _scalar_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a scalar configuration value.")


def _reject_array_valued(value: Any, *, name: str) -> Any:
    """Reject NumPy arrays before scalar validators can coerce them."""

    if isinstance(value, np.ndarray):
        raise _scalar_error(name)
    return value


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


def _patch_feature_mixstyle() -> None:
    mixstyle = importlib.import_module("neureptrace.decoding.mixstyle")

    original_config = mixstyle.source_mixstyle_config
    if not getattr(original_config, _PATCH_MARKER, False):

        @wraps(original_config)
        def source_mixstyle_config(
            *,
            augmentations_per_row: int | str = mixstyle.DEFAULT_MIXSTYLE_AUGMENTATIONS_PER_ROW,
            alpha: float | str = mixstyle.DEFAULT_MIXSTYLE_ALPHA,
            random_state: int | str | None = mixstyle.DEFAULT_MIXSTYLE_RANDOM_STATE,
            domain_pairing: str = "shuffle",
            preserve_domain_mean: Any = False,
            class_conditional: Any = False,
            include_original: Any = True,
        ):
            return original_config(
                augmentations_per_row=_reject_array_valued(augmentations_per_row, name="augmentations_per_row"),
                alpha=_reject_array_valued(alpha, name="alpha"),
                random_state=_reject_array_valued(random_state, name="random_state"),
                domain_pairing=domain_pairing,
                preserve_domain_mean=_normalize_bool(preserve_domain_mean, name="preserve_domain_mean"),
                class_conditional=_normalize_bool(class_conditional, name="class_conditional"),
                include_original=_normalize_bool(include_original, name="include_original"),
            )

        setattr(source_mixstyle_config, _PATCH_MARKER, True)
        mixstyle.source_mixstyle_config = source_mixstyle_config

    original_augment = mixstyle.augment_source_mixstyle
    if not getattr(original_augment, _PATCH_MARKER, False):

        @wraps(original_augment)
        def augment_source_mixstyle(
            source_features,
            source_labels,
            source_domains,
            *,
            augmentations_per_row: int | str = mixstyle.DEFAULT_MIXSTYLE_AUGMENTATIONS_PER_ROW,
            alpha: float | str = mixstyle.DEFAULT_MIXSTYLE_ALPHA,
            random_state: int | str | None = mixstyle.DEFAULT_MIXSTYLE_RANDOM_STATE,
            domain_pairing: str = "shuffle",
            preserve_domain_mean: Any = False,
            class_conditional: Any = False,
            include_original: Any = True,
        ):
            return original_augment(
                source_features,
                source_labels,
                source_domains,
                augmentations_per_row=_reject_array_valued(augmentations_per_row, name="augmentations_per_row"),
                alpha=_reject_array_valued(alpha, name="alpha"),
                random_state=_reject_array_valued(random_state, name="random_state"),
                domain_pairing=domain_pairing,
                preserve_domain_mean=_normalize_bool(preserve_domain_mean, name="preserve_domain_mean"),
                class_conditional=_normalize_bool(class_conditional, name="class_conditional"),
                include_original=_normalize_bool(include_original, name="include_original"),
            )

        setattr(augment_source_mixstyle, _PATCH_MARKER, True)
        mixstyle.augment_source_mixstyle = augment_source_mixstyle


def _patch_domain_mixstyle() -> None:
    source_mixstyle = importlib.import_module("neureptrace.decoding.source_mixstyle")

    original_config = source_mixstyle.source_mixstyle_config
    if not getattr(original_config, _PATCH_MARKER, False):

        @wraps(original_config)
        def source_mixstyle_config(
            *,
            mixes_per_row: int | str = source_mixstyle.DEFAULT_MIXSTYLE_MIXES_PER_ROW,
            alpha: float | str = source_mixstyle.DEFAULT_MIXSTYLE_ALPHA,
            style_strength: float | str = source_mixstyle.DEFAULT_MIXSTYLE_STYLE_STRENGTH,
            synthetic_weight: float | str = source_mixstyle.DEFAULT_MIXSTYLE_SYNTHETIC_WEIGHT,
            include_original: Any = True,
            random_state: int | str | None = 13,
        ):
            return original_config(
                mixes_per_row=_reject_array_valued(mixes_per_row, name="mixes_per_row"),
                alpha=_reject_array_valued(alpha, name="alpha"),
                style_strength=_reject_array_valued(style_strength, name="style_strength"),
                synthetic_weight=_reject_array_valued(synthetic_weight, name="synthetic_weight"),
                include_original=_normalize_bool(include_original, name="include_original"),
                random_state=_reject_array_valued(random_state, name="random_state"),
            )

        setattr(source_mixstyle_config, _PATCH_MARKER, True)
        source_mixstyle.source_mixstyle_config = source_mixstyle_config

    original_augment = source_mixstyle.augment_source_domains_mixstyle
    if not getattr(original_augment, _PATCH_MARKER, False):

        def _coerce_config_with_bool(config: Any):
            if config is None:
                return None
            cfg = source_mixstyle._coerce_config(config)
            return source_mixstyle.source_mixstyle_config(
                mixes_per_row=cfg.mixes_per_row,
                alpha=cfg.alpha,
                style_strength=cfg.style_strength,
                synthetic_weight=cfg.synthetic_weight,
                include_original=_normalize_bool(cfg.include_original, name="include_original"),
                random_state=cfg.random_state,
            )

        @wraps(original_augment)
        def augment_source_domains_mixstyle(
            source_features,
            source_labels,
            source_domains,
            *,
            config: Any = None,
        ):
            return original_augment(
                source_features,
                source_labels,
                source_domains,
                config=_coerce_config_with_bool(config),
            )

        setattr(augment_source_domains_mixstyle, _PATCH_MARKER, True)
        source_mixstyle.augment_source_domains_mixstyle = augment_source_domains_mixstyle


def install() -> None:
    """Install strict MixStyle scalar and boolean option normalization."""

    _patch_feature_mixstyle()
    _patch_domain_mixstyle()


__all__ = ["install"]

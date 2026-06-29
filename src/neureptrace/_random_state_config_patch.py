"""Normalize optional random-state config values without unhashable sentinel checks."""

from __future__ import annotations

import importlib
from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

_INSTALLED = False
_PATCH_MARKER = "_neureptrace_random_state_config_patch_installed"


def _random_state_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a non-negative integer.")


def _is_none_random_state(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _scalar_random_state_value(value: Any, *, name: str) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _random_state_error(name)
        return value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise _random_state_error(name)
    return value


def _normalize_optional_nonnegative_int(
    value: Any,
    *,
    normalizer: Any,
    name: str = "random_state",
) -> int | None:
    if _is_none_random_state(value):
        return None
    return normalizer(_scalar_random_state_value(value, name=name), name=name)


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
            preserve_domain_mean: bool = False,
            class_conditional: bool = False,
            include_original: bool = True,
        ):
            seed = _normalize_optional_nonnegative_int(
                random_state,
                normalizer=mixstyle._normalize_nonnegative_int,
                name="random_state",
            )
            return original_config(
                augmentations_per_row=augmentations_per_row,
                alpha=alpha,
                random_state=seed,
                domain_pairing=domain_pairing,
                preserve_domain_mean=preserve_domain_mean,
                class_conditional=class_conditional,
                include_original=include_original,
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
            preserve_domain_mean: bool = False,
            class_conditional: bool = False,
            include_original: bool = True,
        ):
            seed = _normalize_optional_nonnegative_int(
                random_state,
                normalizer=mixstyle._normalize_nonnegative_int,
                name="random_state",
            )
            return original_augment(
                source_features,
                source_labels,
                source_domains,
                augmentations_per_row=augmentations_per_row,
                alpha=alpha,
                random_state=seed,
                domain_pairing=domain_pairing,
                preserve_domain_mean=preserve_domain_mean,
                class_conditional=class_conditional,
                include_original=include_original,
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
            include_original: bool = True,
            random_state: int | str | None = 13,
        ):
            seed = _normalize_optional_nonnegative_int(
                random_state,
                normalizer=source_mixstyle._normalize_nonnegative_int,
                name="random_state",
            )
            return original_config(
                mixes_per_row=mixes_per_row,
                alpha=alpha,
                style_strength=style_strength,
                synthetic_weight=synthetic_weight,
                include_original=include_original,
                random_state=seed,
            )

        setattr(source_mixstyle_config, _PATCH_MARKER, True)
        source_mixstyle.source_mixstyle_config = source_mixstyle_config

    original_augment = source_mixstyle.augment_source_domains_mixstyle
    if not getattr(original_augment, _PATCH_MARKER, False):

        def _coerce_config_with_seed(config: Any):
            cfg = source_mixstyle._coerce_config(config)
            seed = _normalize_optional_nonnegative_int(
                cfg.random_state,
                normalizer=source_mixstyle._normalize_nonnegative_int,
                name="random_state",
            )
            return replace(cfg, random_state=seed)

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
                config=_coerce_config_with_seed(config),
            )

        setattr(augment_source_domains_mixstyle, _PATCH_MARKER, True)
        source_mixstyle.augment_source_domains_mixstyle = augment_source_domains_mixstyle


def _patch_source_domain_mask() -> None:
    source_domain_mask_module = importlib.import_module("neureptrace.decoding.source_domain_mask")

    original_mask = source_domain_mask_module.source_domain_mask
    if getattr(original_mask, _PATCH_MARKER, False):
        return

    @wraps(original_mask)
    def source_domain_mask(
        source_domains,
        *,
        random_state: int | str | None = 13,
        **kwargs,
    ):
        seed = _normalize_optional_nonnegative_int(
            random_state,
            normalizer=source_domain_mask_module._nonnegative_int,
            name="random_state",
        )
        return original_mask(source_domains, random_state=seed, **kwargs)

    setattr(source_domain_mask, _PATCH_MARKER, True)
    source_domain_mask_module.source_domain_mask = source_domain_mask


def install() -> None:
    """Install random-state validation patches for config helpers."""

    global _INSTALLED
    if _INSTALLED:
        return
    _patch_feature_mixstyle()
    _patch_domain_mixstyle()
    _patch_source_domain_mask()
    _INSTALLED = True


__all__ = ["install"]

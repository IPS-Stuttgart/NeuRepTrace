"""Validate Source MixUp/SMOTE random-state values before RNG construction."""

from __future__ import annotations

import importlib
from dataclasses import replace
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_mixup_random_state_patch_installed"
_NONE_STRINGS = {"", "none", "null"}


def _random_state_error() -> ValueError:
    return ValueError("random_state must be a non-negative integer or none.")


def _normalize_optional_random_state(value: Any, *, normalizer: Any) -> int | None:
    """Normalize optional random-state values accepted by NumPy SeedSequence."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _NONE_STRINGS:
            return None
        value = text
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _random_state_error()
        value = value.item()
    if isinstance(value, (list, tuple, dict, set)):
        raise _random_state_error()
    try:
        return normalizer(value, name="random_state")
    except ValueError as exc:
        raise _random_state_error() from exc


def install() -> None:
    """Install early Source MixUp/SMOTE random-state validation."""

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
                synthetic_per_class=synthetic_per_class,
                alpha=alpha,
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
            cfg = source_mixup._coerce_config(config)
            seed = _normalize_optional_random_state(
                cfg.random_state,
                normalizer=source_mixup._normalize_nonnegative_int,
            )
            return replace(cfg, random_state=seed)

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

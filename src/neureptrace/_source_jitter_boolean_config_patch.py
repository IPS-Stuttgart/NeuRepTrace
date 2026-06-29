"""Normalize Source Feature Jitter booleans and related source augmentation config."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_jitter_boolean_config_patch_installed"
_AUGMENT_METADATA_PATCH_MARKER = "_neureptrace_source_jitter_disabled_metadata_patch_installed"
_MASKING_INT_PATCH_MARKER = "_neureptrace_source_feature_masking_int_config_patch_installed"
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}
_FALSE_STRINGS = {"0", "false", "f", "no", "n", "off"}
_NONE_STRINGS = {"", "none", "null"}


# Regression-note: this file is intentionally kept import-time patch compatible.
def _bool_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a boolean value.")


def _normalize_bool(value: Any, *, name: str) -> bool:
    """Return a strict bool for YAML/CLI-style jitter config values."""

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


def _integer_error(name: str) -> ValueError:
    return ValueError(f"{name} must be an integer or None.")


def _normalize_optional_integer(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _NONE_STRINGS:
            return None
        value = text
    elif isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _integer_error(name)
        return _normalize_optional_integer(value.item(), name=name)
    elif isinstance(value, (list, tuple, dict, set)):
        raise _integer_error(name)
    if isinstance(value, (bool, np.bool_)):
        raise _integer_error(name)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise _integer_error(name) from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise _integer_error(name)
    return int(parsed)


def _positive_optional_integer(value: Any, *, name: str) -> int | None:
    parsed = _normalize_optional_integer(value, name=name)
    if parsed is None:
        return None
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer or None.")
    return parsed


def _nonnegative_optional_integer(value: Any, *, name: str) -> int | None:
    parsed = _normalize_optional_integer(value, name=name)
    if parsed is None:
        return None
    if parsed < 0:
        raise ValueError(f"{name} must be a non-negative integer or None.")
    return parsed


def _install_source_jitter_patch() -> None:
    source_jitter = importlib.import_module("neureptrace.decoding.source_jitter")

    original_config = source_jitter.source_feature_jitter_config
    if not getattr(original_config, _PATCH_MARKER, False):

        @wraps(original_config)
        def source_feature_jitter_config(
            *,
            synthetic_per_class: int | str = 0,
            noise_scale: float | str = source_jitter.DEFAULT_NOISE_SCALE,
            scale_mode: str | None = "global",
            preserve_original: Any = True,
            random_state: Any = 13,
            epsilon: float | str = source_jitter.DEFAULT_EPSILON,
        ):
            return original_config(
                synthetic_per_class=synthetic_per_class,
                noise_scale=noise_scale,
                scale_mode=scale_mode,
                preserve_original=_normalize_bool(preserve_original, name="preserve_original"),
                random_state=_nonnegative_optional_integer(random_state, name="random_state"),
                epsilon=epsilon,
            )

        setattr(source_feature_jitter_config, _PATCH_MARKER, True)
        source_jitter.source_feature_jitter_config = source_feature_jitter_config

    original_augment = source_jitter.augment_source_with_feature_jitter
    if not getattr(original_augment, _AUGMENT_METADATA_PATCH_MARKER, False):

        @wraps(original_augment)
        def augment_source_with_feature_jitter(*args: Any, **kwargs: Any):
            result = original_augment(*args, **kwargs)
            output_rows = int(result.features.shape[0])
            metadata = result.metadata
            if metadata.get("source_feature_jitter_n_output_rows") == output_rows:
                return result
            metadata = dict(metadata)
            metadata["source_feature_jitter_n_output_rows"] = output_rows
            return source_jitter.SourceFeatureJitterResult(
                features=result.features,
                labels=result.labels,
                synthetic_mask=result.synthetic_mask,
                content_indices=result.content_indices,
                noise=result.noise,
                metadata=metadata,
            )

        setattr(augment_source_with_feature_jitter, _AUGMENT_METADATA_PATCH_MARKER, True)
        source_jitter.augment_source_with_feature_jitter = augment_source_with_feature_jitter


def _install_source_masking_patch() -> None:
    source_masking = importlib.import_module("neureptrace.decoding.source_masking")

    original_config = source_masking.source_feature_masking_config
    if getattr(original_config, _MASKING_INT_PATCH_MARKER, False):
        return

    @wraps(original_config)
    def source_feature_masking_config(
        *,
        synthetic_per_class: int | str = 0,
        mask_fraction: float | str = source_masking.DEFAULT_MASK_FRACTION,
        mask_mode: str | None = "feature",
        block_size: Any = None,
        fill_mode: str | None = "feature_mean",
        noise_std: float | str = 0.0,
        preserve_original: bool | int | str = True,
        random_state: Any = 13,
    ):
        return original_config(
            synthetic_per_class=synthetic_per_class,
            mask_fraction=mask_fraction,
            mask_mode=mask_mode,
            block_size=_positive_optional_integer(block_size, name="block_size"),
            fill_mode=fill_mode,
            noise_std=noise_std,
            preserve_original=preserve_original,
            random_state=_nonnegative_optional_integer(random_state, name="random_state"),
        )

    setattr(source_feature_masking_config, _MASKING_INT_PATCH_MARKER, True)
    source_masking.source_feature_masking_config = source_feature_masking_config


def install() -> None:
    """Install numeric/scalar boolean normalization and source masking config validation."""

    _install_source_jitter_patch()
    _install_source_masking_patch()


__all__ = ["install"]

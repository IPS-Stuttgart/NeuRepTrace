"""Normalize Source Feature Jitter inputs and related source augmentation config."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_jitter_boolean_config_patch_installed"
_AUGMENT_METADATA_PATCH_MARKER = "_neureptrace_source_jitter_disabled_metadata_patch_installed"
_JITTER_SINGLETON_AXIS_PATCH_MARKER = "_neureptrace_source_jitter_singleton_axis_vector_patch_installed"
_MASKING_INT_PATCH_MARKER = "_neureptrace_source_feature_masking_int_config_patch_installed"
_MASKING_AUGMENT_CONFIG_PATCH_MARKER = "_neureptrace_source_feature_masking_dataclass_config_patch_installed"
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


def _required_integer_error(name: str) -> ValueError:
    return ValueError(f"{name} must be an integer.")


def _float_error(name: str) -> ValueError:
    return ValueError(f"{name} must be finite.")


def _normalize_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise _required_integer_error(name)
    if isinstance(value, np.ndarray):
        raise _required_integer_error(name)
    if isinstance(value, (list, tuple, dict, set)):
        raise _required_integer_error(name)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise _required_integer_error(name) from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise _required_integer_error(name)
    return int(parsed)


def _nonnegative_integer(value: Any, *, name: str) -> int:
    parsed = _normalize_integer(value, name=name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _normalize_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise _float_error(name)
    if isinstance(value, np.ndarray):
        raise _float_error(name)
    if isinstance(value, (list, tuple, dict, set)):
        raise _float_error(name)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise _float_error(name) from exc
    if not np.isfinite(parsed):
        raise _float_error(name)
    return parsed


def _nonnegative_float(value: Any, *, name: str) -> float:
    parsed = _normalize_float(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return parsed


def _positive_float(value: Any, *, name: str) -> float:
    parsed = _normalize_float(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _unit_interval_float(value: Any, *, name: str) -> float:
    parsed = _normalize_float(value, name=name)
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed


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


def _normalize_jitter_dataclass_config(source_jitter: Any, config: Any) -> Any:
    if not isinstance(config, source_jitter.SourceFeatureJitterConfig):
        return config
    return source_jitter.source_feature_jitter_config(
        synthetic_per_class=config.synthetic_per_class,
        noise_scale=config.noise_scale,
        scale_mode=config.scale_mode,
        preserve_original=config.preserve_original,
        random_state=config.random_state,
        epsilon=config.epsilon,
    )


def _normalize_masking_dataclass_config(source_masking: Any, config: Any) -> Any:
    if not isinstance(config, source_masking.SourceFeatureMaskingConfig):
        return config
    return source_masking.source_feature_masking_config(
        synthetic_per_class=config.synthetic_per_class,
        mask_fraction=config.mask_fraction,
        mask_mode=config.mask_mode,
        block_size=config.block_size,
        fill_mode=config.fill_mode,
        noise_std=config.noise_std,
        preserve_original=config.preserve_original,
        random_state=config.random_state,
    )


def _materialize_features(value: Any) -> Any:
    """Recursively materialize generator-backed feature rows exactly once."""

    if isinstance(value, np.ndarray) or isinstance(value, (str, bytes)):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_features(item) for item in value]


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when every finite nonzero feature survives narrowing."""

    array = np.asarray(values)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = array.astype(np.float32, copy=False)
    if np.any(np.isfinite(array) & ~np.isfinite(compact)):
        return array
    if np.any((array != 0.0) & (compact == 0.0)):
        return array
    return compact


def _disabled_jitter_result(
    source_jitter: Any,
    source_features: Any,
    source_labels: Any,
    source_domains: Any,
    config: Any,
) -> Any:
    """Return the disabled no-op result without computing unused jitter scales."""

    features = source_jitter._feature_matrix(source_features, name="source_features")
    labels = source_jitter._label_vector(
        source_labels,
        expected_length=features.shape[0],
        name="source_labels",
    )
    domains = source_jitter._domain_vector(
        source_domains,
        expected_length=features.shape[0],
    )
    classes, _ = source_jitter.label_counts(labels)
    n_source_domains = int(source_jitter.label_counts(domains)[0].shape[0])
    metadata = source_jitter._metadata(
        config,
        n_source_rows=features.shape[0],
        n_synthetic_rows=0,
        n_classes=classes.shape[0],
        n_source_domains=n_source_domains,
        feature_dim=features.shape[1],
    )
    metadata["source_feature_jitter_n_output_rows"] = int(features.shape[0])
    return source_jitter.SourceFeatureJitterResult(
        features=_compact_float32(features),
        labels=labels.copy(),
        synthetic_mask=np.zeros(features.shape[0], dtype=bool),
        content_indices=np.empty(0, dtype=int),
        noise=np.empty((0, features.shape[1]), dtype=np.float32),
        metadata=metadata,
    )


def _singleton_axis_items(values: Any, *, expected_length: int) -> list[Any] | None:
    if expected_length == 1 or isinstance(values, (str, bytes)):
        return None
    try:
        array = np.asarray(values, dtype=object)
    except ValueError:
        return None
    if array.ndim < 2 or array.size != expected_length or 1 not in array.shape:
        return None
    return array.reshape(-1).tolist()


def _install_source_jitter_patch() -> None:
    source_jitter = importlib.import_module("neureptrace.decoding.source_jitter")

    original_config = source_jitter.source_feature_jitter_config
    if not getattr(original_config, _PATCH_MARKER, False):

        @wraps(original_config)
        def source_feature_jitter_config(
            *,
            synthetic_per_class: Any = 0,
            noise_scale: Any = source_jitter.DEFAULT_NOISE_SCALE,
            scale_mode: str | None = "global",
            preserve_original: Any = True,
            random_state: Any = 13,
            epsilon: Any = source_jitter.DEFAULT_EPSILON,
        ):
            return original_config(
                synthetic_per_class=_nonnegative_integer(synthetic_per_class, name="synthetic_per_class"),
                noise_scale=_nonnegative_float(noise_scale, name="noise_scale"),
                scale_mode=scale_mode,
                preserve_original=_normalize_bool(preserve_original, name="preserve_original"),
                random_state=_nonnegative_optional_integer(random_state, name="random_state"),
                epsilon=_positive_float(epsilon, name="epsilon"),
            )

        setattr(source_feature_jitter_config, _PATCH_MARKER, True)
        source_jitter.source_feature_jitter_config = source_feature_jitter_config

    original_augment = source_jitter.augment_source_with_feature_jitter
    if not getattr(original_augment, _AUGMENT_METADATA_PATCH_MARKER, False):

        @wraps(original_augment)
        def augment_source_with_feature_jitter(
            source_features: Any,
            source_labels: Any,
            *,
            source_domains: Any = None,
            config: Any = None,
        ):
            normalized_config = _normalize_jitter_dataclass_config(source_jitter, config)
            cfg = (
                source_jitter.source_feature_jitter_config()
                if normalized_config is None
                else source_jitter._coerce_config(normalized_config)
            )
            materialized_features = _materialize_features(source_features)
            if not cfg.enabled:
                return _disabled_jitter_result(
                    source_jitter,
                    materialized_features,
                    source_labels,
                    source_domains,
                    cfg,
                )
            result = original_augment(
                materialized_features,
                source_labels,
                source_domains=source_domains,
                config=cfg,
            )
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


def _install_source_jitter_singleton_axis_patch() -> None:
    source_jitter = importlib.import_module("neureptrace.decoding.source_jitter")

    original_atomic = source_jitter._atomic_value_vector
    if getattr(original_atomic, _JITTER_SINGLETON_AXIS_PATCH_MARKER, False):
        return

    @wraps(original_atomic)
    def _atomic_value_vector(values: Any, *, expected_length: int, name: str, require_hashable: bool):
        items = _singleton_axis_items(values, expected_length=expected_length)
        if items is None:
            return original_atomic(
                values,
                expected_length=expected_length,
                name=name,
                require_hashable=require_hashable,
            )
        if len(items) != expected_length:
            raise ValueError(f"{name} must contain one value per feature row: {len(items)} != {expected_length}.")
        if require_hashable:
            for value in items:
                try:
                    hash(value)
                except TypeError as exc:
                    raise ValueError(f"{name} must be hashable; got {value!r}.") from exc
        if source_jitter._contains_composite_value(items):
            return source_jitter._object_vector(items)
        return np.asarray(items).reshape(-1)

    setattr(_atomic_value_vector, _JITTER_SINGLETON_AXIS_PATCH_MARKER, True)
    source_jitter._atomic_value_vector = _atomic_value_vector


def _install_source_masking_patch() -> None:
    source_masking = importlib.import_module("neureptrace.decoding.source_masking")

    original_config = source_masking.source_feature_masking_config
    if not getattr(original_config, _MASKING_INT_PATCH_MARKER, False):

        @wraps(original_config)
        def source_feature_masking_config(
            *,
            synthetic_per_class: Any = 0,
            mask_fraction: Any = source_masking.DEFAULT_MASK_FRACTION,
            mask_mode: str | None = "feature",
            block_size: Any = None,
            fill_mode: str | None = "feature_mean",
            noise_std: Any = 0.0,
            preserve_original: bool | int | str = True,
            random_state: Any = 13,
        ):
            return original_config(
                synthetic_per_class=_nonnegative_integer(synthetic_per_class, name="synthetic_per_class"),
                mask_fraction=_unit_interval_float(mask_fraction, name="mask_fraction"),
                mask_mode=mask_mode,
                block_size=_positive_optional_integer(block_size, name="block_size"),
                fill_mode=fill_mode,
                noise_std=_nonnegative_float(noise_std, name="noise_std"),
                preserve_original=preserve_original,
                random_state=_nonnegative_optional_integer(random_state, name="random_state"),
            )

        setattr(source_feature_masking_config, _MASKING_INT_PATCH_MARKER, True)
        source_masking.source_feature_masking_config = source_feature_masking_config

    original_augment = source_masking.augment_source_with_feature_masking
    if not getattr(original_augment, _MASKING_AUGMENT_CONFIG_PATCH_MARKER, False):

        @wraps(original_augment)
        def augment_source_with_feature_masking(*args: Any, **kwargs: Any):
            if "config" in kwargs:
                kwargs = dict(kwargs)
                kwargs["config"] = _normalize_masking_dataclass_config(source_masking, kwargs["config"])
            return original_augment(*args, **kwargs)

        setattr(augment_source_with_feature_masking, _MASKING_AUGMENT_CONFIG_PATCH_MARKER, True)
        source_masking.augment_source_with_feature_masking = augment_source_with_feature_masking


def install() -> None:
    """Install Source Feature Jitter and source masking compatibility guards."""

    _install_source_jitter_patch()
    _install_source_jitter_singleton_axis_patch()
    _install_source_masking_patch()


__all__ = ["install"]

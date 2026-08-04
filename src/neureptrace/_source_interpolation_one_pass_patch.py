"""Materialize source inputs and preserve exact interpolation configuration values."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from functools import wraps
from pathlib import Path
from typing import Any

import numpy as np

_INTERPOLATION_PATCH_MARKER = "_neureptrace_source_interpolation_one_pass_patch_installed"
_MASKING_PATCH_MARKER = "_neureptrace_source_masking_feature_input_patch_installed"
_SMOTE_INTERPOLATION_PATCH_MARKER = "_neureptrace_source_smote_stable_interpolation_patch_installed"
_SMOTE_OUTPUT_PATCH_MARKER = "_neureptrace_source_smote_disabled_output_patch_installed"
_INTEGER_PRECISION_PATCH_MARKER = "_neureptrace_source_interpolation_integer_precision_patch_installed"


def _materialize_one_pass_iterable(value: Any) -> Any:
    """Expand generator-style containers once while preserving scalar labels."""

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.ndim == 0:
                return _materialize_one_pass_iterable(value.item())
            return _materialize_one_pass_iterable(value.tolist())
        return value
    if isinstance(value, (str, bytes, Mapping)):
        return value
    try:
        iterator = iter(value)
    except TypeError:
        return value
    return [_materialize_one_pass_iterable(item) for item in iterator]


def _contains_boolean_values(value: Any) -> bool:
    """Return whether a materialized feature container includes boolean values."""

    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return value.size > 0
        if value.dtype == object:
            return any(_contains_boolean_values(item) for item in value.flat)
        return False
    if isinstance(value, (str, bytes, Mapping)):
        return False
    try:
        iterator = iter(value)
    except TypeError:
        return False
    return any(_contains_boolean_values(item) for item in iterator)


def _compact_float32(values: np.ndarray) -> np.ndarray:
    """Use float32 only when finite nonzero values survive the conversion."""

    array = np.asarray(values)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        compact = array.astype(np.float32, copy=False)
    if not np.all(np.isfinite(compact)):
        return array
    if np.any((array != 0.0) & (compact == 0.0)):
        return array
    return compact


def _exact_integer(value: Any, *, name: str) -> int:
    """Normalize integer controls without routing exact integers through ``float``."""

    message = f"{name} must be an integer."
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    if isinstance(value, (list, tuple, dict, set, Path)):
        raise ValueError(message)
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (str, Decimal)):
        try:
            number = value if isinstance(value, Decimal) else Decimal(value.strip())
        except (InvalidOperation, ValueError) as exc:
            raise ValueError(message) from exc
        if not number.is_finite() or number != number.to_integral_value():
            raise ValueError(message)
        return int(number)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError(message)
    return int(numeric)


def _install_source_integer_precision_patch() -> None:
    """Share exact integer normalization across MixUp and SMOTE config paths."""

    source_mixup = importlib.import_module("neureptrace.decoding.source_mixup")
    source_smote = importlib.import_module("neureptrace.decoding.source_smote")
    random_state_patch = importlib.import_module("neureptrace._source_mixup_random_state_patch")

    if getattr(source_mixup._normalize_integer, _INTEGER_PRECISION_PATCH_MARKER, False):
        return

    setattr(_exact_integer, _INTEGER_PRECISION_PATCH_MARKER, True)
    source_mixup._normalize_integer = _exact_integer
    source_smote._integer = _exact_integer
    random_state_patch._normalize_integer = _exact_integer


def _install_source_interpolation_patch() -> None:
    module = importlib.import_module("neureptrace.decoding.source_interpolation")
    if getattr(module, _INTERPOLATION_PATCH_MARKER, False):
        return

    original_augment_source_with_interpolation = module.augment_source_with_interpolation
    original_feature_matrix = module._feature_matrix
    original_value_vector = module._value_vector

    def _materialized_values(values: Any) -> Any:
        return values if isinstance(values, (str, bytes)) else _materialize_one_pass_iterable(values)

    @wraps(module.interpolate_rows)
    def interpolate_rows(content: Any, partner: Any, lam: Any) -> np.ndarray:
        left = np.asarray(_materialize_one_pass_iterable(content), dtype=float).reshape(-1)
        right = np.asarray(_materialize_one_pass_iterable(partner), dtype=float).reshape(-1)
        if left.shape != right.shape:
            raise ValueError(f"content and partner must have the same shape: {left.shape} != {right.shape}.")
        if left.size == 0 or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError("content and partner must be finite non-empty vectors.")
        weight = module._unit_interval_float(lam, name="lam")
        return _compact_float32(weight * left + (1.0 - weight) * right)

    @wraps(original_augment_source_with_interpolation)
    def augment_source_with_interpolation(
        source_features: Any,
        source_labels: Any,
        *,
        source_domains: Any = None,
        config: Any = None,
    ) -> Any:
        cfg = module.source_interpolation_config() if config is None else module._coerce_config(config)
        features = module._feature_matrix(
            _materialize_one_pass_iterable(source_features),
            name="source_features",
        )
        labels = module._value_vector(
            _materialized_values(source_labels),
            expected_length=features.shape[0],
            name="source_labels",
        )
        domains = module._domain_vector(
            None if source_domains is None else _materialized_values(source_domains),
            expected_length=features.shape[0],
        )
        classes, _class_counts = module.label_counts(labels)
        domain_ids, _domain_counts = module.label_counts(domains)

        if not cfg.enabled:
            metadata = module._metadata(
                cfg,
                n_source_rows=features.shape[0],
                n_synthetic_rows=0,
                n_classes=classes.shape[0],
                n_source_domains=domain_ids.shape[0],
                feature_dim=features.shape[1],
            )
            compact_features = _compact_float32(features)
            if cfg.preserve_original:
                output_features = compact_features
                output_labels = labels.copy()
                synthetic_mask = np.zeros(features.shape[0], dtype=bool)
            else:
                output_features = np.empty((0, features.shape[1]), dtype=compact_features.dtype)
                output_labels = labels[:0].copy()
                synthetic_mask = np.zeros(0, dtype=bool)
            return module.SourceInterpolationResult(
                features=output_features,
                labels=output_labels,
                synthetic_mask=synthetic_mask,
                content_indices=np.empty(0, dtype=int),
                partner_indices=np.empty(0, dtype=int),
                lambdas=np.empty(0, dtype=float),
                metadata=metadata,
            )

        rng = np.random.default_rng(cfg.random_state)
        rows: list[np.ndarray] = []
        row_labels: list[Any] = []
        content_indices: list[int] = []
        partner_indices: list[int] = []
        lambdas: list[float] = []
        for class_label in classes.tolist():
            class_indices = np.flatnonzero(module.label_equal_mask(labels, class_label))
            if class_indices.size == 0:
                continue
            for _ in range(cfg.synthetic_per_class):
                content_index = int(rng.choice(class_indices))
                partners = module._partner_pool(
                    class_indices,
                    domains,
                    content_index=content_index,
                    pair_mode=cfg.pair_mode,
                )
                partner_index = int(rng.choice(partners))
                lam = float(rng.beta(cfg.alpha, cfg.alpha))
                rows.append(
                    interpolate_rows(
                        features[content_index],
                        features[partner_index],
                        lam,
                    )
                )
                row_labels.append(class_label)
                content_indices.append(content_index)
                partner_indices.append(partner_index)
                lambdas.append(lam)

        synthetic_features = (
            _compact_float32(np.vstack(rows))
            if rows
            else np.empty((0, features.shape[1]), dtype=np.float32)
        )
        synthetic_labels = module._object_array(row_labels)
        if cfg.preserve_original:
            output_features = _compact_float32(np.vstack([features, synthetic_features]))
            output_labels = np.concatenate([labels, synthetic_labels])
            synthetic_mask = np.concatenate(
                [
                    np.zeros(features.shape[0], dtype=bool),
                    np.ones(synthetic_features.shape[0], dtype=bool),
                ]
            )
        else:
            output_features = synthetic_features
            output_labels = synthetic_labels
            synthetic_mask = np.ones(synthetic_features.shape[0], dtype=bool)

        metadata = module._metadata(
            cfg,
            n_source_rows=features.shape[0],
            n_synthetic_rows=synthetic_features.shape[0],
            n_classes=classes.shape[0],
            n_source_domains=domain_ids.shape[0],
            feature_dim=features.shape[1],
        )
        return module.SourceInterpolationResult(
            features=output_features,
            labels=output_labels,
            synthetic_mask=synthetic_mask,
            content_indices=np.asarray(content_indices, dtype=int),
            partner_indices=np.asarray(partner_indices, dtype=int),
            lambdas=np.asarray(lambdas, dtype=float),
            metadata=metadata,
        )

    @wraps(original_feature_matrix)
    def _feature_matrix(values: Any, *, name: str) -> np.ndarray:
        return original_feature_matrix(_materialize_one_pass_iterable(values), name=name)

    @wraps(original_value_vector)
    def _value_vector(values: Any, *, expected_length: int, name: str) -> np.ndarray:
        return original_value_vector(_materialized_values(values), expected_length=expected_length, name=name)

    module.interpolate_rows = interpolate_rows
    module.augment_source_with_interpolation = augment_source_with_interpolation
    module._feature_matrix = _feature_matrix
    module._value_vector = _value_vector
    setattr(module, _INTERPOLATION_PATCH_MARKER, True)


def _install_source_masking_patch() -> None:
    module = importlib.import_module("neureptrace.decoding.source_masking")
    if getattr(module, _MASKING_PATCH_MARKER, False):
        return

    original_feature_matrix = module._feature_matrix

    @wraps(original_feature_matrix)
    def _feature_matrix(values: Any, *, name: str) -> np.ndarray:
        materialized = _materialize_one_pass_iterable(values)
        if _contains_boolean_values(materialized):
            raise ValueError(f"{name} must contain numeric, non-boolean values.")
        return original_feature_matrix(materialized, name=name)

    module._feature_matrix = _feature_matrix
    setattr(module, _MASKING_PATCH_MARKER, True)


def _install_source_smote_interpolation_patch() -> None:
    module = importlib.import_module("neureptrace.decoding.source_smote")
    original_interpolate_rows = module.interpolate_rows
    if not getattr(original_interpolate_rows, _SMOTE_INTERPOLATION_PATCH_MARKER, False):

        @wraps(original_interpolate_rows)
        def interpolate_rows(content_row: Any, partner_row: Any, lam: Any) -> np.ndarray:
            left = np.asarray(content_row, dtype=float).reshape(-1)
            right = np.asarray(partner_row, dtype=float).reshape(-1)
            if left.shape != right.shape or left.size == 0:
                raise ValueError("content_row and partner_row must be non-empty vectors with the same shape.")
            weight = module._unit_interval_float(lam, name="lam")

            same_sign = np.signbit(left) == np.signbit(right)
            row = np.empty_like(left)
            row[same_sign] = left[same_sign] + weight * (right[same_sign] - left[same_sign])
            row[~same_sign] = (1.0 - weight) * left[~same_sign] + weight * right[~same_sign]
            return row.astype(np.float32, copy=False)

        setattr(interpolate_rows, _SMOTE_INTERPOLATION_PATCH_MARKER, True)
        module.interpolate_rows = interpolate_rows

    original_augment_source_with_smote = module.augment_source_with_smote
    if not getattr(original_augment_source_with_smote, _SMOTE_OUTPUT_PATCH_MARKER, False):

        @wraps(original_augment_source_with_smote)
        def augment_source_with_smote(
            source_features: Any,
            source_labels: Any,
            *,
            source_domains: Any = None,
            config: Any = None,
        ) -> Any:
            materialized_features = _materialize_one_pass_iterable(source_features)
            result = original_augment_source_with_smote(
                materialized_features,
                source_labels,
                source_domains=source_domains,
                config=config,
            )
            if result.metadata["source_smote"]:
                return result

            features = _compact_float32(module._feature_matrix(materialized_features, name="source_features"))
            if result.metadata["source_smote_preserve_original"]:
                return module.SourceSmoteResult(
                    features=features,
                    labels=result.labels,
                    synthetic_mask=result.synthetic_mask,
                    content_indices=result.content_indices,
                    partner_indices=result.partner_indices,
                    lambdas=result.lambdas,
                    metadata=result.metadata,
                )
            return module.SourceSmoteResult(
                features=features[:0].copy(),
                labels=result.labels[:0].copy(),
                synthetic_mask=result.synthetic_mask[:0].copy(),
                content_indices=result.content_indices,
                partner_indices=result.partner_indices,
                lambdas=result.lambdas,
                metadata=result.metadata,
            )

        setattr(augment_source_with_smote, _SMOTE_OUTPUT_PATCH_MARKER, True)
        module.augment_source_with_smote = augment_source_with_smote


def install() -> None:
    """Patch source interpolation, masking, SMOTE numerics/output, and integer precision."""

    _install_source_integer_precision_patch()
    _install_source_interpolation_patch()
    _install_source_masking_patch()
    _install_source_smote_interpolation_patch()


__all__ = ["install"]
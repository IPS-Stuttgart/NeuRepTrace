"""Materialize one-pass source interpolation and masking inputs before NumPy conversion."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from functools import wraps
from typing import Any

import numpy as np

_INTERPOLATION_PATCH_MARKER = "_neureptrace_source_interpolation_one_pass_patch_installed"
_MASKING_PATCH_MARKER = "_neureptrace_source_masking_feature_input_patch_installed"
_SMOTE_INTERPOLATION_PATCH_MARKER = "_neureptrace_source_smote_stable_interpolation_patch_installed"
_SMOTE_OUTPUT_PATCH_MARKER = "_neureptrace_source_smote_disabled_output_patch_installed"


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


def _install_source_interpolation_patch() -> None:
    module = importlib.import_module("neureptrace.decoding.source_interpolation")
    if getattr(module, _INTERPOLATION_PATCH_MARKER, False):
        return

    original_augment_source_with_interpolation = module.augment_source_with_interpolation
    original_feature_matrix = module._feature_matrix
    original_value_vector = module._value_vector

    @wraps(original_augment_source_with_interpolation)
    def augment_source_with_interpolation(*args: Any, **kwargs: Any) -> Any:
        result = original_augment_source_with_interpolation(*args, **kwargs)
        if result.metadata["source_interpolation"] or result.metadata["source_interpolation_preserve_original"]:
            return result
        return module.SourceInterpolationResult(
            features=result.features[:0].copy(),
            labels=result.labels[:0].copy(),
            synthetic_mask=result.synthetic_mask[:0].copy(),
            content_indices=result.content_indices,
            partner_indices=result.partner_indices,
            lambdas=result.lambdas,
            metadata=result.metadata,
        )

    @wraps(original_feature_matrix)
    def _feature_matrix(values: Any, *, name: str) -> np.ndarray:
        return original_feature_matrix(_materialize_one_pass_iterable(values), name=name)

    @wraps(original_value_vector)
    def _value_vector(values: Any, *, expected_length: int, name: str) -> np.ndarray:
        materialized = values if isinstance(values, (str, bytes)) else _materialize_one_pass_iterable(values)
        return original_value_vector(materialized, expected_length=expected_length, name=name)

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
        def augment_source_with_smote(*args: Any, **kwargs: Any) -> Any:
            result = original_augment_source_with_smote(*args, **kwargs)
            if result.metadata["source_smote"] or result.metadata["source_smote_preserve_original"]:
                return result
            return module.SourceSmoteResult(
                features=result.features[:0].copy(),
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
    """Patch source interpolation, masking, and SMOTE numeric/output behavior."""

    _install_source_interpolation_patch()
    _install_source_masking_patch()
    _install_source_smote_interpolation_patch()


__all__ = ["install"]

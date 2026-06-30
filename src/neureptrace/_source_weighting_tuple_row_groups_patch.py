"""Preserve tuple-valued row groups in source-weight helpers."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_SAMPLE_WEIGHTS_PATCH_MARKER = "_neureptrace_source_weighting_tuple_row_groups_patch_installed"
_GROUP_LIST_PATCH_MARKER = "_neureptrace_source_weighting_tuple_group_list_patch_installed"
_SOURCE_BALANCE_CONFIG_PATCH_MARKER = "_neureptrace_source_balance_config_revalidation_patch_installed"


def _hashable_group_value(value: Any) -> Any:
    """Normalize array/list row identifiers into hashable mapping keys."""

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return tuple(value.tolist())
    if isinstance(value, list):
        return tuple(value)
    return value


def _object_vector(values: Sequence[Any]) -> np.ndarray:
    items = [_hashable_group_value(value) for value in values]
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _row_group_vector(row_groups: Sequence[Any] | np.ndarray) -> np.ndarray:
    """Return one row-group object per sample without flattening composite IDs."""

    if isinstance(row_groups, np.ndarray):
        array = np.asarray(row_groups, dtype=object)
        if array.ndim == 0:
            return _object_vector([array.item()])
        if array.ndim == 1:
            return _object_vector(array.tolist())
        if array.ndim == 2 and array.shape[1] == 1:
            return _object_vector(array.reshape(-1).tolist())
        if array.ndim == 2:
            return _object_vector(tuple(row.tolist()) for row in array)
        raise ValueError(f"row_groups must be one- or two-dimensional; got shape {array.shape}.")

    if isinstance(row_groups, (str, bytes)):
        return _object_vector([row_groups])
    try:
        items = list(row_groups)
    except TypeError:
        items = [row_groups]
    return _object_vector(items)


def _install_source_weighting_tuple_row_groups() -> None:
    source_weighting = importlib.import_module("neureptrace.decoding.source_weighting")

    original_sample_weights = source_weighting.sample_weights_from_group_weights
    if not getattr(original_sample_weights, _SAMPLE_WEIGHTS_PATCH_MARKER, False):

        @wraps(original_sample_weights)
        def sample_weights_from_group_weights(
            row_groups: Sequence[Any] | np.ndarray,
            group_weights: dict[Any, float] | None,
            *,
            default: float = 1.0,
            normalize: bool = True,
        ) -> np.ndarray | None:
            if group_weights is None:
                return None
            rows = _row_group_vector(row_groups)
            lookup = {group: source_weighting._nonnegative_float(weight, name="source_group_weight") for group, weight in group_weights.items()}
            default_value = source_weighting._nonnegative_float(default, name="source_group_weight_default")
            weights = np.asarray([lookup.get(group, default_value) for group in rows.tolist()], dtype=np.float64)
            if normalize:
                weights = source_weighting._mean_one(weights)
            return weights

        setattr(sample_weights_from_group_weights, _SAMPLE_WEIGHTS_PATCH_MARKER, True)
        source_weighting.sample_weights_from_group_weights = sample_weights_from_group_weights

    original_group_list = source_weighting._group_list
    if not getattr(original_group_list, _GROUP_LIST_PATCH_MARKER, False):

        @wraps(original_group_list)
        def _group_list(groups: Sequence[Any] | np.ndarray | None, *, source_scores=None, source_features=None):
            if groups is not None:
                return list(dict.fromkeys(_row_group_vector(groups).tolist()))
            return original_group_list(groups, source_scores=source_scores, source_features=source_features)

        setattr(_group_list, _GROUP_LIST_PATCH_MARKER, True)
        source_weighting._group_list = _group_list

    source_weighting._row_group_vector = _row_group_vector


def _install_source_balance_config_revalidation() -> None:
    source_balance = importlib.import_module("neureptrace.decoding.source_balance")
    original_coerce_config = source_balance._coerce_config
    if getattr(original_coerce_config, _SOURCE_BALANCE_CONFIG_PATCH_MARKER, False):
        return

    @wraps(original_coerce_config)
    def _coerce_config(config):
        if isinstance(config, source_balance.SourceBalanceConfig):
            return source_balance.source_balance_config(
                strategy=config.strategy,
                target=config.target,
                normalize_weights=config.normalize_weights,
                random_state=config.random_state,
            )
        return original_coerce_config(config)

    setattr(_coerce_config, _SOURCE_BALANCE_CONFIG_PATCH_MARKER, True)
    source_balance._coerce_config = _coerce_config


def install() -> None:
    """Patch source weighting/balancing helpers to keep composite and config inputs safe."""

    _install_source_weighting_tuple_row_groups()
    _install_source_balance_config_revalidation()


install()

__all__ = ["install"]

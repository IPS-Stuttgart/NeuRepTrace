"""Reject boolean reaction-time observations."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping

import numpy as np

_TO_FLOAT_MARKER = "_neureptrace_reaction_time_boolean_to_float_patch_installed"
_VALUES_MARKER = "_neureptrace_reaction_time_boolean_values_patch_installed"


def _is_boolean_scalar(value: object) -> bool:
    return isinstance(value, (bool, np.bool_))


def _materialize_nested_iterables(value: object) -> object:
    """Materialize one-pass inputs before checking and forwarding them."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_nested_iterables(value.tolist())
    if isinstance(value, (str, bytes, Mapping)):
        return value
    if isinstance(value, Iterable):
        return [_materialize_nested_iterables(item) for item in value]
    return value


def _contains_boolean(value: object) -> bool:
    if _is_boolean_scalar(value):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return value.size > 0
        if value.dtype == object:
            return any(_contains_boolean(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes, Mapping)):
        return False
    if isinstance(value, np.generic):
        return _is_boolean_scalar(value.item())
    if isinstance(value, Iterable):
        return any(_contains_boolean(item) for item in value)
    return False


def install() -> None:
    """Prevent boolean flags from becoming zero/one-second observations."""

    module = importlib.import_module("neureptrace.behavior.reaction_time")

    original_to_float = module._to_float
    if not getattr(original_to_float, _TO_FLOAT_MARKER, False):

        def to_float(value: object) -> float:
            if _is_boolean_scalar(value):
                return np.nan
            return original_to_float(value)

        setattr(to_float, _TO_FLOAT_MARKER, True)
        module._to_float = to_float

    original_reaction_time_rows_from_values = module.reaction_time_rows_from_values
    if not getattr(original_reaction_time_rows_from_values, _VALUES_MARKER, False):

        def reaction_time_rows_from_values(
            values,
            *,
            participant=None,
            dataset="main",
            reaction_time_scale=1.0,
        ):
            materialized = _materialize_nested_iterables(values)
            if _contains_boolean(materialized):
                raise ValueError("reaction-time values must be numeric observations, not boolean flags.")
            return original_reaction_time_rows_from_values(
                materialized,
                participant=participant,
                dataset=dataset,
                reaction_time_scale=reaction_time_scale,
            )

        setattr(reaction_time_rows_from_values, _VALUES_MARKER, True)
        module.reaction_time_rows_from_values = reaction_time_rows_from_values


__all__ = ["install"]

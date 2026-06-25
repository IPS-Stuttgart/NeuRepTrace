"""Runtime validation patch for dataset-spec numeric fields.

YAML and JSON dataset specs often decode configuration values as plain Python
scalars.  Since booleans are integer-like in Python, direct ``int(...)`` or
``float(...)`` coercion can silently turn malformed numeric fields such as
``index_base: true`` into valid-looking numbers.  Keep the public parser strict
by rejecting booleans, non-integral integers, and non-finite floats at the
specification boundary.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_dataset_spec_numeric_validation_patch_installed"


def _is_boolean(value: Any) -> bool:
    return isinstance(value, (bool, np.bool_))


def _finite_float(value: Any, *, name: str) -> float:
    if _is_boolean(value):
        raise ValueError(f"{name} must be a finite numeric value.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric value.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite numeric value.")
    return float(number)


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if _is_boolean(value):
        raise ValueError(f"{name} must be an integer.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer.")
    integer = int(number)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return integer


def _optional_int(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    if value is None:
        return None
    return _integer(value, name=key)


def _optional_float(mapping: Mapping[str, Any], key: str) -> float | None:
    value = mapping.get(key)
    if value is None:
        return None
    return _finite_float(value, name=key)


def _two_float_tuple(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numeric values.")
    lower = _finite_float(value[0], name=f"{name}[0]")
    if name == "preprocessing_defaults.frequency_range_hz":
        try:
            upper = float(value[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}[1] must be a finite numeric value.") from exc
        if np.isposinf(upper):
            return lower, upper
    return lower, _finite_float(value[1], name=f"{name}[1]")


def install() -> None:
    """Install strict numeric validation for dataset-spec scalar fields."""

    from neureptrace import dataset_spec

    if getattr(dataset_spec, _PATCH_MARKER, False):
        return

    def _parse_labels(mapping: Mapping[str, Any]) -> Any:
        chance_classes = None
        if mapping.get("chance_classes") is not None:
            chance_classes = _integer(mapping["chance_classes"], name="labels.chance_classes", minimum=1)
        index_base = 0
        if mapping.get("index_base") is not None:
            index_base = _integer(mapping["index_base"], name="labels.index_base", minimum=0)
        return dataset_spec.LabelSpec(
            column=dataset_spec._optional_str(mapping, "column"),
            chance_classes=chance_classes,
            index_base=index_base,
            subtract_one_when_no_null_class=bool(mapping.get("subtract_one_when_no_null_class", False)),
        )

    dataset_spec._optional_int = _optional_int
    dataset_spec._optional_float = _optional_float
    dataset_spec._two_float_tuple = _two_float_tuple
    dataset_spec._parse_labels = _parse_labels
    setattr(dataset_spec, _PATCH_MARKER, True)


__all__ = ["install"]

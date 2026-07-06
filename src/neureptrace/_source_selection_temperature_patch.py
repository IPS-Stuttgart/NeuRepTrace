"""Validation guards for source-domain and probability temperature controls."""

from __future__ import annotations

import numpy as np

from . import _source_numpy_string_alias_config_patch

_PATCH_ATTR = "_neureptrace_rejects_boolean_source_selection_temperature"
_SOURCE_TEMPERATURE_CLASS_INDEX_PATCH_ATTR = "_neureptrace_source_temperature_sparse_integer_indices_patch"


def _integer_index_classes_with_sparse_support(labels: np.ndarray, *, n_classes: int) -> np.ndarray | None:
    """Infer full probability-column classes from valid integer index labels.

    Source-temperature fitting can legitimately receive a source fold that lacks
    one or more classes.  When labels are integer column indices and the
    probability matrix width is known, the absent columns are still identifiable
    as ``range(n_classes)`` and should not force callers to pass ``classes=``.
    """

    indices: list[int] = []
    for label in labels.tolist():
        if isinstance(label, (bool, np.bool_, str, bytes)):
            return None
        if isinstance(label, np.generic):
            label = label.item()
        if isinstance(label, np.ndarray):
            if label.ndim != 0:
                return None
            label = label.item()
        try:
            value = float(label)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value) or value != np.floor(value):
            return None
        indices.append(int(value))

    if not indices:
        return None
    if any(index < 0 or index >= n_classes for index in indices):
        raise ValueError(f"integer source_labels must be valid class indices from 0 to {n_classes - 1} when classes is omitted.")

    values = np.empty(n_classes, dtype=object)
    for index in range(n_classes):
        values[index] = index
    return values


def _install_source_temperature_class_index_patch() -> None:
    from neureptrace.decoding import source_temperature

    original = source_temperature._integer_index_classes
    if getattr(original, _SOURCE_TEMPERATURE_CLASS_INDEX_PATCH_ATTR, False):
        return

    def _integer_index_classes(labels: np.ndarray, *, n_classes: int) -> np.ndarray | None:
        return _integer_index_classes_with_sparse_support(labels, n_classes=n_classes)

    setattr(_integer_index_classes, _SOURCE_TEMPERATURE_CLASS_INDEX_PATCH_ATTR, True)
    _integer_index_classes.__wrapped__ = original
    source_temperature._integer_index_classes = _integer_index_classes


def install() -> None:
    """Reject boolean softmax temperatures and install source temperature guards."""

    _source_numpy_string_alias_config_patch.install()

    from neureptrace.decoding import source_selection

    original = source_selection._resolve_temperature
    if not getattr(original, _PATCH_ATTR, False):

        def _resolve_temperature_checked(distance_gaps: np.ndarray, temperature: float | str) -> float:
            if isinstance(temperature, (bool, np.bool_)):
                raise ValueError("softmax_temperature must be a positive finite value or 'auto', not a boolean.")
            return original(distance_gaps, temperature)

        setattr(_resolve_temperature_checked, _PATCH_ATTR, True)
        _resolve_temperature_checked.__wrapped__ = original
        source_selection._resolve_temperature = _resolve_temperature_checked

    _install_source_temperature_class_index_patch()

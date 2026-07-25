"""Validation guards for source-domain and probability temperature controls."""

from __future__ import annotations

from functools import wraps

import numpy as np

from . import _source_numpy_string_alias_config_patch

_PATCH_ATTR = "_neureptrace_rejects_boolean_source_selection_temperature"
_SOURCE_TEMPERATURE_CLASS_INDEX_PATCH_ATTR = "_neureptrace_source_temperature_sparse_integer_indices_patch"
_SOURCE_TEMPERATURE_NORMALIZATION_PATCH_ATTR = "_neureptrace_source_temperature_stable_probability_normalization_patch"
_SOURCE_TEMPERATURE_COMPLEX_SCALAR_PATCH_ATTR = "_neureptrace_source_temperature_complex_scalar_patch"


def _contains_complex(value: object) -> bool:
    """Return whether a materialized probability input contains complex values."""

    if isinstance(value, (complex, np.complexfloating)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.complexfloating):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_complex(item) for item in value.reshape(-1).tolist())
        return False
    if isinstance(value, (str, bytes)):
        return False
    try:
        iterator = iter(value)
    except TypeError:
        return False
    return any(_contains_complex(item) for item in iterator)


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


def _install_source_temperature_complex_scalar_patch() -> None:
    """Reject complex scalar temperature controls before float conversion."""

    from neureptrace.decoding import source_temperature

    original = source_temperature._positive_float
    if getattr(original, _SOURCE_TEMPERATURE_COMPLEX_SCALAR_PATCH_ATTR, False):
        return

    @wraps(original)
    def _positive_float(value, *, name: str) -> float:
        if isinstance(value, (complex, np.complexfloating)):
            raise ValueError(f"{name} must be positive and finite.")
        return original(value, name=name)

    setattr(_positive_float, _SOURCE_TEMPERATURE_COMPLEX_SCALAR_PATCH_ATTR, True)
    source_temperature._positive_float = _positive_float


def _install_source_temperature_probability_normalization_patch() -> None:
    """Normalize finite source-temperature score rows without overflowing."""

    from neureptrace.decoding import source_temperature

    original = source_temperature._probability_matrix
    if getattr(original, _SOURCE_TEMPERATURE_NORMALIZATION_PATCH_ATTR, False):
        return

    def _probability_matrix(values, *, name: str, epsilon: float) -> np.ndarray:
        eps = source_temperature._probability_epsilon(epsilon)
        materialized = source_temperature._materialize_one_pass_iterable(values)
        if source_temperature._contains_boolean(materialized):
            raise ValueError(f"{name} must contain numeric probabilities, not boolean values.")
        if _contains_complex(materialized):
            raise ValueError(f"{name} must contain real-valued probabilities, not complex values.")
        try:
            matrix = np.asarray(materialized, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain numeric probabilities.") from exc
        if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
            raise ValueError(f"{name} must be a non-empty two-dimensional matrix with at least two columns.")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
            raise ValueError(f"{name} must contain finite non-negative values.")

        row_maxima = np.max(matrix, axis=1, keepdims=True)
        if np.any(row_maxima <= 0.0):
            raise ValueError(f"{name} rows must have positive probability mass.")
        scaled = matrix / row_maxima
        scaled_sums = np.sum(scaled, axis=1, keepdims=True)
        if np.any(row_maxima <= eps / scaled_sums):
            raise ValueError(f"{name} rows must have positive probability mass.")
        return scaled / scaled_sums

    setattr(_probability_matrix, _SOURCE_TEMPERATURE_NORMALIZATION_PATCH_ATTR, True)
    _probability_matrix.__wrapped__ = original
    source_temperature._probability_matrix = _probability_matrix


def install() -> None:
    """Reject invalid temperature controls and install source-temperature guards."""

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
    _install_source_temperature_complex_scalar_patch()
    _install_source_temperature_probability_normalization_patch()

"""Strict source-only probability temperature scaling.

This module fits a single scalar temperature from source validation probability
rows and source labels, then applies the fixed temperature to held-out probability
rows.  It is a Protocol-1 calibration helper: held-out probabilities are
transformed but never used to fit the temperature.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SOURCE_TEMPERATURE_PROTOCOL = "strict_source_only_probability_temperature"
SOURCE_TEMPERATURE_CATEGORY = "1_strict_source_only"
DEFAULT_TEMPERATURE_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SourceTemperatureConfig:
    """Configuration for source-only temperature selection."""

    temperatures: tuple[float, ...] = DEFAULT_TEMPERATURE_GRID
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceTemperatureResult:
    """Temperature-scaled probabilities and provenance metadata."""

    probabilities: np.ndarray
    temperature: float
    source_losses: Mapping[float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_source_temperature_scaling(
    *,
    source_probabilities: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    test_probabilities: Sequence[Sequence[float]] | np.ndarray,
    classes: Sequence[Any] | np.ndarray | None = None,
    config: SourceTemperatureConfig | Mapping[str, Any] | None = None,
) -> SourceTemperatureResult:
    """Fit temperature on source rows and apply it to held-out probabilities."""

    cfg = source_temperature_config() if config is None else _coerce_config(config)
    source = _probability_matrix(source_probabilities, name="source_probabilities", epsilon=cfg.epsilon)
    test = _probability_matrix(test_probabilities, name="test_probabilities", epsilon=cfg.epsilon)
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_probabilities and test_probabilities must have the same class width: {source.shape[1]} != {test.shape[1]}.")
    labels = _object_vector(source_labels, expected_length=source.shape[0], name="source_labels")
    class_values = _classes(labels, classes, n_classes=source.shape[1])
    class_to_index = _class_to_index(class_values)
    label_index = np.asarray([class_to_index[_value_key(label)] for label in labels.tolist()], dtype=int)
    losses = {temperature: negative_log_likelihood(apply_temperature(source, temperature=temperature, epsilon=cfg.epsilon), label_index, epsilon=cfg.epsilon) for temperature in cfg.temperatures}
    best_temperature = min(losses, key=lambda value: (losses[value], value))
    scaled = apply_temperature(test, temperature=best_temperature, epsilon=cfg.epsilon)
    return SourceTemperatureResult(
        probabilities=scaled.astype(np.float32, copy=False),
        temperature=float(best_temperature),
        source_losses={float(key): float(value) for key, value in losses.items()},
        metadata=_metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], n_classes=source.shape[1], temperature=best_temperature),
    )


def apply_temperature(probabilities: Sequence[Sequence[float]] | np.ndarray, *, temperature: float | str, epsilon: float = DEFAULT_EPSILON) -> np.ndarray:
    """Apply scalar temperature to probability rows."""

    matrix = _probability_matrix(probabilities, name="probabilities", epsilon=epsilon)
    temp = _positive_float(temperature, name="temperature")
    logits = np.log(np.maximum(matrix, float(epsilon))) / temp
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits, -80.0, 80.0))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def negative_log_likelihood(probabilities: Sequence[Sequence[float]] | np.ndarray, labels: Sequence[int] | np.ndarray, *, epsilon: float = DEFAULT_EPSILON) -> float:
    """Return mean negative log likelihood for integer label indices."""

    matrix = _probability_matrix(probabilities, name="probabilities", epsilon=epsilon)
    indices = np.asarray(labels, dtype=int).reshape(-1)
    if indices.shape[0] != matrix.shape[0]:
        raise ValueError("labels must contain one value per probability row.")
    if np.any(indices < 0) or np.any(indices >= matrix.shape[1]):
        raise ValueError("labels contain class indices outside the probability width.")
    return float(-np.mean(np.log(np.maximum(matrix[np.arange(matrix.shape[0]), indices], float(epsilon)))))


def source_temperature_config(
    *,
    temperatures: Sequence[float] | str = DEFAULT_TEMPERATURE_GRID,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceTemperatureConfig:
    """Normalize source-temperature options."""

    if isinstance(temperatures, str):
        raw_values = tuple(part.strip() for part in temperatures.replace(";", ",").split(",") if part.strip())
    else:
        try:
            raw_values = tuple(temperatures)
        except TypeError as exc:
            raise ValueError("temperatures must contain positive finite values.") from exc
    values = tuple(_positive_float(value, name="temperatures") for value in raw_values)
    if not values:
        raise ValueError("temperatures must contain at least one value.")
    return SourceTemperatureConfig(temperatures=values, epsilon=_positive_float(epsilon, name="epsilon"))


def _coerce_config(config: SourceTemperatureConfig | Mapping[str, Any]) -> SourceTemperatureConfig:
    if isinstance(config, SourceTemperatureConfig):
        return source_temperature_config(temperatures=config.temperatures, epsilon=config.epsilon)
    return source_temperature_config(**dict(config))


def _classes(labels: np.ndarray, classes: Sequence[Any] | np.ndarray | None, *, n_classes: int) -> np.ndarray:
    if classes is None:
        values = _unique_values(labels)
    else:
        values = _object_vector(classes, expected_length=n_classes, name="classes")
    if values.shape[0] != n_classes:
        raise ValueError(f"classes must contain one value per probability column: {values.shape[0]} != {n_classes}.")
    class_keys = [_value_key(value) for value in values.tolist()]
    if len(set(class_keys)) != values.shape[0]:
        raise ValueError("classes must be unique.")
    class_key_set = set(class_keys)
    missing = sorted({label for label in labels.tolist() if _value_key(label) not in class_key_set}, key=repr)
    if missing:
        raise ValueError(f"source_labels contain labels absent from classes: {missing}.")
    return values


def _class_to_index(classes: np.ndarray) -> dict[Any, int]:
    return {_value_key(label): index for index, label in enumerate(classes.tolist())}


def _object_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            items = [array.item()]
        elif array.ndim == 1:
            if array.shape[0] == expected_length:
                items = array.tolist()
            elif expected_length == 1:
                items = [tuple(array.tolist())]
            else:
                items = array.tolist()
        else:
            rows = array.reshape(array.shape[0], -1)
            if rows.shape[1] == 1:
                items = rows[:, 0].tolist()
            else:
                items = [tuple(row.tolist()) for row in rows]
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError:
            items = [values]
        if len(items) != expected_length and expected_length == 1 and isinstance(values, tuple):
            items = [values]
    if len(items) != expected_length:
        raise ValueError(f"{name} must contain one value per expected row/column: {len(items)} != {expected_length}.")
    vector = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        vector[index] = item
    return vector


def _unique_values(values: np.ndarray) -> np.ndarray:
    unique: list[Any] = []
    seen: set[Any] = set()
    for value in values.tolist():
        key = _value_key(value)
        if key not in seen:
            seen.add(key)
            unique.append(value)
    vector = np.empty(len(unique), dtype=object)
    for index, value in enumerate(unique):
        vector[index] = value
    return vector


def _value_key(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    try:
        hash(value)
    except TypeError:
        if isinstance(value, np.ndarray):
            return tuple(_value_key(item) for item in value.tolist())
        if isinstance(value, (list, tuple)):
            return tuple(_value_key(item) for item in value)
        if isinstance(value, dict):
            return tuple(sorted((_value_key(key), _value_key(item)) for key, item in value.items()))
        return repr(value)
    return value


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix with at least two columns.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain finite non-negative values.")
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= float(epsilon)):
        raise ValueError(f"{name} rows must have positive probability mass.")
    return matrix / row_sums


def _metadata(cfg: SourceTemperatureConfig, *, n_source_rows: int, n_test_rows: int, n_classes: int, temperature: float) -> dict[str, Any]:
    return {
        "source_temperature_scaling": True,
        "source_temperature_protocol": SOURCE_TEMPERATURE_PROTOCOL,
        "source_temperature_protocol_category": SOURCE_TEMPERATURE_CATEGORY,
        "source_temperature_uses_source_probabilities": True,
        "source_temperature_uses_source_labels": True,
        "source_temperature_uses_test_probabilities_for_fitting": False,
        "source_temperature_uses_test_labels": False,
        "source_temperature_valid_for_strict_source_only": True,
        "source_temperature_valid_for_benchmark": True,
        "source_temperature_n_source_rows": int(n_source_rows),
        "source_temperature_n_test_rows": int(n_test_rows),
        "source_temperature_n_classes": int(n_classes),
        "source_temperature_selected": float(temperature),
        "source_temperature_grid": "|".join(f"{value:.12g}" for value in cfg.temperatures),
        "source_temperature_epsilon": float(cfg.epsilon),
    }


def _reject_array_scalar(value: Any, *, name: str) -> None:
    if isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a scalar value, not an array.")


def _positive_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be numeric, not boolean.")
    _reject_array_scalar(value, name=name)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed

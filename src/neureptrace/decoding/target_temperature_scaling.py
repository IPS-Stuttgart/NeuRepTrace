"""Target-calibrated probability temperature scaling.

This module fits a single scalar temperature on a disjoint labeled target
calibration subset, then applies that temperature to held-out probability rows.
It is a Category-3 helper because calibration labels from the target subject are
used for fitting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neureptrace._object_label_utils import values_equal

TARGET_TEMPERATURE_PROTOCOL = "supervised_target_temperature_scaling"
TARGET_TEMPERATURE_CATEGORY = "3_supervised_calibrated_target_alignment"
DEFAULT_TEMPERATURE_GRID = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class TargetTemperatureConfig:
    """Configuration for target-calibrated temperature scaling."""

    temperature_grid: tuple[float, ...] = DEFAULT_TEMPERATURE_GRID
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class TargetTemperatureResult:
    """Temperature-scaled probabilities and calibration provenance."""

    probabilities: np.ndarray
    calibration_probabilities: np.ndarray
    temperature: float
    classes: np.ndarray
    calibration_nll_by_temperature: Mapping[float, float]
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_target_temperature_scaling(
    *,
    calibration_probabilities: Sequence[Sequence[float]] | np.ndarray,
    calibration_labels: Sequence[Any] | np.ndarray,
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    classes: Sequence[Any] | np.ndarray,
    config: TargetTemperatureConfig | Mapping[str, Any] | None = None,
) -> TargetTemperatureResult:
    """Fit target temperature on labeled calibration rows and score held-out rows.

    The calibration rows and scored rows should be disjoint in benchmark use.  The
    scored labels are intentionally not accepted by this API.
    """

    cfg = target_temperature_config() if config is None else _coerce_config(config)
    class_values = _class_vector(classes)
    cal_prob = _probability_matrix(
        calibration_probabilities,
        expected_columns=class_values.shape[0],
        name="calibration_probabilities",
        epsilon=cfg.epsilon,
    )
    score_prob = _probability_matrix(probabilities, expected_columns=class_values.shape[0], name="probabilities", epsilon=cfg.epsilon)
    cal_labels = _label_vector(calibration_labels, expected_length=cal_prob.shape[0], name="calibration_labels")
    unknown = _unknown_labels(cal_labels, class_values)
    if unknown:
        raise ValueError(f"calibration_labels contain labels absent from classes: {unknown!r}.")

    nll_by_temperature = {
        temperature: negative_log_likelihood(
            apply_temperature_to_probabilities(cal_prob, temperature=temperature, epsilon=cfg.epsilon),
            cal_labels,
            classes=class_values,
            epsilon=cfg.epsilon,
        )
        for temperature in cfg.temperature_grid
    }
    best_temperature = min(nll_by_temperature, key=lambda value: (nll_by_temperature[value], value))
    scaled_calibration = apply_temperature_to_probabilities(cal_prob, temperature=best_temperature, epsilon=cfg.epsilon)
    scaled_probabilities = apply_temperature_to_probabilities(score_prob, temperature=best_temperature, epsilon=cfg.epsilon)
    metadata = {
        "target_temperature_scaling": True,
        "target_temperature_protocol": TARGET_TEMPERATURE_PROTOCOL,
        "target_temperature_protocol_category": TARGET_TEMPERATURE_CATEGORY,
        "target_temperature_uses_target_calibration_probabilities": True,
        "target_temperature_uses_target_calibration_labels": True,
        "target_temperature_uses_scored_target_labels": False,
        "target_temperature_valid_for_strict_source_only": False,
        "target_temperature_valid_for_unlabeled_target_adaptation": False,
        "target_temperature_valid_for_supervised_calibration": True,
        "target_temperature_valid_for_benchmark": False,
        "target_temperature_n_calibration_rows": int(cal_prob.shape[0]),
        "target_temperature_n_scored_rows": int(score_prob.shape[0]),
        "target_temperature_n_classes": int(class_values.shape[0]),
        "target_temperature_selected": float(best_temperature),
        "target_temperature_grid": "|".join(f"{value:.12g}" for value in cfg.temperature_grid),
        "target_temperature_calibration_nll": "|".join(f"{temp:.12g}:{nll:.12g}" for temp, nll in nll_by_temperature.items()),
        "target_temperature_epsilon": float(cfg.epsilon),
    }
    return TargetTemperatureResult(
        probabilities=scaled_probabilities.astype(np.float32, copy=False),
        calibration_probabilities=scaled_calibration.astype(np.float32, copy=False),
        temperature=float(best_temperature),
        classes=class_values,
        calibration_nll_by_temperature=nll_by_temperature,
        metadata=metadata,
    )


def target_temperature_config(
    *,
    temperature_grid: Sequence[float] | str = DEFAULT_TEMPERATURE_GRID,
    epsilon: float | str = DEFAULT_EPSILON,
) -> TargetTemperatureConfig:
    """Normalize temperature-scaling options."""

    return TargetTemperatureConfig(
        temperature_grid=_temperature_grid(temperature_grid),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def apply_temperature_to_probabilities(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    temperature: float | str,
    epsilon: float = DEFAULT_EPSILON,
) -> np.ndarray:
    """Apply temperature scaling to probability rows."""

    eps = _positive_float(epsilon, name="epsilon")
    prob = _probability_matrix(probabilities, expected_columns=None, name="probabilities", epsilon=eps)
    temp = _positive_float(temperature, name="temperature")
    logits = np.log(np.maximum(prob, eps)) / temp
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits, -50.0, 50.0))
    return _normalize_probability_rows(exp_logits, epsilon=eps)


def negative_log_likelihood(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence[Any] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray,
    epsilon: float = DEFAULT_EPSILON,
) -> float:
    """Return mean negative log likelihood for labels in class order."""

    eps = _positive_float(epsilon, name="epsilon")
    class_values = _class_vector(classes)
    prob = _probability_matrix(probabilities, expected_columns=class_values.shape[0], name="probabilities", epsilon=eps)
    label_vector = _label_vector(labels, expected_length=prob.shape[0], name="labels")
    columns = np.asarray([_class_index(label, class_values, name="labels") for label in label_vector], dtype=int)
    return float(-np.mean(np.log(np.maximum(prob[np.arange(prob.shape[0]), columns], eps))))


def _coerce_config(config: TargetTemperatureConfig | Mapping[str, Any]) -> TargetTemperatureConfig:
    if isinstance(config, TargetTemperatureConfig):
        return config
    return target_temperature_config(**dict(config))


def _temperature_grid(values: Sequence[float] | str) -> tuple[float, ...]:
    if isinstance(values, str):
        parts = [part.strip() for part in values.replace(";", ",").split(",") if part.strip()]
        parsed = tuple(_positive_float(part, name="temperature_grid") for part in parts)
    else:
        parsed = tuple(_positive_float(value, name="temperature_grid") for value in values)
    if not parsed:
        raise ValueError("temperature_grid must contain at least one value.")
    return tuple(sorted(set(parsed)))


def _class_vector(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    vector = _object_label_vector(values)
    if vector.shape[0] < 2:
        raise ValueError("classes must contain at least two values.")
    for index, label in enumerate(vector):
        if any(values_equal(label, prior_label) for prior_label in vector[:index]):
            raise ValueError("classes must be unique.")
    return vector


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = _object_label_vector(values)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per row: {vector.shape[0]} != {expected_length}.")
    return vector


def _object_label_vector(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            return _object_vector([values.item()])
        rows = values.reshape(values.shape[0], -1)
        if rows.shape[1] == 1:
            return rows[:, 0].astype(object, copy=False).reshape(-1)
        return _object_vector([tuple(row.tolist()) for row in rows])
    if isinstance(values, (str, bytes)):
        return _object_vector([values])
    try:
        items = list(values)
    except TypeError:
        return _object_vector([values])
    if any(isinstance(item, (list, tuple, np.ndarray)) for item in items):
        return _object_vector([_label_object(item) for item in items])
    return np.asarray(items, dtype=object).reshape(-1)


def _label_object(value: object) -> object:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            return array.item()
        flattened = array.reshape(-1)
        if flattened.shape[0] == 1:
            return flattened[0]
        return tuple(flattened.tolist())
    if isinstance(value, list):
        return tuple(value)
    return value


def _object_vector(values: Sequence[object]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def _unknown_labels(labels: np.ndarray, classes: np.ndarray) -> list[object]:
    unknown: list[object] = []
    for label in labels:
        if _has_label(classes, label):
            continue
        if not _has_label(unknown, label):
            unknown.append(label)
    return unknown


def _has_label(values: Sequence[object] | np.ndarray, label: object) -> bool:
    return any(values_equal(value, label) for value in values)


def _class_index(label: object, classes: np.ndarray, *, name: str) -> int:
    for index, class_label in enumerate(classes):
        if values_equal(label, class_label):
            return index
    raise ValueError(f"{name} contain a value absent from classes: {label!r}.")


def _probability_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, expected_columns: int | None, name: str, epsilon: float) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix with at least two columns.")
    if expected_columns is not None and matrix.shape[1] != expected_columns:
        raise ValueError(f"{name} must have one column per class: {matrix.shape[1]} != {expected_columns}.")
    return _normalize_probability_rows(matrix, epsilon=epsilon)


def _normalize_probability_rows(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    eps = _positive_float(epsilon, name="epsilon")
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    matrix = np.maximum(matrix, eps)
    return matrix / np.sum(matrix, axis=1, keepdims=True)


def _positive_float(value: float | str, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed

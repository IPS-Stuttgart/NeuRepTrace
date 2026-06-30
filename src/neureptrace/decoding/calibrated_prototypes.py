"""Few-shot calibrated prototype decoder.

This module implements a compact Category-3 decoder.  Source class prototypes are
estimated from source labels, calibration prototypes are estimated from a small
labeled calibration subset of the held-out domain, and the final prototype is a
shrinkage blend of the two.  Evaluation rows are only scored.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CALIBRATED_PROTOTYPE_PROTOCOL = "supervised_calibrated_prototype_blend"
CALIBRATED_PROTOTYPE_CATEGORY = "3_supervised_calibrated_target_alignment"
DEFAULT_PRIOR_STRENGTH = 4.0
DEFAULT_TEMPERATURE = 1.0
DEFAULT_EPSILON = 1e-8


@dataclass(frozen=True, slots=True)
class CalibratedPrototypeConfig:
    """Configuration for calibrated prototype blending."""

    prior_strength: float = DEFAULT_PRIOR_STRENGTH
    fixed_calibration_weight: float | None = None
    temperature: float = DEFAULT_TEMPERATURE
    diagonal_scale: bool = True
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class CalibratedPrototypeResult:
    """Calibrated prototype predictions and provenance."""

    probabilities: np.ndarray
    predictions: np.ndarray
    classes: np.ndarray
    prototypes: np.ndarray
    source_prototypes: np.ndarray
    calibration_prototypes: np.ndarray
    calibration_counts: np.ndarray
    blend_weights: np.ndarray
    feature_scale: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def fit_calibrated_prototype_decoder(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    calibration_features: Sequence[Sequence[float]] | np.ndarray,
    calibration_labels: Sequence[Any] | np.ndarray,
    eval_features: Sequence[Sequence[float]] | np.ndarray,
    config: CalibratedPrototypeConfig | Mapping[str, Any] | None = None,
) -> CalibratedPrototypeResult:
    """Fit source/calibration prototype blend and score evaluation rows."""

    cfg = calibrated_prototype_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    calibration = _feature_matrix(calibration_features, name="calibration_features")
    evaluation = _feature_matrix(eval_features, name="eval_features")
    if source.shape[1] != calibration.shape[1] or source.shape[1] != evaluation.shape[1]:
        raise ValueError("source, calibration, and evaluation features must have the same feature width.")
    source_y = _label_vector(source_labels, expected_length=source.shape[0], name="source_labels")
    calibration_y = _label_vector(calibration_labels, expected_length=calibration.shape[0], name="calibration_labels")
    classes = np.asarray(tuple(dict.fromkeys(source_y.tolist())), dtype=object)
    if classes.shape[0] < 2:
        raise ValueError("At least two source classes are required.")
    unknown = sorted({label for label in calibration_y.tolist() if label not in set(classes.tolist())}, key=repr)
    if unknown:
        raise ValueError(f"calibration_labels contain labels absent from source classes: {unknown}.")

    source_proto, source_counts = _class_means(source, source_y, classes=classes)
    calibration_proto, calibration_counts = _class_means(calibration, calibration_y, classes=classes, fill_values=source_proto)
    weights = _blend_weights(calibration_counts, prior_strength=cfg.prior_strength, fixed_weight=cfg.fixed_calibration_weight)
    prototypes = (1.0 - weights[:, None]) * source_proto + weights[:, None] * calibration_proto
    scale = _feature_scale(np.vstack([source, calibration]), enabled=cfg.diagonal_scale, epsilon=cfg.epsilon)
    distances = _scaled_squared_distances(evaluation, prototypes, scale=scale)
    probabilities = _softmax(-distances / cfg.temperature)
    predictions = classes[np.argmax(probabilities, axis=1)]
    return CalibratedPrototypeResult(
        probabilities=probabilities.astype(np.float32, copy=False),
        predictions=predictions,
        classes=classes,
        prototypes=prototypes.astype(np.float32, copy=False),
        source_prototypes=source_proto.astype(np.float32, copy=False),
        calibration_prototypes=calibration_proto.astype(np.float32, copy=False),
        calibration_counts=calibration_counts.astype(int, copy=False),
        blend_weights=weights.astype(np.float32, copy=False),
        feature_scale=scale.astype(np.float32, copy=False),
        metadata=_metadata(cfg, classes=classes, source_counts=source_counts, calibration_counts=calibration_counts, blend_weights=weights, n_source=source.shape[0], n_calibration=calibration.shape[0], n_eval=evaluation.shape[0], feature_dim=source.shape[1]),
    )


def calibrated_prototype_config(
    *,
    prior_strength: float | str = DEFAULT_PRIOR_STRENGTH,
    fixed_calibration_weight: float | str | None = None,
    temperature: float | str = DEFAULT_TEMPERATURE,
    diagonal_scale: bool = True,
    epsilon: float | str = DEFAULT_EPSILON,
) -> CalibratedPrototypeConfig:
    """Normalize calibrated prototype options."""

    fixed = None if fixed_calibration_weight in {None, "", "none", "None"} else _unit_interval_float(fixed_calibration_weight, name="fixed_calibration_weight")
    return CalibratedPrototypeConfig(
        prior_strength=_positive_float(prior_strength, name="prior_strength"),
        fixed_calibration_weight=fixed,
        temperature=_positive_float(temperature, name="temperature"),
        diagonal_scale=bool(diagonal_scale),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def _coerce_config(config: CalibratedPrototypeConfig | Mapping[str, Any]) -> CalibratedPrototypeConfig:
    if isinstance(config, CalibratedPrototypeConfig):
        return config
    return calibrated_prototype_config(**dict(config))


def _class_means(features: np.ndarray, labels: np.ndarray, *, classes: np.ndarray, fill_values: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    means = np.zeros((classes.shape[0], features.shape[1]), dtype=float)
    counts = np.zeros(classes.shape[0], dtype=int)
    for index, label in enumerate(classes.tolist()):
        mask = labels == label
        counts[index] = int(np.count_nonzero(mask))
        if counts[index] > 0:
            means[index] = np.mean(features[mask], axis=0)
        elif fill_values is not None:
            means[index] = fill_values[index]
        else:
            raise ValueError(f"No rows available for class {label!r}.")
    return means, counts


def _blend_weights(counts: np.ndarray, *, prior_strength: float, fixed_weight: float | None) -> np.ndarray:
    if fixed_weight is not None:
        return np.full(counts.shape[0], float(fixed_weight), dtype=float)
    return counts.astype(float) / (counts.astype(float) + float(prior_strength))


def _feature_scale(features: np.ndarray, *, enabled: bool, epsilon: float) -> np.ndarray:
    if not enabled:
        return np.ones(features.shape[1], dtype=float)
    scale = np.std(features - np.mean(features, axis=0), axis=0, ddof=1 if features.shape[0] > 1 else 0)
    return np.maximum(scale, float(epsilon))


def _scaled_squared_distances(features: np.ndarray, prototypes: np.ndarray, *, scale: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] / scale - prototypes[None, :, :] / scale
    return np.sum(diff * diff, axis=2)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _metadata(cfg: CalibratedPrototypeConfig, *, classes: np.ndarray, source_counts: np.ndarray, calibration_counts: np.ndarray, blend_weights: np.ndarray, n_source: int, n_calibration: int, n_eval: int, feature_dim: int) -> dict[str, Any]:
    return {
        "calibrated_prototype": True,
        "calibrated_prototype_protocol": CALIBRATED_PROTOTYPE_PROTOCOL,
        "calibrated_prototype_protocol_category": CALIBRATED_PROTOTYPE_CATEGORY,
        "calibrated_prototype_uses_source_features": True,
        "calibrated_prototype_uses_source_labels": True,
        "calibrated_prototype_uses_calibration_features": True,
        "calibrated_prototype_uses_calibration_labels": True,
        "calibrated_prototype_uses_eval_labels": False,
        "calibrated_prototype_valid_for_strict_source_only": False,
        "calibrated_prototype_valid_for_unlabeled_target_adaptation": False,
        "calibrated_prototype_valid_for_benchmark": False,
        "calibrated_prototype_n_source_rows": int(n_source),
        "calibrated_prototype_n_calibration_rows": int(n_calibration),
        "calibrated_prototype_n_eval_rows": int(n_eval),
        "calibrated_prototype_feature_dim": int(feature_dim),
        "calibrated_prototype_n_classes": int(classes.shape[0]),
        "calibrated_prototype_prior_strength": float(cfg.prior_strength),
        "calibrated_prototype_fixed_calibration_weight": "" if cfg.fixed_calibration_weight is None else float(cfg.fixed_calibration_weight),
        "calibrated_prototype_temperature": float(cfg.temperature),
        "calibrated_prototype_diagonal_scale": bool(cfg.diagonal_scale),
        "calibrated_prototype_class_counts": "|".join(f"{label}:{int(src)}:{int(cal)}:{float(weight):.6g}" for label, src, cal, weight in zip(classes.tolist(), source_counts, calibration_counts, blend_weights, strict=True)),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must contain one value per row: {vector.shape[0]} != {expected_length}.")
    return vector


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _unit_interval_float(value: float | str, *, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0 or parsed > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return parsed

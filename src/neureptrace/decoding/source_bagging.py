"""Strict source-only bootstrap bagging decoder.

This module provides a dependency-light Protocol-1 ensemble baseline for
cross-subject feature decoding.  Each base estimator is trained on a bootstrap or
subsample of labeled source rows, optionally with a source-only feature subset.
Held-out rows are scored by averaging estimator probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCE_BAGGING_PROTOCOL = "strict_source_only_bootstrap_bagging_decoder"
SOURCE_BAGGING_CATEGORY = "1_strict_source_only"
DEFAULT_N_ESTIMATORS = 25
DEFAULT_SAMPLE_FRACTION = 1.0
DEFAULT_FEATURE_FRACTION = 1.0
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SourceBaggingConfig:
    """Configuration for the source-only bagging decoder."""

    n_estimators: int = DEFAULT_N_ESTIMATORS
    sample_fraction: float = DEFAULT_SAMPLE_FRACTION
    feature_fraction: float = DEFAULT_FEATURE_FRACTION
    bootstrap_rows: bool = True
    bootstrap_features: bool = False
    class_balanced: bool = True
    random_state: int | None = 13
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class SourceBaggingResult:
    """Bagged predictions and provenance metadata."""

    probabilities: np.ndarray
    predictions: np.ndarray
    classes: np.ndarray
    estimators: tuple[BaseEstimator, ...]
    row_indices: tuple[np.ndarray, ...]
    feature_indices: tuple[np.ndarray, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_estimators(self) -> int:
        """Number of fitted base estimators."""

        return len(self.estimators)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_source_bagging_decoder(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    estimator: BaseEstimator | None = None,
    config: SourceBaggingConfig | Mapping[str, Any] | None = None,
) -> SourceBaggingResult:
    """Fit a strict source-only bagged classifier and score held-out rows.

    Parameters
    ----------
    source_features, source_labels:
        Labeled source rows used to fit all bootstrap estimators.
    test_features:
        Rows to score.  These rows are never used for fitting.
    estimator:
        Optional sklearn-compatible estimator.  If omitted, a standardized
        balanced logistic-regression classifier is used.
    config:
        Bagging options.  Mappings are normalized through
        :func:`source_bagging_config`.
    """

    cfg = source_bagging_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    test = _feature_matrix(test_features, name="test_features")
    if source.shape[1] != test.shape[1]:
        raise ValueError(f"source_features and test_features must have the same feature width: {source.shape[1]} != {test.shape[1]}.")
    labels = _label_vector(source_labels, expected_length=source.shape[0], name="source_labels")
    classes = _unique_labels(labels)
    if classes.shape[0] < 2:
        raise ValueError("At least two source classes are required.")
    class_to_code = {class_label: index for index, class_label in enumerate(classes.tolist())}
    label_codes = np.asarray([class_to_code[class_label] for class_label in labels.tolist()], dtype=int)
    encoded_classes = np.arange(classes.shape[0], dtype=int)
    template = _default_estimator(cfg) if estimator is None else estimator
    rng = np.random.default_rng(cfg.random_state)

    probabilities: list[np.ndarray] = []
    estimators: list[BaseEstimator] = []
    row_indices: list[np.ndarray] = []
    feature_indices: list[np.ndarray] = []
    for _ in range(cfg.n_estimators):
        rows = _sample_rows(labels, classes=classes, cfg=cfg, rng=rng)
        feats = _sample_features(source.shape[1], cfg=cfg, rng=rng)
        model = clone(template)
        model.fit(source[rows][:, feats], label_codes[rows])
        probabilities.append(_aligned_probabilities(model, test[:, feats], classes=encoded_classes, epsilon=cfg.epsilon))
        estimators.append(model)
        row_indices.append(rows.astype(int, copy=False))
        feature_indices.append(feats.astype(int, copy=False))

    mean_probabilities = _normalize_probability_rows(np.mean(probabilities, axis=0), epsilon=cfg.epsilon)
    predictions = classes[np.argmax(mean_probabilities, axis=1)]
    metadata = _metadata(cfg, n_source_rows=source.shape[0], n_test_rows=test.shape[0], feature_dim=source.shape[1], classes=classes)
    return SourceBaggingResult(
        probabilities=mean_probabilities.astype(np.float32, copy=False),
        predictions=predictions,
        classes=classes,
        estimators=tuple(estimators),
        row_indices=tuple(row_indices),
        feature_indices=tuple(feature_indices),
        metadata=metadata,
    )


def source_bagging_config(
    *,
    n_estimators: int | str = DEFAULT_N_ESTIMATORS,
    sample_fraction: float | str = DEFAULT_SAMPLE_FRACTION,
    feature_fraction: float | str = DEFAULT_FEATURE_FRACTION,
    bootstrap_rows: bool | str = True,
    bootstrap_features: bool | str = False,
    class_balanced: bool | str = True,
    random_state: int | str | None = 13,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceBaggingConfig:
    """Normalize public source-bagging options."""

    return SourceBaggingConfig(
        n_estimators=_positive_int(n_estimators, name="n_estimators"),
        sample_fraction=_positive_float(sample_fraction, name="sample_fraction"),
        feature_fraction=_positive_float(feature_fraction, name="feature_fraction"),
        bootstrap_rows=_boolean(bootstrap_rows, name="bootstrap_rows"),
        bootstrap_features=_boolean(bootstrap_features, name="bootstrap_features"),
        class_balanced=_boolean(class_balanced, name="class_balanced"),
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def _coerce_config(config: SourceBaggingConfig | Mapping[str, Any]) -> SourceBaggingConfig:
    if isinstance(config, SourceBaggingConfig):
        return config
    return source_bagging_config(**dict(config))


def _default_estimator(cfg: SourceBaggingConfig) -> BaseEstimator:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=cfg.random_state))


def _sample_rows(labels: np.ndarray, *, classes: np.ndarray, cfg: SourceBaggingConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.class_balanced:
        rows = []
        for class_label in classes.tolist():
            class_rows = np.flatnonzero(_label_equal_mask(labels, class_label))
            n_take = max(1, int(round(class_rows.size * cfg.sample_fraction)))
            rows.append(rng.choice(class_rows, size=n_take, replace=cfg.bootstrap_rows))
        return rng.permutation(np.concatenate(rows)).astype(int, copy=False)
    n_take = max(classes.shape[0], int(round(labels.shape[0] * cfg.sample_fraction)))
    rows = rng.choice(labels.shape[0], size=n_take, replace=cfg.bootstrap_rows).astype(int, copy=False)
    if _unique_label_count(labels[rows]) < classes.shape[0]:
        forced = [int(rng.choice(np.flatnonzero(_label_equal_mask(labels, class_label)))) for class_label in classes.tolist()]
        rows[: len(forced)] = forced
        rng.shuffle(rows)
    return rows


def _sample_features(n_features: int, *, cfg: SourceBaggingConfig, rng: np.random.Generator) -> np.ndarray:
    n_take = max(1, min(n_features, int(round(n_features * cfg.feature_fraction))))
    if n_take == n_features and not cfg.bootstrap_features:
        return np.arange(n_features, dtype=int)
    sampled = rng.choice(n_features, size=n_take, replace=cfg.bootstrap_features)
    return np.sort(sampled).astype(int, copy=False)


def _aligned_probabilities(model: BaseEstimator, features: np.ndarray, *, classes: np.ndarray, epsilon: float) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(features), dtype=float)
        model_classes = np.asarray(getattr(model, "classes_", classes), dtype=object)
        aligned = np.full((features.shape[0], classes.shape[0]), float(epsilon), dtype=float)
        class_to_column = {class_label: index for index, class_label in enumerate(classes.tolist())}
        for column, class_label in enumerate(model_classes.tolist()):
            if class_label in class_to_column:
                aligned[:, class_to_column[class_label]] = raw[:, column]
        return _normalize_probability_rows(aligned, epsilon=epsilon)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return _normalize_probability_rows(np.exp(np.clip(scores - np.max(scores, axis=1, keepdims=True), -50.0, 50.0)), epsilon=epsilon)
    predictions = np.asarray(model.predict(features), dtype=object)
    output = np.full((features.shape[0], classes.shape[0]), float(epsilon), dtype=float)
    class_to_column = {class_label: index for index, class_label in enumerate(classes.tolist())}
    for row, label in enumerate(predictions.tolist()):
        if label in class_to_column:
            output[row, class_to_column[label]] = 1.0
    return _normalize_probability_rows(output, epsilon=epsilon)


def _normalize_probability_rows(probabilities: np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite, non-negative, and two-dimensional.")
    matrix = np.maximum(matrix, float(epsilon))
    return matrix / np.sum(matrix, axis=1, keepdims=True)


def _metadata(cfg: SourceBaggingConfig, *, n_source_rows: int, n_test_rows: int, feature_dim: int, classes: np.ndarray) -> dict[str, Any]:
    return {
        "source_bagging_decoder": True,
        "source_bagging_protocol": SOURCE_BAGGING_PROTOCOL,
        "source_bagging_protocol_category": SOURCE_BAGGING_CATEGORY,
        "source_bagging_uses_source_features": True,
        "source_bagging_uses_source_labels": True,
        "source_bagging_uses_test_features_for_fitting": False,
        "source_bagging_uses_test_labels": False,
        "source_bagging_valid_for_strict_source_only": True,
        "source_bagging_valid_for_benchmark": True,
        "source_bagging_n_source_rows": int(n_source_rows),
        "source_bagging_n_test_rows": int(n_test_rows),
        "source_bagging_feature_dim": int(feature_dim),
        "source_bagging_n_classes": int(classes.shape[0]),
        "source_bagging_n_estimators": int(cfg.n_estimators),
        "source_bagging_sample_fraction": float(cfg.sample_fraction),
        "source_bagging_feature_fraction": float(cfg.feature_fraction),
        "source_bagging_bootstrap_rows": bool(cfg.bootstrap_rows),
        "source_bagging_bootstrap_features": bool(cfg.bootstrap_features),
        "source_bagging_class_balanced": bool(cfg.class_balanced),
        "source_bagging_random_state": "" if cfg.random_state is None else int(cfg.random_state),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    items = _label_values(values)
    if len(items) != expected_length:
        raise ValueError(f"{name} must contain one value per row: {len(items)} != {expected_length}.")
    return _object_array(items)


def _label_values(values: Sequence[Any] | np.ndarray) -> list[Any]:
    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            return [_hashable_label(array.item())]
        if array.ndim == 1:
            return [_hashable_label(value) for value in array.tolist()]
        if array.ndim == 2 and array.shape[1] == 1:
            return [_hashable_label(value) for value in array.reshape(-1).tolist()]
        rows = array.reshape(array.shape[0], -1)
        return [tuple(_hashable_label(value) for value in row.tolist()) for row in rows]
    if isinstance(values, (str, bytes)):
        return [values]
    try:
        items = list(values)
    except TypeError:
        items = [values]
    return [_hashable_label(value) for value in items]


def _hashable_label(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=object)
        if array.ndim == 0:
            return _hashable_label(array.item())
        return tuple(_hashable_label(item) for item in array.tolist())
    if isinstance(value, list):
        return tuple(_hashable_label(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_hashable_label(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _hashable_label(item)) for key, item in value.items())
    return value


def _object_array(values: Sequence[Any]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def _unique_labels(labels: np.ndarray) -> np.ndarray:
    unique: list[Any] = []
    for label in labels.tolist():
        if not any(_labels_equal(label, existing) for existing in unique):
            unique.append(label)
    return _object_array(unique)


def _unique_label_count(labels: np.ndarray) -> int:
    return int(_unique_labels(labels).shape[0])


def _label_equal_mask(labels: np.ndarray, target: Any) -> np.ndarray:
    return np.asarray([_labels_equal(label, target) for label in labels.tolist()], dtype=bool)


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(equal, (bool, np.bool_)):
        return bool(equal)
    try:
        return bool(np.all(equal))
    except (TypeError, ValueError):
        return False


def _boolean(value: bool | str, *, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value.")


def _positive_int(value: int | str, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_int(value: int | str, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return integer


def _integer(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite.") from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed

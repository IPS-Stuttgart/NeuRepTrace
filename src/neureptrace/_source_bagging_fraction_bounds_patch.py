"""Reject malformed Source Bagging numeric options, feature matrices, and estimator outputs."""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_bagging_fraction_bounds_patch_installed"


def _fraction_error(name: str) -> ValueError:
    return ValueError(f"{name} must be in (0, 1].")


def _positive_int_error(name: str) -> ValueError:
    return ValueError(f"{name} must be a positive integer.")


def _positive_float_error(name: str) -> ValueError:
    return ValueError(f"{name} must be positive and finite.")


def _feature_matrix_error(name: str) -> ValueError:
    return ValueError(f"{name} must contain numeric feature values, not boolean flags.")


def _estimator_row_count_error(source: str) -> ValueError:
    return ValueError(f"source bagging estimator {source} must contain one row per test feature row.")


def _estimator_probability_column_error() -> ValueError:
    return ValueError("source bagging estimator classes_ length must match probability columns.")


def _estimator_score_column_error() -> ValueError:
    return ValueError("source bagging estimator decision_function output must contain one column per class.")


def _positive_int(value: Any, *, name: str) -> int:
    """Return a positive integer while rejecting booleans and non-scalar arrays."""

    if isinstance(value, (bool, np.bool_)):
        raise _positive_int_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0 or np.issubdtype(value.dtype, np.bool_):
            raise _positive_int_error(name)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise _positive_int_error(name)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (bool, np.bool_, list, tuple, dict, set)):
        raise _positive_int_error(name)
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise _positive_int_error(name) from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1.0:
        raise _positive_int_error(name)
    return int(parsed)


def _bounded_fraction(value: Any, *, name: str) -> float:
    """Return a finite fraction in ``(0, 1]`` while rejecting booleans."""

    if isinstance(value, (bool, np.bool_)):
        raise _fraction_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _fraction_error(name)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise _fraction_error(name)
    if isinstance(value, (list, tuple, dict, set)):
        raise _fraction_error(name)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise _fraction_error(name) from exc
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed > 1.0:
        raise _fraction_error(name)
    return parsed


def _positive_float(value: Any, *, name: str) -> float:
    """Return a positive finite scalar while rejecting boolean and array controls."""

    if isinstance(value, (bool, np.bool_)):
        raise _positive_float_error(name)
    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise _positive_float_error(name)
        value = value.item()
        if isinstance(value, (bool, np.bool_)):
            raise _positive_float_error(name)
    if isinstance(value, (list, tuple, dict, set)):
        raise _positive_float_error(name)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise _positive_float_error(name) from exc
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise _positive_float_error(name)
    return parsed


def _materialize_one_pass_iterables(value: object) -> object:
    """Materialize nested one-pass iterables before validation and NumPy coercion."""

    if isinstance(value, np.ndarray):
        if value.dtype != object:
            return value
        return _materialize_one_pass_iterables(value.tolist())
    if isinstance(value, (str, bytes)):
        return value
    if not isinstance(value, Iterable):
        return value
    return [_materialize_one_pass_iterables(item) for item in value]


def _contains_boolean_value(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_contains_boolean_value(item) for item in value.ravel(order="C"))
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.generic):
        return isinstance(value.item(), (bool, np.bool_))
    if isinstance(value, Iterable):
        return any(_contains_boolean_value(item) for item in value)
    return False


def _validate_config(cfg: Any) -> Any:
    _positive_int(cfg.n_estimators, name="n_estimators")
    _bounded_fraction(cfg.sample_fraction, name="sample_fraction")
    _bounded_fraction(cfg.feature_fraction, name="feature_fraction")
    _positive_float(cfg.epsilon, name="epsilon")
    return cfg


def _install_without_replacement_row_sampling(source_bagging: Any) -> None:
    original_sample_rows = source_bagging._sample_rows
    if getattr(original_sample_rows, _PATCH_MARKER, False):
        return

    @wraps(original_sample_rows)
    def _sample_rows(labels: np.ndarray, *, classes: np.ndarray, cfg: Any, rng: np.random.Generator) -> np.ndarray:
        if cfg.class_balanced or cfg.bootstrap_rows:
            return original_sample_rows(labels, classes=classes, cfg=cfg, rng=rng)

        n_take = max(classes.shape[0], int(round(labels.shape[0] * cfg.sample_fraction)))
        rows = rng.choice(labels.shape[0], size=n_take, replace=False).astype(int, copy=False)
        selected_labels = labels[rows]
        missing_classes = [
            class_label
            for class_label in classes.tolist()
            if not np.any(source_bagging._label_equal_mask(selected_labels, class_label))
        ]
        for class_label in missing_classes:
            selected_labels = labels[rows]
            donor_positions = [
                position
                for position, selected_label in enumerate(selected_labels.tolist())
                if np.count_nonzero(source_bagging._label_equal_mask(selected_labels, selected_label)) > 1
            ]
            if not donor_positions:
                raise RuntimeError("Cannot repair source-bagging class coverage without duplicating rows.")

            class_rows = np.flatnonzero(source_bagging._label_equal_mask(labels, class_label))
            available_rows = class_rows[~np.isin(class_rows, rows)]
            if available_rows.size == 0:
                raise RuntimeError("Cannot repair source-bagging class coverage without duplicating rows.")

            donor_position = int(rng.choice(np.asarray(donor_positions, dtype=int)))
            rows[donor_position] = int(rng.choice(available_rows))

        rng.shuffle(rows)
        return rows.astype(int, copy=False)

    setattr(_sample_rows, _PATCH_MARKER, True)
    source_bagging._sample_rows = _sample_rows


def _install_aligned_probability_validation(source_bagging: Any) -> None:
    original_aligned_probabilities = source_bagging._aligned_probabilities
    if getattr(original_aligned_probabilities, _PATCH_MARKER, False):
        return

    @wraps(original_aligned_probabilities)
    def _aligned_probabilities(model: Any, features: np.ndarray, *, classes: np.ndarray, epsilon: float) -> np.ndarray:
        n_rows = int(features.shape[0])
        if hasattr(model, "predict_proba"):
            raw = np.asarray(model.predict_proba(features), dtype=float)
            if raw.ndim != 2 or raw.shape[0] != n_rows:
                raise _estimator_row_count_error("probabilities")
            model_classes = np.asarray(getattr(model, "classes_", classes), dtype=object).reshape(-1)
            if model_classes.shape[0] != raw.shape[1]:
                raise _estimator_probability_column_error()
            aligned = np.full((n_rows, classes.shape[0]), float(epsilon), dtype=float)
            for column, class_label in enumerate(model_classes.tolist()):
                class_index = source_bagging._label_index_or_none(class_label, classes)
                if class_index is not None:
                    aligned[:, class_index] = raw[:, column]
            return source_bagging._normalize_probability_rows(aligned, epsilon=epsilon)
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(features), dtype=float)
            if scores.ndim == 1:
                if scores.shape[0] != n_rows:
                    raise _estimator_row_count_error("decision_function output")
                scores = np.column_stack([-scores, scores])
            elif scores.ndim == 2 and scores.shape[1] == 1 and classes.shape[0] == 2:
                if scores.shape[0] != n_rows:
                    raise _estimator_row_count_error("decision_function output")
                scores = np.column_stack([-scores[:, 0], scores[:, 0]])
            elif scores.ndim != 2 or scores.shape[0] != n_rows:
                raise _estimator_row_count_error("decision_function output")
            if scores.shape[1] != classes.shape[0]:
                raise _estimator_score_column_error()
            logits = np.exp(np.clip(scores - np.max(scores, axis=1, keepdims=True), -50.0, 50.0))
            return source_bagging._normalize_probability_rows(logits, epsilon=epsilon)
        predictions = np.asarray(model.predict(features), dtype=object)
        if predictions.ndim == 0 or predictions.shape[0] != n_rows:
            raise _estimator_row_count_error("predictions")
        output = np.full((n_rows, classes.shape[0]), float(epsilon), dtype=float)
        for row, label in enumerate(predictions.tolist()):
            class_index = source_bagging._label_index_or_none(label, classes)
            if class_index is not None:
                output[row, class_index] = 1.0
        return source_bagging._normalize_probability_rows(output, epsilon=epsilon)

    setattr(_aligned_probabilities, _PATCH_MARKER, True)
    source_bagging._aligned_probabilities = _aligned_probabilities


def install() -> None:
    """Install source-bagging numeric-option, feature-matrix, sampling, and estimator-output validation."""

    source_bagging = importlib.import_module("neureptrace.decoding.source_bagging")

    _install_without_replacement_row_sampling(source_bagging)
    _install_aligned_probability_validation(source_bagging)

    original_feature_matrix = source_bagging._feature_matrix
    if not getattr(original_feature_matrix, _PATCH_MARKER, False):

        @wraps(original_feature_matrix)
        def _feature_matrix(values: Any, *, name: str):
            materialized = _materialize_one_pass_iterables(values)
            if _contains_boolean_value(materialized):
                raise _feature_matrix_error(name)
            return original_feature_matrix(materialized, name=name)

        setattr(_feature_matrix, _PATCH_MARKER, True)
        source_bagging._feature_matrix = _feature_matrix

    original_config = source_bagging.source_bagging_config
    if not getattr(original_config, _PATCH_MARKER, False):

        @wraps(original_config)
        def source_bagging_config(
            *,
            n_estimators: Any = source_bagging.DEFAULT_N_ESTIMATORS,
            sample_fraction: Any = source_bagging.DEFAULT_SAMPLE_FRACTION,
            feature_fraction: Any = source_bagging.DEFAULT_FEATURE_FRACTION,
            bootstrap_rows: Any = True,
            bootstrap_features: Any = False,
            class_balanced: Any = True,
            random_state: Any = 13,
            epsilon: Any = source_bagging.DEFAULT_EPSILON,
        ):
            return _validate_config(
                original_config(
                    n_estimators=_positive_int(n_estimators, name="n_estimators"),
                    sample_fraction=_bounded_fraction(sample_fraction, name="sample_fraction"),
                    feature_fraction=_bounded_fraction(feature_fraction, name="feature_fraction"),
                    bootstrap_rows=bootstrap_rows,
                    bootstrap_features=bootstrap_features,
                    class_balanced=class_balanced,
                    random_state=random_state,
                    epsilon=_positive_float(epsilon, name="epsilon"),
                )
            )

        setattr(source_bagging_config, _PATCH_MARKER, True)
        source_bagging.source_bagging_config = source_bagging_config

    normalized_config = source_bagging.source_bagging_config
    original_coerce_config = source_bagging._coerce_config
    if not getattr(original_coerce_config, _PATCH_MARKER, False):

        @wraps(original_coerce_config)
        def _coerce_config(config: Any):
            if isinstance(config, source_bagging.SourceBaggingConfig):
                return normalized_config(
                    n_estimators=config.n_estimators,
                    sample_fraction=config.sample_fraction,
                    feature_fraction=config.feature_fraction,
                    bootstrap_rows=config.bootstrap_rows,
                    bootstrap_features=config.bootstrap_features,
                    class_balanced=config.class_balanced,
                    random_state=config.random_state,
                    epsilon=config.epsilon,
                )
            return _validate_config(original_coerce_config(config))

        setattr(_coerce_config, _PATCH_MARKER, True)
        source_bagging._coerce_config = _coerce_config


install()

__all__ = ["install"]

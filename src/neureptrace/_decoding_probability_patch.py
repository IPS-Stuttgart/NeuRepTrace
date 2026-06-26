"""Runtime robustness patch for decoder probability emissions.

The public decoding API is used by several downstream workflows that treat
classifier outputs as emission probabilities.  This patch keeps that API stable
while making probability outputs valid even when an estimator emits small
numerical pathologies such as NaNs, negative entries, or all-zero rows.
It also keeps binary ``decision_function`` fallbacks calibrated: a one-dimensional
binary decision score is a class-margin/logit difference, so converting it to
``[-score, score]`` would silently double the logit margin.
It can be folded directly into ``neureptrace.decoding`` later.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps

import numpy as np


_PATCH_MARKER = "_neureptrace_probability_patch_installed"
_BINARY_DECISION_PATCH_MARKER = "_neureptrace_binary_decision_probability_patch_installed"


def _prediction_fallback_probabilities(
    model,
    features,
    *,
    n_rows: int,
    n_classes: int,
) -> np.ndarray:
    """Return one-hot prediction fallbacks when possible, else uniform rows."""

    fallback = np.full((n_rows, n_classes), 1.0 / float(n_classes), dtype=float)
    if not (hasattr(model, "predict") and hasattr(model, "classes_")):
        return fallback

    classes = np.asarray(model.classes_)
    if classes.shape[0] != n_classes:
        return fallback

    try:
        predictions = np.asarray(model.predict(features))
    except Exception:  # pragma: no cover - defensive fallback for third-party estimators
        return fallback
    if predictions.shape[0] != n_rows:
        return fallback

    class_to_index = {label: index for index, label in enumerate(classes.tolist())}
    for row_index, label in enumerate(predictions.tolist()):
        try:
            class_index = class_to_index[label]
        except (KeyError, TypeError):
            continue
        fallback[row_index, :] = 0.0
        fallback[row_index, class_index] = 1.0
    return fallback


def _sanitize_probability_matrix(
    probabilities,
    *,
    model,
    features,
) -> np.ndarray:
    """Clip and normalize classifier probability outputs row-wise."""

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Predicted probabilities must be a two-dimensional matrix.")
    n_rows, n_classes = probabilities.shape
    if n_classes < 1:
        raise ValueError("Predicted probabilities must contain at least one class column.")

    clean = np.where(np.isfinite(probabilities) & (probabilities > 0.0), probabilities, 0.0)

    positive_infinite = np.isposinf(probabilities)
    rows_with_positive_infinity = positive_infinite.any(axis=1)
    if np.any(rows_with_positive_infinity):
        clean[rows_with_positive_infinity, :] = positive_infinite[rows_with_positive_infinity, :].astype(float)

    row_sums = clean.sum(axis=1, keepdims=True)
    valid_rows = row_sums[:, 0] > 0.0
    if np.any(valid_rows):
        clean[valid_rows, :] = clean[valid_rows, :] / row_sums[valid_rows, :]
    if np.any(~valid_rows):
        fallback = _prediction_fallback_probabilities(
            model,
            features,
            n_rows=n_rows,
            n_classes=n_classes,
        )
        clean[~valid_rows, :] = fallback[~valid_rows, :]
    return clean


def _binary_decision_scores_to_logits(scores) -> np.ndarray:
    """Convert 1-D binary decision margins to two logits without changing margin."""

    margins = np.asarray(scores, dtype=float).reshape(-1)
    half_margins = 0.5 * margins
    return np.column_stack([-half_margins, half_margins])


def _patch_source_free_decision_fallback() -> None:
    source_free = importlib.import_module("neureptrace.decoding.source_free")
    original = source_free._predict_source_probabilities
    if getattr(original, _BINARY_DECISION_PATCH_MARKER, False):
        return

    @wraps(original)
    def _predict_source_probabilities(model, features, classes):
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(features), dtype=float)
        elif hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(features), dtype=float)
            if scores.ndim == 1:
                scores = _binary_decision_scores_to_logits(scores)
            elif scores.ndim != 2:
                raise ValueError("source_model decision_function output must be one- or two-dimensional.")
            probabilities = source_free._softmax_rows(scores)
        else:
            raise ValueError("source_model must expose predict_proba or decision_function.")
        model_classes = source_free._as_label_vector(getattr(model, "classes_", classes), "source_model.classes_")
        return source_free._align_probability_columns(probabilities, model_classes=model_classes, classes=classes)

    setattr(_predict_source_probabilities, _BINARY_DECISION_PATCH_MARKER, True)
    source_free._predict_source_probabilities = _predict_source_probabilities


def _patch_source_ensemble_decision_fallback() -> None:
    source_ensemble = importlib.import_module("neureptrace.decoding.source_ensemble")
    original = source_ensemble._decision_probabilities
    if getattr(original, _BINARY_DECISION_PATCH_MARKER, False):
        return

    @wraps(original)
    def _decision_probabilities(model, features, classes):
        scores = np.asarray(model.decision_function(features), dtype=float)
        model_classes = np.asarray(getattr(model, "classes_", ()), dtype=object)
        if scores.ndim == 1:
            if model_classes.size == 0:
                if classes.shape[0] != 2:
                    raise ValueError("Binary decision_function alignment requires model.classes_ when the global class set is not binary.")
                model_classes = classes
            if model_classes.shape[0] != 2:
                raise ValueError("One-dimensional decision_function output requires exactly two model classes.")
            scores = _binary_decision_scores_to_logits(scores)
        elif scores.ndim == 2:
            if model_classes.size == 0:
                if scores.shape[1] != classes.shape[0]:
                    raise ValueError("Multiclass decision_function alignment requires model.classes_ when output width differs from the global class count.")
                model_classes = classes
        else:
            raise ValueError("decision_function output must be one- or two-dimensional.")
        if scores.shape[0] != features.shape[0]:
            raise ValueError("decision_function output must contain one row per feature row.")
        if scores.shape[1] != model_classes.shape[0]:
            raise ValueError("decision_function output width must match model.classes_.")
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        return np.exp(np.clip(shifted, -50.0, 50.0)), model_classes

    setattr(_decision_probabilities, _BINARY_DECISION_PATCH_MARKER, True)
    source_ensemble._decision_probabilities = _decision_probabilities


def _patch_decision_probability_helper(module_name: str, function_name: str) -> None:
    module = importlib.import_module(module_name)
    original = getattr(module, function_name)
    if getattr(original, _BINARY_DECISION_PATCH_MARKER, False):
        return

    @wraps(original)
    def _probabilities_or_none(model, features):
        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(features), dtype=float)
            return module._normalize_probability_rows(probabilities)
        if hasattr(model, "decision_function"):
            scores = np.asarray(model.decision_function(features), dtype=float)
            if scores.ndim == 1:
                scores = _binary_decision_scores_to_logits(scores)
            elif scores.ndim != 2:
                raise ValueError("decision_function output must be one- or two-dimensional.")
            shifted = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
            return module._normalize_probability_rows(exp_scores)
        return None

    setattr(_probabilities_or_none, _BINARY_DECISION_PATCH_MARKER, True)
    setattr(module, function_name, _probabilities_or_none)


def _patch_binary_decision_probability_fallbacks() -> None:
    _patch_source_free_decision_fallback()
    _patch_source_ensemble_decision_fallback()
    _patch_decision_probability_helper("neureptrace.decoding.subspace_alignment", "_probabilities_or_none")
    _patch_decision_probability_helper("neureptrace.decoding.transfer_components", "_predict_probabilities_or_none")
    _patch_decision_probability_helper("neureptrace.decoding.transfer_component_analysis", "_predict_probabilities_or_none")


def install() -> None:
    """Install robust probability, regularization-grid, and decision-score helpers."""

    from neureptrace import decoding

    if not getattr(decoding, _PATCH_MARKER, False):
        original_predict_emission_probabilities = decoding.predict_emission_probabilities
        original_parse_c_grid = decoding.parse_c_grid

        def predict_emission_probabilities(model, features: np.ndarray, *, emission_mode: str = "calibrated") -> np.ndarray:
            emission_mode_normalized = decoding.normalize_emission_mode(emission_mode)
            if hasattr(model, "predict_proba") and not (
                emission_mode_normalized == "uncalibrated" and hasattr(model, "decision_function")
            ):
                probabilities = model.predict_proba(features)
            else:
                probabilities = original_predict_emission_probabilities(
                    model,
                    features,
                    emission_mode=emission_mode_normalized,
                )
            return _sanitize_probability_matrix(probabilities, model=model, features=features)

        def parse_c_grid(values: Sequence[float] | str | None) -> tuple[float, ...]:
            grid = original_parse_c_grid(values)
            if any((not np.isfinite(value)) or value <= 0.0 for value in grid):
                raise ValueError("All C values must be positive finite numbers.")
            return grid

        predict_emission_probabilities.__doc__ = original_predict_emission_probabilities.__doc__
        parse_c_grid.__doc__ = original_parse_c_grid.__doc__
        decoding.predict_emission_probabilities = predict_emission_probabilities
        decoding.parse_c_grid = parse_c_grid
        setattr(decoding, _PATCH_MARKER, True)

    _patch_binary_decision_probability_fallbacks()

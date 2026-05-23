"""Runtime robustness patch for decoder probability emissions.

The public decoding API is used by several downstream workflows that treat
classifier outputs as emission probabilities.  This patch keeps that API stable
while making probability outputs valid even when an estimator emits small
numerical pathologies such as NaNs, negative entries, or all-zero rows.
It can be folded directly into ``neureptrace.decoding`` later.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


_PATCH_MARKER = "_neureptrace_probability_patch_installed"


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


def install() -> None:
    """Install robust probability and regularization-grid helpers."""

    from neureptrace import decoding

    if getattr(decoding, _PATCH_MARKER, False):
        return

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

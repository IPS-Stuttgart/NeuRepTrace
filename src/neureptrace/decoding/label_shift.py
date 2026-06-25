"""Category-2 label-shift adaptation for probability rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

LABEL_SHIFT_PROTOCOL = "unlabeled_target_label_shift_adaptation"
LABEL_SHIFT_CATEGORY = "2_unlabeled_target_adaptive"
LABEL_SHIFT_METHODS = ("em", "bbse", "bbse_em")


@dataclass(frozen=True, slots=True)
class LabelShiftAdaptationResult:
    """Adapted probabilities and protocol metadata."""

    probabilities: np.ndarray
    target_prior: tuple[float, ...]
    source_prior: tuple[float, ...]
    prior_ratio: tuple[float, ...]
    classes: tuple[Any, ...]
    method: str
    n_iterations: int
    converged: bool
    confusion_matrix: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def adapt_label_shift_probabilities(
    target_probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    method: str | None = "em",
    source_prior: Mapping[Any, float] | Sequence[float] | np.ndarray | None = None,
    source_labels: Sequence[Any] | np.ndarray | None = None,
    classes: Sequence[Any] | np.ndarray | None = None,
    source_validation_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    source_validation_labels: Sequence[Any] | np.ndarray | None = None,
    max_iter: int | str = 1000,
    tol: float | str = 1e-9,
    epsilon: float | str = 1e-12,
    bbse_regularization: float | str = 1e-6,
) -> LabelShiftAdaptationResult:
    """Estimate target priors from unlabeled target probabilities and adjust rows.

    This is a Category-2 protocol: source labels or source validation predictions
    may be used, target probabilities may be used, and target labels are not part
    of the public API.
    """

    target = _probability_matrix(target_probabilities, name="target_probabilities", epsilon=epsilon)
    method_name = normalize_label_shift_method(method)
    eps = _positive_float(epsilon, name="epsilon")
    class_order = _resolve_classes(target.shape[1], classes, source_labels, source_validation_labels)
    source = _resolve_source_prior(
        target.shape[1],
        source_prior=source_prior,
        source_labels=source_labels,
        source_validation_labels=source_validation_labels,
        classes=class_order,
        epsilon=eps,
    )
    confusion = None
    initial = None
    if method_name in {"bbse", "bbse_em"}:
        if source_validation_probabilities is None or source_validation_labels is None:
            raise ValueError("BBSE requires source_validation_probabilities and source_validation_labels.")
        source_probs = _probability_matrix(
            source_validation_probabilities,
            name="source_validation_probabilities",
            epsilon=eps,
            expected_classes=target.shape[1],
        )
        confusion = soft_confusion_matrix(source_probs, source_validation_labels, classes=class_order, epsilon=eps)
        initial = estimate_target_prior_bbse(target, confusion_matrix=confusion, regularization=bbse_regularization, epsilon=eps)
        if method_name == "bbse":
            adjusted = adjust_probabilities_to_prior(target, source_prior=source, target_prior=initial, epsilon=eps)
            return _result(method_name, adjusted, initial, source, class_order, 0, True, confusion, bbse_regularization)

    prior, adjusted, iterations, converged = estimate_target_prior_em(
        target,
        source_prior=source,
        initial_target_prior=initial,
        max_iter=max_iter,
        tol=tol,
        epsilon=eps,
    )
    return _result(method_name, adjusted, prior, source, class_order, iterations, converged, confusion, bbse_regularization if confusion is not None else None)


def estimate_target_prior_em(
    target_probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_prior: Sequence[float] | np.ndarray,
    initial_target_prior: Sequence[float] | np.ndarray | None = None,
    max_iter: int | str = 1000,
    tol: float | str = 1e-9,
    epsilon: float | str = 1e-12,
) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Estimate target prior by EM and return adjusted probabilities."""

    probs = _probability_matrix(target_probabilities, name="target_probabilities", epsilon=epsilon)
    eps = _positive_float(epsilon, name="epsilon")
    source = _prior_vector(source_prior, n_classes=probs.shape[1], name="source_prior", epsilon=eps)
    prior = source.copy() if initial_target_prior is None else _prior_vector(initial_target_prior, n_classes=probs.shape[1], name="initial_target_prior", epsilon=eps)
    n_iter = _positive_int(max_iter, name="max_iter")
    tolerance = _nonnegative_float(tol, name="tol")
    adjusted = adjust_probabilities_to_prior(probs, source_prior=source, target_prior=prior, epsilon=eps)
    for iteration in range(1, n_iter + 1):
        new_prior = _normalize_prior(np.mean(adjusted, axis=0), epsilon=eps)
        converged = bool(np.max(np.abs(new_prior - prior)) <= tolerance)
        prior = new_prior
        adjusted = adjust_probabilities_to_prior(probs, source_prior=source, target_prior=prior, epsilon=eps)
        if converged:
            return prior, adjusted, iteration, True
    return prior, adjusted, n_iter, False


def estimate_target_prior_bbse(
    target_probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    confusion_matrix: Sequence[Sequence[float]] | np.ndarray,
    regularization: float | str = 1e-6,
    epsilon: float | str = 1e-12,
) -> np.ndarray:
    """Estimate target prior by soft confusion-matrix inversion."""

    probs = _probability_matrix(target_probabilities, name="target_probabilities", epsilon=epsilon)
    eps = _positive_float(epsilon, name="epsilon")
    confusion = _confusion_matrix(confusion_matrix, n_classes=probs.shape[1], epsilon=eps)
    ridge = _nonnegative_float(regularization, name="regularization")
    marginal = _normalize_prior(np.mean(probs, axis=0), epsilon=eps)
    lhs = confusion.T @ confusion + ridge * np.eye(confusion.shape[1])
    rhs = confusion.T @ marginal
    try:
        prior = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        prior = np.linalg.pinv(lhs) @ rhs
    return _normalize_prior(np.maximum(prior, eps), epsilon=eps)


def soft_confusion_matrix(
    source_validation_probabilities: Sequence[Sequence[float]] | np.ndarray,
    source_validation_labels: Sequence[Any] | np.ndarray,
    *,
    classes: Sequence[Any] | np.ndarray,
    epsilon: float | str = 1e-12,
) -> np.ndarray:
    """Return a column-normalized soft confusion matrix from source validation rows."""

    probs = _probability_matrix(source_validation_probabilities, name="source_validation_probabilities", epsilon=epsilon)
    labels = _object_vector(source_validation_labels, name="source_validation_labels")
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("source_validation_labels must contain one label per source validation row.")
    class_order = _object_vector(classes, name="classes")
    if class_order.shape[0] != probs.shape[1]:
        raise ValueError("classes length must match probability columns.")
    confusion = np.zeros((probs.shape[1], probs.shape[1]), dtype=float)
    for class_index, class_label in enumerate(class_order.tolist()):
        mask = labels == class_label
        if not np.any(mask):
            raise ValueError(f"source_validation_labels contain no rows for class {class_label!r}.")
        confusion[:, class_index] = np.mean(probs[mask], axis=0)
    return _column_normalize(confusion, epsilon=_positive_float(epsilon, name="epsilon"))


def adjust_probabilities_to_prior(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    *,
    source_prior: Sequence[float] | np.ndarray,
    target_prior: Sequence[float] | np.ndarray,
    epsilon: float | str = 1e-12,
) -> np.ndarray:
    """Apply prior-ratio correction to probability rows."""

    matrix = _probability_matrix(probabilities, name="probabilities", epsilon=epsilon)
    eps = _positive_float(epsilon, name="epsilon")
    source = _prior_vector(source_prior, n_classes=matrix.shape[1], name="source_prior", epsilon=eps)
    target = _prior_vector(target_prior, n_classes=matrix.shape[1], name="target_prior", epsilon=eps)
    adjusted = matrix * (target / np.maximum(source, eps))[None, :]
    return _normalize_probability_rows(adjusted, epsilon=eps)


def normalize_label_shift_method(value: str | None) -> str:
    """Normalize method aliases."""

    normalized = "em" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "saerens": "em",
        "saerens_em": "em",
        "expectation_maximization": "em",
        "black_box_shift": "bbse",
        "black_box_shift_estimation": "bbse",
        "confusion_inversion": "bbse",
        "bbse_initialized_em": "bbse_em",
        "bbse_then_em": "bbse_em",
    }.get(normalized, normalized)
    if normalized not in LABEL_SHIFT_METHODS:
        raise ValueError(f"Unknown label-shift method {value!r}. Available methods: {', '.join(LABEL_SHIFT_METHODS)}.")
    return normalized


def _result(method: str, probabilities: np.ndarray, target_prior: np.ndarray, source_prior: np.ndarray, classes: tuple[Any, ...], n_iterations: int, converged: bool, confusion_matrix: np.ndarray | None, bbse_regularization: float | str | None) -> LabelShiftAdaptationResult:
    ratio = target_prior / np.maximum(source_prior, 1e-12)
    metadata = {
        "label_shift_adaptation": True,
        "label_shift_protocol": LABEL_SHIFT_PROTOCOL,
        "label_shift_protocol_category": LABEL_SHIFT_CATEGORY,
        "label_shift_method": method,
        "label_shift_uses_source_labels": True,
        "label_shift_uses_source_validation_probabilities": confusion_matrix is not None,
        "label_shift_uses_target_probabilities": True,
        "label_shift_uses_target_labels": False,
        "label_shift_valid_for_strict_source_only": False,
        "label_shift_valid_for_unlabeled_target_adaptation": True,
        "label_shift_valid_for_target_calibration": False,
        "label_shift_n_target_rows": int(probabilities.shape[0]),
        "label_shift_n_classes": int(probabilities.shape[1]),
        "label_shift_source_prior": _format_vector(source_prior),
        "label_shift_target_prior": _format_vector(target_prior),
        "label_shift_prior_ratio": _format_vector(ratio),
        "label_shift_n_iterations": int(n_iterations),
        "label_shift_converged": bool(converged),
        "label_shift_bbse_regularization": "" if bbse_regularization is None else float(bbse_regularization),
        "label_shift_confusion_condition_number": "" if confusion_matrix is None else float(np.linalg.cond(confusion_matrix)),
    }
    return LabelShiftAdaptationResult(
        probabilities=probabilities.astype(np.float32, copy=False),
        target_prior=tuple(float(v) for v in target_prior),
        source_prior=tuple(float(v) for v in source_prior),
        prior_ratio=tuple(float(v) for v in ratio),
        classes=classes,
        method=method,
        n_iterations=int(n_iterations),
        converged=bool(converged),
        confusion_matrix=None if confusion_matrix is None else np.asarray(confusion_matrix, dtype=float),
        metadata=metadata,
    )


def _resolve_classes(n_classes: int, classes, source_labels, source_validation_labels) -> tuple[Any, ...]:
    if classes is not None:
        class_order = _object_vector(classes, name="classes")
        if class_order.shape[0] != n_classes:
            raise ValueError("classes length must match probability columns.")
        return tuple(class_order.tolist())
    labels = source_labels if source_labels is not None else source_validation_labels
    if labels is not None:
        order = tuple(dict.fromkeys(_object_vector(labels, name="source_labels").tolist()))
        if len(order) != n_classes:
            raise ValueError("Inferred class count from labels does not match probability columns; pass classes explicitly.")
        return order
    return tuple(range(n_classes))


def _resolve_source_prior(n_classes: int, *, source_prior, source_labels, source_validation_labels, classes: Sequence[Any], epsilon: float) -> np.ndarray:
    if source_prior is not None:
        if isinstance(source_prior, Mapping):
            values = np.asarray([source_prior.get(label, 0.0) for label in classes], dtype=float)
        else:
            values = np.asarray(source_prior, dtype=float).reshape(-1)
        return _prior_vector(values, n_classes=n_classes, name="source_prior", epsilon=epsilon)
    labels = source_labels if source_labels is not None else source_validation_labels
    if labels is None:
        raise ValueError("source_prior, source_labels, or source_validation_labels are required.")
    label_vector = _object_vector(labels, name="source_labels")
    counts = np.asarray([np.count_nonzero(label_vector == label) for label in classes], dtype=float)
    if np.any(counts <= 0.0):
        raise ValueError("source labels must contain at least one row per class.")
    return _normalize_prior(counts, epsilon=epsilon)


def _format_vector(values: np.ndarray) -> str:
    return "|".join(f"{float(v):.12g}" for v in np.asarray(values, dtype=float).reshape(-1))


def _prior_vector(values, *, n_classes: int, name: str, epsilon: float) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.shape[0] != n_classes:
        raise ValueError(f"{name} must contain one value per class.")
    return _normalize_prior(vector, epsilon=epsilon)


def _normalize_prior(values, *, epsilon: float) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    if vector.size < 1 or not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("Prior vectors must contain finite non-negative values.")
    vector = np.maximum(vector, epsilon)
    total = float(np.sum(vector))
    if total <= 0.0:
        raise ValueError("Prior vectors must have positive mass.")
    return vector / total


def _probability_matrix(values, *, name: str, epsilon: float | str, expected_classes: int | None = None) -> np.ndarray:
    eps = _positive_float(epsilon, name="epsilon")
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError(f"{name} must be two-dimensional with at least two class columns.")
    if expected_classes is not None and matrix.shape[1] != expected_classes:
        raise ValueError(f"{name} has the wrong number of class columns.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError(f"{name} must contain finite non-negative probabilities.")
    return _normalize_probability_rows(matrix, epsilon=eps)


def _normalize_probability_rows(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    clipped = np.maximum(np.asarray(matrix, dtype=float), epsilon)
    row_sums = np.sum(clipped, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Probability rows must have positive mass.")
    return clipped / row_sums


def _confusion_matrix(values, *, n_classes: int, epsilon: float | str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != (n_classes, n_classes) or not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("confusion_matrix must be a finite non-negative square class matrix.")
    return _column_normalize(matrix, epsilon=_positive_float(epsilon, name="epsilon"))


def _column_normalize(matrix: np.ndarray, *, epsilon: float) -> np.ndarray:
    clipped = np.maximum(matrix, epsilon)
    return clipped / np.sum(clipped, axis=0, keepdims=True)


def _object_vector(values, *, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.dtype == object and values.ndim == 1:
        return values.reshape(-1)
    try:
        items = list(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence.") from exc
    vector = np.empty(len(items), dtype=object)
    for index, value in enumerate(items):
        vector[index] = value
    return vector


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _nonnegative_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return parsed

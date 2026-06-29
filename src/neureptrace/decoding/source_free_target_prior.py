from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from neureptrace.decoding.source_free import PseudoLabelSelection, SourceFreeAdaptationResult, fit_source_free_predict_proba

_EPS = 1e-12
_NEGATIVE_TOLERANCE = 1e-10
TargetPriorCorrection = Literal["none", "balanced"]
TargetPriorEstimator = Literal["mean", "confidence_weighted", "entropy_weighted"]


@dataclass(frozen=True, slots=True)
class SourceFreeTargetPriorCorrectionResult:
    """Source-free adaptation result with optional unlabeled target-prior correction.

    ``base_result`` is the uncorrected source-free adaptation result.  The final
    ``probabilities`` may be target-prior corrected, and ``metadata`` keeps the
    same protocol-hygiene contract: target labels and source rows are not used by
    the correction step.
    """

    base_result: SourceFreeAdaptationResult
    probabilities: np.ndarray
    metadata: dict[str, Any]


def fit_source_free_target_prior_predict_proba(
    *,
    source_model: Any,
    target_features: np.ndarray,
    classes: np.ndarray | list[Any] | tuple[Any, ...] | None = None,
    confidence_threshold: float = 0.75,
    max_iterations: int = 5,
    min_class_count: int = 1,
    min_active_classes: int = 2,
    prototype_weight: float = 0.5,
    prototype_temperature: float = 1.0,
    standardize_target: bool = True,
    feature_space: Literal["input", "model_preprocessor", "auto"] = "auto",
    pseudo_label_selection: PseudoLabelSelection = "confidence",
    balanced_topk_per_class: int | None = None,
    prototype_estimator: str = "hard",
    target_prior_correction: TargetPriorCorrection = "balanced",
    target_prior_strength: float = 1.0,
    target_prior_estimator: TargetPriorEstimator = "mean",
    target_prior_smoothing: float = 0.0,
    target_prior_floor: float = 0.0,
) -> SourceFreeTargetPriorCorrectionResult:
    """Fit source-free adaptation and correct target priors from unlabeled predictions.

    The target-prior correction estimates only the marginal predicted class
    distribution on the unlabeled target batch.  It never accepts target labels,
    source samples, or source labels during adaptation/correction, so it remains
    compatible with Protocol 2 / 2.5 style OpenNeuro runs.

    ``target_prior_smoothing`` blends the estimated marginal toward a uniform
    prior before correction. ``target_prior_floor`` then lower-bounds each class
    prior before renormalization.  These guards make the balanced correction less
    brittle on OpenNeuro folds where a source model nearly collapses to one
    pseudo-class.
    """

    base_result = fit_source_free_predict_proba(
        source_model=source_model,
        target_features=target_features,
        classes=classes,
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations,
        min_class_count=min_class_count,
        min_active_classes=min_active_classes,
        prototype_weight=prototype_weight,
        prototype_temperature=prototype_temperature,
        standardize_target=standardize_target,
        feature_space=feature_space,
        pseudo_label_selection=pseudo_label_selection,
        balanced_topk_per_class=balanced_topk_per_class,
        prototype_estimator=prototype_estimator,
    )
    raw_target_prior = estimate_target_class_prior(
        base_result.probabilities,
        estimator=target_prior_estimator,
    )
    probabilities, target_prior = apply_target_prior_correction(
        base_result.probabilities,
        mode=target_prior_correction,
        strength=target_prior_strength,
        prior=raw_target_prior,
        smoothing=target_prior_smoothing,
        floor=target_prior_floor,
    )
    metadata = {
        **base_result.metadata,
        "source_free_target_prior_correction": _target_prior_correction_mode(target_prior_correction),
        "source_free_target_prior_strength": _bounded_strength(target_prior_strength),
        "source_free_target_prior_estimator": _target_prior_estimator_mode(target_prior_estimator),
        "source_free_target_prior_smoothing": _bounded_strength(target_prior_smoothing, name="target_prior_smoothing"),
        "source_free_target_prior_floor": _bounded_floor(target_prior_floor),
        "source_free_target_raw_class_prior": format_target_prior(raw_target_prior),
        "source_free_target_class_prior": format_target_prior(target_prior),
        "source_free_target_prior_uses_target_labels": False,
        "source_free_target_prior_uses_source_rows": False,
        "source_free_valid_for_benchmark": True,
    }
    return SourceFreeTargetPriorCorrectionResult(
        base_result=base_result,
        probabilities=probabilities,
        metadata=metadata,
    )


def apply_target_prior_correction(
    probabilities: np.ndarray,
    *,
    mode: TargetPriorCorrection | str = "balanced",
    strength: float = 1.0,
    prior: np.ndarray | None = None,
    estimator: TargetPriorEstimator | str = "mean",
    smoothing: float = 0.0,
    floor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities corrected by the unlabeled target marginal prior.

    ``estimator`` is used only when ``prior`` is not supplied.  ``smoothing`` and
    ``floor`` are applied to the prior returned alongside the corrected
    probabilities, so the metadata can record the exact prior used for the
    correction.
    """

    normalized = _normalize_probability_rows(probabilities)
    parsed_mode = _target_prior_correction_mode(mode)
    parsed_strength = _bounded_strength(strength)
    if parsed_mode == "none":
        if prior is None:
            raw_prior = estimate_target_class_prior(normalized, estimator=estimator)
        else:
            raw_prior = _validate_prior(prior, n_classes=normalized.shape[1])
        target_prior = stabilize_target_class_prior(
            raw_prior,
            smoothing=smoothing,
            floor=floor,
        )
        return normalized, target_prior
    raw_prior = (
        estimate_target_class_prior(normalized, estimator=estimator)
        if prior is None
        else _validate_prior(prior, n_classes=normalized.shape[1])
    )
    target_prior = stabilize_target_class_prior(
        raw_prior,
        smoothing=smoothing,
        floor=floor,
    )
    if parsed_strength == 0.0:
        return normalized, target_prior
    corrected = normalized / np.power(target_prior[np.newaxis, :], parsed_strength)
    return _normalize_probability_rows(corrected), target_prior


def estimate_target_class_prior(
    probabilities: np.ndarray,
    *,
    estimator: TargetPriorEstimator | str = "mean",
) -> np.ndarray:
    """Estimate target class prior from unlabeled predicted probabilities."""

    normalized = _normalize_probability_rows(probabilities)
    parsed_estimator = _target_prior_estimator_mode(estimator)
    if parsed_estimator == "mean":
        prior = normalized.mean(axis=0)
    else:
        confidence = normalized.max(axis=1)
        if parsed_estimator == "confidence_weighted":
            weights = confidence
        elif parsed_estimator == "entropy_weighted":
            entropy = -np.sum(normalized * np.log(np.clip(normalized, _EPS, 1.0)), axis=1)
            max_entropy = np.log(float(normalized.shape[1]))
            weights = 1.0 - entropy / max(max_entropy, _EPS)
        else:  # pragma: no cover - parser keeps this unreachable.
            raise ValueError(f"unsupported target-prior estimator: {parsed_estimator!r}")
        weights = np.clip(np.asarray(weights, dtype=float), _EPS, None)
        prior = np.average(normalized, axis=0, weights=weights)
    return _validate_prior(prior, n_classes=normalized.shape[1])


def stabilize_target_class_prior(
    prior: np.ndarray,
    *,
    smoothing: float = 0.0,
    floor: float = 0.0,
) -> np.ndarray:
    """Blend a target-prior estimate toward uniform and optionally floor it."""

    target_prior = _validate_prior(prior, n_classes=np.asarray(prior).reshape(-1).shape[0])
    smooth = _bounded_strength(smoothing, name="target_prior_smoothing")
    prior_floor = _bounded_floor(floor)
    n_classes = int(target_prior.shape[0])
    if smooth:
        uniform = np.full(n_classes, 1.0 / n_classes, dtype=float)
        target_prior = (1.0 - smooth) * target_prior + smooth * uniform
    if prior_floor:
        if prior_floor * n_classes >= 1.0:
            raise ValueError("target_prior_floor must be smaller than 1 / n_classes.")
        target_prior = np.maximum(target_prior, prior_floor)
    return _validate_prior(target_prior, n_classes=n_classes)


def format_target_prior(prior: np.ndarray) -> str:
    """Serialize a target-prior vector for run metadata."""

    return "|".join(f"{float(value):.6g}" for value in np.asarray(prior, dtype=float).reshape(-1))


def _target_prior_correction_mode(value: Any) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"", "none", "off", "false", "no", "0", "disabled", "disable"}:
        return "none"
    if mode in {
        "balanced",
        "uniform",
        "train_uniform",
        "target_balanced",
        "balanced_smoothed",
        "smoothed_balanced",
        "on",
        "true",
        "yes",
        "1",
        "enabled",
        "enable",
    }:
        return "balanced"
    raise ValueError("target_prior_correction must be one of: none, balanced.")


def _target_prior_estimator_mode(value: Any) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"", "mean", "posterior_mean", "average"}:
        return "mean"
    if mode in {"confidence_weighted", "confidence", "maxprob_weighted", "max_probability_weighted"}:
        return "confidence_weighted"
    if mode in {"entropy_weighted", "low_entropy_weighted", "certainty_weighted"}:
        return "entropy_weighted"
    raise ValueError("target_prior_estimator must be one of: mean, confidence_weighted, entropy_weighted.")


def _bounded_strength(value: Any, *, name: str = "target_prior_strength") -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite in [0, 1].")
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be finite in [0, 1].")
    return number


def _bounded_floor(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("target_prior_floor must be finite in [0, 1).")
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number >= 1.0:
        raise ValueError("target_prior_floor must be finite in [0, 1).")
    return number


def _validate_prior(prior: np.ndarray, *, n_classes: int) -> np.ndarray:
    target_prior = np.asarray(prior, dtype=float).reshape(-1)
    if target_prior.shape[0] != int(n_classes):
        raise ValueError("target prior must have one entry per class.")
    if not np.all(np.isfinite(target_prior)) or np.any(target_prior < 0.0):
        raise ValueError("target prior must contain finite non-negative entries.")
    if float(target_prior.sum()) <= 0.0:
        raise ValueError("target prior must contain positive mass.")
    target_prior = np.clip(target_prior / target_prior.sum(), _EPS, None)
    return target_prior / target_prior.sum()


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    raw = np.asarray(probabilities)
    if np.issubdtype(raw.dtype, np.bool_) or (
        raw.dtype == object and any(isinstance(value, (bool, np.bool_)) for value in raw.reshape(-1))
    ):
        raise ValueError("probabilities must be numeric probability values, not boolean indicators.")
    try:
        array = raw.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be numeric.") from exc
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional matrix with at least two classes.")
    if not np.all(np.isfinite(array)):
        raise ValueError("probabilities must be finite.")
    if np.any(array < -_NEGATIVE_TOLERANCE):
        raise ValueError("probabilities must be non-negative.")
    if np.any(array < 0.0):
        array = np.where(array < 0.0, 0.0, array)
    row_sums = array.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return array / row_sums


__all__ = [
    "SourceFreeTargetPriorCorrectionResult",
    "apply_target_prior_correction",
    "estimate_target_class_prior",
    "fit_source_free_target_prior_predict_proba",
    "format_target_prior",
    "stabilize_target_class_prior",
]

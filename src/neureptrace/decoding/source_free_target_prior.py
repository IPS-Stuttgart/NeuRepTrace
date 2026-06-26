from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from neureptrace.decoding.source_free import SourceFreeAdaptationResult, fit_source_free_predict_proba

_EPS = 1e-12
TargetPriorCorrection = Literal["none", "balanced"]


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
    target_prior_correction: TargetPriorCorrection = "balanced",
    target_prior_strength: float = 1.0,
) -> SourceFreeTargetPriorCorrectionResult:
    """Fit source-free adaptation and correct target priors from unlabeled predictions.

    The target-prior correction estimates only the marginal predicted class
    distribution on the unlabeled target batch.  It never accepts target labels,
    source samples, or source labels during adaptation/correction, so it remains
    compatible with Protocol 2 / 2.5 style OpenNeuro runs.
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
    )
    probabilities, target_prior = apply_target_prior_correction(
        base_result.probabilities,
        mode=target_prior_correction,
        strength=target_prior_strength,
    )
    metadata = {
        **base_result.metadata,
        "source_free_target_prior_correction": _target_prior_correction_mode(target_prior_correction),
        "source_free_target_prior_strength": _bounded_strength(target_prior_strength),
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
) -> tuple[np.ndarray, np.ndarray]:
    """Return probabilities corrected by the unlabeled target marginal prior."""

    normalized = _normalize_probability_rows(probabilities)
    parsed_mode = _target_prior_correction_mode(mode)
    parsed_strength = _bounded_strength(strength)
    target_prior = estimate_target_class_prior(normalized) if prior is None else _validate_prior(prior, n_classes=normalized.shape[1])
    if parsed_mode == "none" or parsed_strength == 0.0:
        return normalized, target_prior
    corrected = normalized / np.power(target_prior[np.newaxis, :], parsed_strength)
    return _normalize_probability_rows(corrected), target_prior


def estimate_target_class_prior(probabilities: np.ndarray) -> np.ndarray:
    """Estimate target class prior from unlabeled predicted probabilities."""

    normalized = _normalize_probability_rows(probabilities)
    return _validate_prior(normalized.mean(axis=0), n_classes=normalized.shape[1])


def format_target_prior(prior: np.ndarray) -> str:
    """Serialize a target-prior vector for run metadata."""

    return "|".join(f"{float(value):.6g}" for value in np.asarray(prior, dtype=float).reshape(-1))


def _target_prior_correction_mode(value: Any) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode in {"", "none", "off", "false"}:
        return "none"
    if mode in {"balanced", "uniform", "train_uniform", "target_balanced"}:
        return "balanced"
    raise ValueError("target_prior_correction must be one of: none, balanced.")


def _bounded_strength(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("target_prior_strength must be finite in [0, 1].")
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError("target_prior_strength must be finite in [0, 1].")
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
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional matrix with at least two classes.")
    if not np.all(np.isfinite(array)):
        raise ValueError("probabilities must be finite.")
    array = np.clip(array, 0.0, None)
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
]

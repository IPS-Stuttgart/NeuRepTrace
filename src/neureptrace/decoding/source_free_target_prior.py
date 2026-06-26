from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from neureptrace.decoding.source_free import (
    SourceFreeAdaptationResult,
    _as_2d_array,
    _bounded_float,
    _normalize_probability_rows,
    _predict_source_probabilities,
    _resolve_classes,
    fit_source_free_predict_proba,
)

_EPS = 1e-12
TargetPriorCorrection = Literal["none", "balanced"]
TargetPriorCorrectionStage = Literal["post", "pre", "both"]


@dataclass(frozen=True, slots=True)
class SourceFreeTargetPriorCorrectionResult:
    """Source-free adaptation result with optional unlabeled target-prior correction.

    ``base_result`` is the uncorrected or pre-corrected source-free adaptation
    result. The final ``probabilities`` may be target-prior corrected, and
    ``metadata`` keeps the same protocol-hygiene contract: target labels and
    source rows are not used by the correction step.
    """

    base_result: SourceFreeAdaptationResult
    probabilities: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FittedTargetPriorCorrection:
    """Fitted target-prior correction estimated from unlabeled target predictions."""

    classes: np.ndarray
    prior: np.ndarray
    mode: str
    strength: float

    def correct_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Return row-normalized probabilities after applying this correction."""

        corrected, _ = apply_target_prior_correction(
            probabilities,
            mode=self.mode,
            strength=self.strength,
            prior=self.prior,
        )
        return corrected

    def metadata(self) -> dict[str, Any]:
        return {
            "source_free_target_prior_correction": self.mode,
            "source_free_target_prior_strength": float(self.strength),
            "source_free_target_class_prior": format_target_prior(self.prior),
            "source_free_target_prior_classes": _format_classes(self.classes),
            "source_free_target_prior_uses_target_features": self.mode != "none" and self.strength > 0.0,
            "source_free_target_prior_uses_target_labels": False,
            "source_free_target_prior_uses_source_rows": False,
            "source_free_target_prior_valid_for_benchmark": True,
        }


class TargetPriorCorrectedSourceModel:
    """Source-model wrapper whose probabilities are corrected before adaptation."""

    def __init__(self, source_model: Any, correction: FittedTargetPriorCorrection):
        self.source_model = source_model
        self.correction = correction
        self.classes_ = np.asarray(correction.classes, dtype=object)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = _as_2d_array(features, "features")
        probabilities = _predict_source_probabilities(self.source_model, features, self.classes_)
        return self.correction.correct_probabilities(probabilities)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)
        return np.log(np.clip(probabilities, _EPS, 1.0))


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
    target_prior_correction_stage: TargetPriorCorrectionStage = "post",
    pseudo_label_selection: Literal["confidence", "balanced_topk"] = "confidence",
    balanced_topk_per_class: int | None = None,
) -> SourceFreeTargetPriorCorrectionResult:
    """Fit source-free adaptation with optional unlabeled target-prior correction.

    ``target_prior_correction_stage`` controls where the correction is applied:

    - ``post`` preserves the original behavior and corrects final probabilities.
    - ``pre`` wraps the frozen source model before pseudo-label/prototype fitting.
    - ``both`` applies the train-free wrapper before adaptation and a final row
      renormalization with the same target prior.

    All modes estimate the target prior from source-model predictions on
    unlabeled target features only.  Target labels remain scoring-only.
    """

    mode = _target_prior_correction_mode(target_prior_correction)
    strength = _bounded_strength(target_prior_strength)
    stage = _target_prior_correction_stage(target_prior_correction_stage)

    correction: FittedTargetPriorCorrection | None = None
    model_for_adaptation = source_model
    classes_for_adaptation = classes
    if mode != "none" and strength > 0.0 and stage in {"pre", "both"}:
        correction = fit_target_prior_correction(
            source_model=source_model,
            target_features=target_features,
            classes=classes,
            mode=mode,
            strength=strength,
        )
        model_for_adaptation = TargetPriorCorrectedSourceModel(source_model, correction)
        classes_for_adaptation = correction.classes

    base_result = fit_source_free_predict_proba(
        source_model=model_for_adaptation,
        target_features=target_features,
        classes=classes_for_adaptation,
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
    )

    probabilities = base_result.probabilities
    if mode != "none" and strength > 0.0 and stage in {"post", "both"}:
        if correction is None:
            probabilities, target_prior = apply_target_prior_correction(
                base_result.probabilities,
                mode=mode,
                strength=strength,
            )
            correction = FittedTargetPriorCorrection(
                classes=np.asarray(base_result.adapter.classes_, dtype=object),
                prior=target_prior,
                mode=mode,
                strength=strength,
            )
        else:
            probabilities = correction.correct_probabilities(base_result.probabilities)
    elif correction is None:
        correction = FittedTargetPriorCorrection(
            classes=np.asarray(base_result.adapter.classes_, dtype=object),
            prior=estimate_target_class_prior(base_result.probabilities),
            mode=mode,
            strength=0.0 if mode == "none" else strength,
        )

    metadata = {
        **base_result.metadata,
        **correction.metadata(),
        "source_free_target_prior_correction_stage": stage if mode != "none" and strength > 0.0 else "none",
        "source_free_valid_for_benchmark": True,
    }
    return SourceFreeTargetPriorCorrectionResult(
        base_result=base_result,
        probabilities=probabilities,
        metadata=metadata,
    )


def fit_target_prior_correction(
    *,
    source_model: Any,
    target_features: np.ndarray,
    classes: np.ndarray | list[Any] | tuple[Any, ...] | None = None,
    mode: TargetPriorCorrection | str = "balanced",
    strength: float = 1.0,
) -> FittedTargetPriorCorrection:
    """Fit a target-prior correction from unlabeled source-model predictions."""

    x_target = _as_2d_array(target_features, "target_features")
    classes_array = _resolve_classes(source_model, classes)
    probabilities = _predict_source_probabilities(source_model, x_target, classes_array)
    return FittedTargetPriorCorrection(
        classes=classes_array,
        prior=estimate_target_class_prior(probabilities),
        mode=_target_prior_correction_mode(mode),
        strength=_bounded_strength(strength),
    )


def fit_target_prior_corrected_source_model(
    *,
    source_model: Any,
    target_features: np.ndarray,
    classes: np.ndarray | list[Any] | tuple[Any, ...] | None = None,
    mode: TargetPriorCorrection | str = "balanced",
    strength: float = 1.0,
) -> TargetPriorCorrectedSourceModel:
    """Return a source-model wrapper that applies pre-adaptation prior correction."""

    correction = fit_target_prior_correction(
        source_model=source_model,
        target_features=target_features,
        classes=classes,
        mode=mode,
        strength=strength,
    )
    return TargetPriorCorrectedSourceModel(source_model, correction)


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


def _target_prior_correction_stage(value: Any) -> str:
    stage = str(value).strip().lower().replace("-", "_")
    if stage in {"post", "after", "after_adaptation", "final"}:
        return "post"
    if stage in {"pre", "before", "before_adaptation", "source_model"}:
        return "pre"
    if stage in {"both", "pre_and_post", "source_and_final"}:
        return "both"
    raise ValueError("target_prior_correction_stage must be one of: post, pre, both.")


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
    return np.clip(target_prior / target_prior.sum(), _EPS, None)


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


def _format_classes(classes: np.ndarray) -> str:
    return "|".join(str(class_label) for class_label in np.asarray(classes, dtype=object).tolist())


__all__ = [
    "FittedTargetPriorCorrection",
    "SourceFreeTargetPriorCorrectionResult",
    "TargetPriorCorrectedSourceModel",
    "apply_target_prior_correction",
    "estimate_target_class_prior",
    "fit_source_free_target_prior_predict_proba",
    "fit_target_prior_corrected_source_model",
    "fit_target_prior_correction",
    "format_target_prior",
]

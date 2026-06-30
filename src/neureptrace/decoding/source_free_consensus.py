"""Source-free consensus ensembling for OpenNeuro Protocol 2.5 runs.

The utilities in this module combine several source-free target-adaptation
variants without using held-out target labels.  This is useful for OpenNeuro LOSO
folds where one variant is confident but collapsed to a single pseudo-class,
while another variant is better balanced but less confident.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from neureptrace.decoding.source_free_target_prior import (
    SourceFreeTargetPriorCorrectionResult,
    fit_source_free_target_prior_predict_proba,
)

_EPS = 1e-12
SOURCE_FREE_CONSENSUS_PROTOCOL = "source_free_unlabeled_target_consensus"
SOURCE_FREE_CONSENSUS_CATEGORY = "2_5_source_free_unlabeled_target_consensus"
ConsensusMode = Literal["logit_mean", "arithmetic_mean"]


@dataclass(frozen=True, slots=True)
class SourceFreeConsensusVariant:
    """One source-free variant to include in a target-batch consensus."""

    name: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    weight: float | None = None


@dataclass(frozen=True, slots=True)
class SourceFreeConsensusResult:
    """Consensus probabilities and source-free metadata."""

    probabilities: np.ndarray
    variant_probabilities: tuple[np.ndarray, ...]
    variant_results: tuple[SourceFreeTargetPriorCorrectionResult, ...]
    weights: np.ndarray
    metadata: dict[str, Any]


def default_source_free_consensus_variants() -> tuple[SourceFreeConsensusVariant, ...]:
    """Return a conservative Protocol-2.5 source-free ensemble recipe."""

    return (
        SourceFreeConsensusVariant(
            "source_raw",
            {
                "max_iterations": 0,
                "target_prior_correction": "none",
            },
        ),
        SourceFreeConsensusVariant(
            "balanced_topk",
            {
                "max_iterations": 5,
                "pseudo_label_selection": "balanced_topk",
                "balanced_topk_per_class": 4,
                "target_prior_correction": "none",
            },
        ),
        SourceFreeConsensusVariant(
            "robust_prior",
            {
                "max_iterations": 5,
                "pseudo_label_selection": "balanced_topk",
                "balanced_topk_per_class": 4,
                "target_prior_correction": "balanced",
                "target_prior_strength": 0.75,
                "target_prior_estimator": "entropy_weighted",
                "target_prior_smoothing": 0.25,
                "target_prior_floor": 0.02,
            },
        ),
    )


def fit_source_free_consensus_predict_proba(
    *,
    source_model: Any,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    classes: Sequence[Any] | np.ndarray | None = None,
    variants: SourceFreeConsensusVariant | Mapping[str, Any] | str | Sequence[SourceFreeConsensusVariant | Mapping[str, Any] | str] | None = None,
    consensus_mode: ConsensusMode | str = "logit_mean",
    confidence_weight: float | str = 1.0,
    balance_weight: float | str = 1.0,
    weight_temperature: float | str = 1.0,
) -> SourceFreeConsensusResult:
    """Fit several source-free variants and combine their target probabilities.

    The consensus uses only a fitted source model and unlabeled target features.
    It does not accept target labels or source rows during target adaptation.
    """

    target_matrix = _feature_matrix(target_features, name="target_features")
    specs = _coerce_variants(variants)
    variant_results = tuple(
        fit_source_free_target_prior_predict_proba(
            source_model=source_model,
            target_features=target_matrix,
            classes=None if classes is None else np.asarray(classes, dtype=object).reshape(-1),
            **dict(spec.kwargs),
        )
        for spec in specs
    )
    variant_probabilities = tuple(result.probabilities for result in variant_results)
    weights = _resolve_variant_weights(
        specs,
        variant_probabilities,
        confidence_weight=confidence_weight,
        balance_weight=balance_weight,
        weight_temperature=weight_temperature,
    )
    probabilities = combine_probability_variants(
        variant_probabilities,
        weights=weights,
        mode=consensus_mode,
    )
    metadata = _consensus_metadata(
        specs,
        variant_probabilities,
        weights,
        mode=consensus_mode,
        confidence_weight=confidence_weight,
        balance_weight=balance_weight,
        weight_temperature=weight_temperature,
    )
    return SourceFreeConsensusResult(
        probabilities=probabilities,
        variant_probabilities=variant_probabilities,
        variant_results=variant_results,
        weights=weights,
        metadata=metadata,
    )


def combine_probability_variants(
    probability_variants: Sequence[np.ndarray],
    *,
    weights: Sequence[float] | np.ndarray | None = None,
    mode: ConsensusMode | str = "logit_mean",
) -> np.ndarray:
    """Combine same-shaped probability matrices by arithmetic or logit mean."""

    matrices = _probability_tensor(probability_variants)
    normalized_weights = _normalize_weights(
        np.ones(matrices.shape[0], dtype=float) if weights is None else np.asarray(weights, dtype=float),
        n_variants=matrices.shape[0],
    )
    parsed_mode = _consensus_mode(mode)
    if parsed_mode == "arithmetic_mean":
        combined = np.tensordot(normalized_weights, matrices, axes=(0, 0))
        return _normalize_probability_rows(combined)
    log_probabilities = np.log(np.clip(matrices, _EPS, 1.0))
    pooled_logits = np.tensordot(normalized_weights, log_probabilities, axes=(0, 0))
    return _softmax_rows(pooled_logits)


def estimate_consensus_variant_weights(
    probability_variants: Sequence[np.ndarray],
    *,
    confidence_weight: float | str = 1.0,
    balance_weight: float | str = 1.0,
    temperature: float | str = 1.0,
) -> np.ndarray:
    """Score variants from unlabeled confidence and marginal class balance."""

    matrices = _probability_tensor(probability_variants)
    confidence_scale = _nonnegative_float(confidence_weight, name="confidence_weight")
    balance_scale = _nonnegative_float(balance_weight, name="balance_weight")
    temp = _positive_float(temperature, name="weight_temperature")
    scores = []
    for probabilities in matrices:
        row_entropy = _row_entropy(probabilities).mean()
        confidence_score = 1.0 - row_entropy / _max_entropy(probabilities.shape[1])
        marginal_entropy = _entropy(probabilities.mean(axis=0)) / _max_entropy(probabilities.shape[1])
        score = confidence_scale * confidence_score + balance_scale * marginal_entropy
        scores.append(float(score))
    return _softmax_vector(np.asarray(scores, dtype=float) / temp)


def _coerce_variants(
    variants: SourceFreeConsensusVariant | Mapping[str, Any] | str | Sequence[SourceFreeConsensusVariant | Mapping[str, Any] | str] | None,
) -> tuple[SourceFreeConsensusVariant, ...]:
    if variants is None:
        return default_source_free_consensus_variants()
    if isinstance(variants, (SourceFreeConsensusVariant, Mapping, str)):
        variant_items = (variants,)
    else:
        try:
            variant_items = tuple(variants)
        except TypeError as exc:
            raise ValueError("variants must contain names, mappings, or SourceFreeConsensusVariant instances.") from exc
    specs: list[SourceFreeConsensusVariant] = []
    for index, variant in enumerate(variant_items):
        if isinstance(variant, SourceFreeConsensusVariant):
            specs.append(variant)
        elif isinstance(variant, str):
            specs.append(SourceFreeConsensusVariant(variant, _named_variant_kwargs(variant)))
        elif isinstance(variant, Mapping):
            name = str(variant.get("name", f"variant_{index}"))
            kwargs = variant.get("kwargs", {})
            if not isinstance(kwargs, Mapping):
                raise ValueError("variant kwargs must be a mapping.")
            weight = variant.get("weight")
            specs.append(SourceFreeConsensusVariant(name=name, kwargs=dict(kwargs), weight=None if weight is None else float(weight)))
        else:
            raise ValueError("variants must contain names, mappings, or SourceFreeConsensusVariant instances.")
    if not specs:
        raise ValueError("At least one source-free consensus variant is required.")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("source-free consensus variant names must be unique.")
    return tuple(specs)


def _named_variant_kwargs(name: str) -> Mapping[str, Any]:
    lookup = {spec.name: spec.kwargs for spec in default_source_free_consensus_variants()}
    if name not in lookup:
        raise ValueError(f"Unknown source-free consensus variant {name!r}.")
    return dict(lookup[name])


def _resolve_variant_weights(
    specs: Sequence[SourceFreeConsensusVariant],
    probability_variants: Sequence[np.ndarray],
    *,
    confidence_weight: float | str,
    balance_weight: float | str,
    weight_temperature: float | str,
) -> np.ndarray:
    specified = [spec.weight for spec in specs]
    if any(weight is not None for weight in specified):
        if any(weight is None for weight in specified):
            raise ValueError("Either specify all consensus variant weights or none of them.")
        return _normalize_weights(np.asarray(specified, dtype=float), n_variants=len(specs))
    return estimate_consensus_variant_weights(
        probability_variants,
        confidence_weight=confidence_weight,
        balance_weight=balance_weight,
        temperature=weight_temperature,
    )


def _consensus_metadata(
    specs: Sequence[SourceFreeConsensusVariant],
    probability_variants: Sequence[np.ndarray],
    weights: np.ndarray,
    *,
    mode: ConsensusMode | str,
    confidence_weight: float | str,
    balance_weight: float | str,
    weight_temperature: float | str,
) -> dict[str, Any]:
    matrices = _probability_tensor(probability_variants)
    return {
        "source_free_consensus": True,
        "source_free_consensus_protocol": SOURCE_FREE_CONSENSUS_PROTOCOL,
        "source_free_consensus_protocol_category": SOURCE_FREE_CONSENSUS_CATEGORY,
        "source_free_consensus_uses_pretrained_source_model": True,
        "source_free_consensus_uses_target_features": True,
        "source_free_consensus_uses_target_labels": False,
        "source_free_consensus_uses_source_rows_during_adaptation": False,
        "source_free_consensus_valid_for_protocol_2_5": True,
        "source_free_consensus_valid_for_benchmark": True,
        "source_free_consensus_mode": _consensus_mode(mode),
        "source_free_consensus_variants": "|".join(spec.name for spec in specs),
        "source_free_consensus_weights": _format_float_vector(weights),
        "source_free_consensus_confidence_weight": _nonnegative_float(confidence_weight, name="confidence_weight"),
        "source_free_consensus_balance_weight": _nonnegative_float(balance_weight, name="balance_weight"),
        "source_free_consensus_weight_temperature": _positive_float(weight_temperature, name="weight_temperature"),
        "source_free_consensus_n_variants": int(matrices.shape[0]),
        "source_free_consensus_n_target_rows": int(matrices.shape[1]),
        "source_free_consensus_n_classes": int(matrices.shape[2]),
    }


def _probability_tensor(probability_variants: Sequence[np.ndarray]) -> np.ndarray:
    if not probability_variants:
        raise ValueError("At least one probability matrix is required.")
    matrices = [_normalize_probability_rows(matrix) for matrix in probability_variants]
    first_shape = matrices[0].shape
    for matrix in matrices:
        if matrix.shape != first_shape:
            raise ValueError("All probability variants must have the same shape.")
    return np.stack(matrices, axis=0)


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty 2D matrix with at least two classes.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must contain positive mass.")
    return matrix / row_sums


def _normalize_weights(weights: np.ndarray, *, n_variants: int) -> np.ndarray:
    vector = np.asarray(weights, dtype=float).reshape(-1)
    if vector.shape[0] != int(n_variants):
        raise ValueError("weights must contain one entry per probability variant.")
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError("weights must be finite and non-negative.")
    if float(vector.sum()) <= 0.0:
        raise ValueError("weights must contain positive mass.")
    return vector / vector.sum()


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty 2D feature matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    return matrix


def _consensus_mode(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text in {"logit_mean", "geometric", "geometric_mean", "log_probability_mean"}:
        return "logit_mean"
    if text in {"arithmetic", "arithmetic_mean", "mean", "probability_mean"}:
        return "arithmetic_mean"
    raise ValueError("consensus_mode must be one of: logit_mean, arithmetic_mean.")


def _row_entropy(probabilities: np.ndarray) -> np.ndarray:
    return -np.sum(probabilities * np.log(np.clip(probabilities, _EPS, 1.0)), axis=1)


def _entropy(probabilities: np.ndarray) -> float:
    vector = np.asarray(probabilities, dtype=float).reshape(-1)
    vector = np.clip(vector / vector.sum(), _EPS, 1.0)
    return float(-np.sum(vector * np.log(vector)))


def _max_entropy(n_classes: int) -> float:
    return max(float(np.log(float(n_classes))), _EPS)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    matrix = np.asarray(logits, dtype=float)
    shifted = matrix - np.max(matrix, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _softmax_vector(logits: np.ndarray) -> np.ndarray:
    vector = np.asarray(logits, dtype=float).reshape(-1)
    shifted = vector - np.max(vector)
    exp_values = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_values / exp_values.sum()


def _nonnegative_float(value: float | str, *, name: str) -> float:
    parsed = _float(value, name=name)
    if parsed < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _float(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return parsed


def _float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _format_float_vector(values: Sequence[float] | np.ndarray) -> str:
    return "|".join(f"{float(value):.6g}" for value in np.asarray(values, dtype=float).reshape(-1))


__all__ = [
    "SOURCE_FREE_CONSENSUS_CATEGORY",
    "SOURCE_FREE_CONSENSUS_PROTOCOL",
    "SourceFreeConsensusResult",
    "SourceFreeConsensusVariant",
    "combine_probability_variants",
    "default_source_free_consensus_variants",
    "estimate_consensus_variant_weights",
    "fit_source_free_consensus_predict_proba",
]

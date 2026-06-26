from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from neureptrace.decoding.source_free import fit_source_free_predict_proba
from neureptrace.decoding.source_free_target_prior import apply_target_prior_correction, format_target_prior

_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class SourceFreeGridResult:
    probabilities: np.ndarray
    metadata: dict[str, Any]
    ranked: tuple[dict[str, Any], ...]


def fit_source_free_grid_predict_proba(
    *,
    source_model: Any,
    target_features: np.ndarray,
    classes: Any = None,
    confidence_thresholds=(0.75,),
    prototype_weights=(0.5,),
    prior_strengths=(0.0, 0.5, 1.0),
    pseudo_label_selections=("confidence", "balanced_topk"),
    balanced_topk_per_class_values=(None, 4),
    max_iterations: int = 5,
    min_class_count: int = 1,
    min_active_classes: int = 2,
) -> SourceFreeGridResult:
    """Pick a source-free variant with an unlabeled probability-shape score."""

    rows: list[dict[str, Any]] = []
    best_probabilities: np.ndarray | None = None
    best_metadata: dict[str, Any] | None = None
    for threshold, prototype_weight, prior_strength, selection, topk in product(
        tuple(confidence_thresholds), tuple(prototype_weights), tuple(prior_strengths), tuple(pseudo_label_selections), tuple(balanced_topk_per_class_values)
    ):
        if selection == "confidence" and topk is not None:
            continue
        result = fit_source_free_predict_proba(
            source_model=source_model,
            target_features=target_features,
            classes=classes,
            confidence_threshold=threshold,
            max_iterations=max_iterations,
            min_class_count=min_class_count,
            min_active_classes=min_active_classes,
            prototype_weight=prototype_weight,
            pseudo_label_selection=selection,
            balanced_topk_per_class=topk,
        )
        probabilities = result.probabilities
        prior_strength = _bounded_unit(prior_strength, "prior_strength")
        if prior_strength > 0.0:
            probabilities, prior = apply_target_prior_correction(probabilities, strength=prior_strength)
        else:
            prior = probabilities.mean(axis=0)
        score, terms = score_probability_shape(probabilities, active_classes=int(result.metadata.get("source_free_active_classes", 0)))
        metadata = {
            **result.metadata,
            "source_free_grid_selection": True,
            "source_free_grid_confidence_threshold": float(threshold),
            "source_free_grid_prototype_weight": float(prototype_weight),
            "source_free_grid_prior_strength": float(prior_strength),
            "source_free_grid_prior": format_target_prior(prior),
            "source_free_grid_selected_score": float(score),
        }
        row = {"score": float(score), "selection": selection, "topk": topk, "threshold": float(threshold), "prototype_weight": float(prototype_weight), "prior_strength": float(prior_strength), **terms}
        rows.append(row)
        if best_metadata is None or score > float(best_metadata["source_free_grid_selected_score"]):
            best_metadata = metadata
            best_probabilities = probabilities
    if best_probabilities is None or best_metadata is None:
        raise ValueError("source-free grid produced no candidates")
    ranked = tuple(sorted(rows, key=lambda row: row["score"], reverse=True))
    best_metadata = {**best_metadata, "source_free_grid_candidate_count": len(ranked)}
    return SourceFreeGridResult(probabilities=best_probabilities, metadata=best_metadata, ranked=ranked)


def score_probability_shape(probabilities: np.ndarray, *, active_classes: int = 0) -> tuple[float, dict[str, float]]:
    p = _normalize(probabilities)
    n_classes = p.shape[1]
    marginal_entropy = _entropy(p.mean(axis=0))
    row_entropy = float(np.mean([_entropy(row) for row in p]))
    confidence = float(np.mean(np.max(p, axis=1)))
    active_fraction = float(min(max(active_classes, 0), n_classes) / n_classes)
    score = marginal_entropy + 0.5 * active_fraction + 0.25 * confidence - 0.1 * row_entropy
    return float(score), {"marginal_entropy": marginal_entropy, "row_entropy": row_entropy, "confidence": confidence, "active_fraction": active_fraction}


def _normalize(probabilities: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional matrix with at least two classes")
    p = np.clip(p, 0.0, None)
    row_sums = p.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0) or not np.all(np.isfinite(p)):
        raise ValueError("probabilities must be finite with positive row mass")
    return p / row_sums


def _entropy(probabilities: np.ndarray) -> float:
    p = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), _EPS, None)
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)) / np.log(p.size))


def _bounded_unit(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be finite in [0, 1]")
    return number

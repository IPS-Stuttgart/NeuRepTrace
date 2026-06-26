"""Domain-classifier importance weighting for unlabeled target adaptation.

This module implements a compact covariate-shift weighting helper for
cross-subject M/EEG transfer.  A binary domain classifier is trained to separate
labeled source feature rows from unlabeled held-out target feature rows.  Its
source-row domain posteriors are converted into non-negative source sample weights.

The public API intentionally has no target-label argument.  This is a Category-2
protocol because unlabeled target features affect source training weights.
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

DOMAIN_IMPORTANCE_PROTOCOL = "unlabeled_target_domain_classifier_importance_weighting"
DOMAIN_IMPORTANCE_CATEGORY = "2_unlabeled_target_adaptive"
DEFAULT_WEIGHT_CLIP = (0.05, 20.0)
DEFAULT_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class DomainImportanceConfig:
    """Configuration for domain-classifier source weighting."""

    clip: tuple[float, float] | None = DEFAULT_WEIGHT_CLIP
    normalize: bool = True
    account_for_sample_priors: bool = True
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class DomainImportanceResult:
    """Source weights and domain probabilities from unlabeled target adaptation."""

    sample_weights: np.ndarray
    source_target_probabilities: np.ndarray
    target_target_probabilities: np.ndarray
    domain_classifier: BaseEstimator
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_domain_classifier_importance_weights(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    estimator: BaseEstimator | None = None,
    config: DomainImportanceConfig | Mapping[str, Any] | None = None,
) -> DomainImportanceResult:
    """Estimate source sample weights from an unlabeled target domain classifier.

    Parameters
    ----------
    source_features:
        Source feature rows that will later be used with source labels by a
        downstream classifier.
    target_features:
        Unlabeled held-out target rows used only to estimate source/target feature
        density ratios.
    estimator:
        Optional sklearn-compatible probabilistic binary classifier.  If omitted,
        a standardized logistic regression classifier is used.
    config:
        Weight clipping and normalization settings.  A mapping is normalized
        through :func:`domain_importance_config`.

    Returns
    -------
    DomainImportanceResult
        One non-negative sample weight per source row, plus protocol metadata.

    Notes
    -----
    The API intentionally has no ``target_labels`` parameter.  The returned source
    weights are Category-2 because they depend on unlabeled ``X_t``.
    """

    cfg = domain_importance_config() if config is None else _coerce_config(config)
    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"source_features and target_features must have the same feature width: {source.shape[1]} != {target.shape[1]}.")
    classifier = clone(_default_estimator() if estimator is None else estimator)
    domain_features = np.vstack([source, target])
    domain_labels = np.concatenate([np.zeros(source.shape[0], dtype=int), np.ones(target.shape[0], dtype=int)])
    classifier.fit(domain_features, domain_labels)
    source_probabilities = _target_domain_probabilities(classifier, source, epsilon=cfg.epsilon)
    target_probabilities = _target_domain_probabilities(classifier, target, epsilon=cfg.epsilon)
    weights = _posterior_to_importance_weights(
        source_probabilities,
        n_source=source.shape[0],
        n_target=target.shape[0],
        account_for_sample_priors=cfg.account_for_sample_priors,
        epsilon=cfg.epsilon,
    )
    if cfg.clip is not None:
        weights = np.clip(weights, cfg.clip[0], cfg.clip[1])
    if cfg.normalize:
        mean_weight = float(np.mean(weights))
        if mean_weight <= 0.0 or not np.isfinite(mean_weight):
            raise ValueError("Domain-importance weights have non-positive or non-finite mean.")
        weights = weights / mean_weight
    metadata = _metadata(
        cfg,
        n_source_rows=source.shape[0],
        n_target_rows=target.shape[0],
        feature_dim=source.shape[1],
        weights=weights,
        source_probabilities=source_probabilities,
        target_probabilities=target_probabilities,
        estimator_name=type(classifier).__name__,
    )
    return DomainImportanceResult(
        sample_weights=weights.astype(np.float32, copy=False),
        source_target_probabilities=source_probabilities.astype(np.float32, copy=False),
        target_target_probabilities=target_probabilities.astype(np.float32, copy=False),
        domain_classifier=classifier,
        metadata=metadata,
    )


def domain_importance_config(
    *,
    clip: Sequence[float] | str | None = DEFAULT_WEIGHT_CLIP,
    normalize: bool = True,
    account_for_sample_priors: bool = True,
    epsilon: float | str = DEFAULT_EPSILON,
) -> DomainImportanceConfig:
    """Normalize public domain-importance options."""

    return DomainImportanceConfig(
        clip=_normalize_clip(clip),
        normalize=bool(normalize),
        account_for_sample_priors=bool(account_for_sample_priors),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def apply_domain_importance_weights(
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    result: DomainImportanceResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source features, labels, and checked domain-importance weights."""

    features = _feature_matrix(source_features, name="source_features")
    labels = np.asarray(source_labels).reshape(-1)
    if labels.shape[0] != features.shape[0]:
        raise ValueError(f"source_labels must contain one value per source row: {labels.shape[0]} != {features.shape[0]}.")
    weights = np.asarray(result.sample_weights, dtype=float).reshape(-1)
    if weights.shape[0] != features.shape[0]:
        raise ValueError(f"result.sample_weights length does not match source_features rows: {weights.shape[0]} != {features.shape[0]}.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("Domain-importance weights must be finite and non-negative.")
    return features, labels, weights


def _target_domain_probabilities(model: BaseEstimator, features: np.ndarray, *, epsilon: float) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError("Domain importance weighting requires an estimator with predict_proba.")
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    classes = np.asarray(getattr(model, "classes_", [0, 1])).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[0] != features.shape[0]:
        raise ValueError("Domain classifier predict_proba returned an invalid probability matrix.")
    target_columns = np.flatnonzero(classes == 1)
    if target_columns.size != 1:
        raise ValueError("Domain classifier classes_ must contain target-domain label 1 exactly once.")
    target_probability = probabilities[:, int(target_columns[0])]
    return np.clip(target_probability, epsilon, 1.0 - epsilon)


def _posterior_to_importance_weights(
    target_probability: np.ndarray,
    *,
    n_source: int,
    n_target: int,
    account_for_sample_priors: bool,
    epsilon: float,
) -> np.ndarray:
    source_probability = np.clip(1.0 - target_probability, epsilon, 1.0)
    odds = target_probability / source_probability
    if account_for_sample_priors:
        odds = odds * (float(n_source) / float(n_target))
    return np.asarray(odds, dtype=float)


def _coerce_config(config: DomainImportanceConfig | Mapping[str, Any]) -> DomainImportanceConfig:
    if isinstance(config, DomainImportanceConfig):
        return config
    return domain_importance_config(**dict(config))


def _default_estimator() -> BaseEstimator:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=13))


def _metadata(
    cfg: DomainImportanceConfig,
    *,
    n_source_rows: int,
    n_target_rows: int,
    feature_dim: int,
    weights: np.ndarray,
    source_probabilities: np.ndarray,
    target_probabilities: np.ndarray,
    estimator_name: str,
) -> dict[str, Any]:
    return {
        "domain_importance_weighting": True,
        "domain_importance_protocol": DOMAIN_IMPORTANCE_PROTOCOL,
        "domain_importance_protocol_category": DOMAIN_IMPORTANCE_CATEGORY,
        "domain_importance_estimator": estimator_name,
        "domain_importance_uses_source_features": True,
        "domain_importance_uses_source_labels": False,
        "domain_importance_uses_target_features": True,
        "domain_importance_uses_target_labels": False,
        "domain_importance_valid_for_strict_source_only": False,
        "domain_importance_valid_for_unlabeled_target_adaptation": True,
        "domain_importance_valid_for_benchmark": False,
        "domain_importance_n_source_rows": int(n_source_rows),
        "domain_importance_n_target_rows": int(n_target_rows),
        "domain_importance_feature_dim": int(feature_dim),
        "domain_importance_clip_min": "" if cfg.clip is None else float(cfg.clip[0]),
        "domain_importance_clip_max": "" if cfg.clip is None else float(cfg.clip[1]),
        "domain_importance_normalize": bool(cfg.normalize),
        "domain_importance_account_for_sample_priors": bool(cfg.account_for_sample_priors),
        "domain_importance_epsilon": float(cfg.epsilon),
        "domain_importance_weight_min": float(np.min(weights)),
        "domain_importance_weight_max": float(np.max(weights)),
        "domain_importance_weight_mean": float(np.mean(weights)),
        "domain_importance_source_target_probability_mean": float(np.mean(source_probabilities)),
        "domain_importance_target_target_probability_mean": float(np.mean(target_probabilities)),
    }


def _normalize_clip(value: Sequence[float] | str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "none", "off", "false", "null"}:
            return None
        parts = [part.strip() for chunk in text.replace(";", ",").split(",") for part in chunk.split() if part.strip()]
        if len(parts) != 2:
            raise ValueError("clip must contain exactly two values, e.g. '0.05,20'.")
        lower, upper = (float(parts[0]), float(parts[1]))
    else:
        values = list(value)
        if len(values) != 2:
            raise ValueError("clip must contain exactly two values.")
        lower, upper = (float(values[0]), float(values[1]))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower < 0.0 or upper <= 0.0 or lower > upper:
        raise ValueError("clip bounds must be finite non-negative values with lower <= upper and upper > 0.")
    return lower, upper


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


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

"""Source-only random-subspace ensembles for M/EEG decoding.

This module implements a dependency-light random-subspace ensemble.  Each member
is trained on a source-label fold using a deterministic random subset of feature
columns, and held-out rows are only scored.  No target rows influence fitting,
feature selection, model weighting, or model selection.
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

RANDOM_SUBSPACE_PROTOCOL = "strict_source_only_random_subspace_ensemble"
RANDOM_SUBSPACE_CATEGORY = "1_strict_source_only"
DEFAULT_N_ESTIMATORS = 16
DEFAULT_FEATURE_FRACTION = 0.5
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class RandomSubspaceEnsembleConfig:
    """Configuration for a strict source-only random-subspace ensemble."""

    n_estimators: int = DEFAULT_N_ESTIMATORS
    feature_fraction: float = DEFAULT_FEATURE_FRACTION
    min_features: int = 1
    bootstrap_rows: bool = False
    row_fraction: float = 1.0
    random_state: int | None = 13
    epsilon: float = DEFAULT_EPSILON


@dataclass(frozen=True, slots=True)
class RandomSubspaceMember:
    """One fitted ensemble member and its selected rows/features."""

    estimator_index: int
    model: BaseEstimator
    feature_indices: np.ndarray
    row_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class RandomSubspaceEnsembleResult:
    """Predictions, probabilities, and provenance for the ensemble."""

    probabilities: np.ndarray
    predictions: np.ndarray
    classes: np.ndarray
    members: tuple[RandomSubspaceMember, ...]
    member_probabilities: tuple[np.ndarray, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_random_subspace_ensemble(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence[Any] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: RandomSubspaceEnsembleConfig | Mapping[str, Any] | None = None,
    estimator: BaseEstimator | None = None,
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> RandomSubspaceEnsembleResult:
    """Fit a strict source-only random-subspace ensemble and score test rows.

    Parameters
    ----------
    train_features, train_labels:
        Source/training rows and labels.  Feature subsets and optional bootstrap
        row samples are drawn only from these rows.
    test_features:
        Rows to score after the ensemble is fitted.  They are not used during
        fitting or feature-subspace selection.
    config:
        Ensemble settings.  A mapping is normalized through
        :func:`random_subspace_ensemble_config`.
    estimator:
        Optional sklearn-compatible classifier.  If omitted, a standardized
        logistic regression classifier is used.
    sample_weight:
        Optional source/training sample weights.  When bootstrapping rows, weights
        are subset to the sampled rows.

    Returns
    -------
    RandomSubspaceEnsembleResult
        Averaged probabilities, predictions, fitted members, and protocol
        metadata.
    """

    cfg = random_subspace_ensemble_config() if config is None else _coerce_config(config)
    train = _feature_matrix(train_features, name="train_features")
    test = _feature_matrix(test_features, name="test_features")
    if train.shape[1] != test.shape[1]:
        raise ValueError(f"train_features and test_features must have the same feature width: {train.shape[1]} != {test.shape[1]}.")
    labels = np.asarray(train_labels, dtype=object).reshape(-1)
    if labels.shape[0] != train.shape[0]:
        raise ValueError(f"train_labels must contain one value per train row: {labels.shape[0]} != {train.shape[0]}.")
    classes = np.asarray(tuple(dict.fromkeys(labels.tolist())), dtype=object)
    if classes.shape[0] < 2:
        raise ValueError("Random-subspace ensemble requires at least two classes.")
    class_to_code = {class_label: index for index, class_label in enumerate(classes.tolist())}
    label_codes = np.asarray([class_to_code[class_label] for class_label in labels.tolist()], dtype=int)
    encoded_classes = np.arange(classes.shape[0], dtype=int)
    weights = None if sample_weight is None else _sample_weight(sample_weight, expected_length=train.shape[0])
    rng = np.random.default_rng(cfg.random_state)
    feature_subspaces = sample_feature_subspaces(
        n_features=train.shape[1],
        n_estimators=cfg.n_estimators,
        feature_fraction=cfg.feature_fraction,
        min_features=cfg.min_features,
        random_state=cfg.random_state,
    )
    model_template = _default_estimator(cfg) if estimator is None else estimator
    members: list[RandomSubspaceMember] = []
    member_probabilities: list[np.ndarray] = []
    for estimator_index, feature_indices in enumerate(feature_subspaces):
        row_indices = _row_indices(
            n_rows=train.shape[0],
            row_fraction=cfg.row_fraction,
            bootstrap_rows=cfg.bootstrap_rows,
            rng=rng,
        )
        if np.unique(labels[row_indices]).shape[0] < 2:
            row_indices = np.arange(train.shape[0], dtype=int)
        model = clone(model_template)
        fit_kwargs = {} if weights is None else {"sample_weight": weights[row_indices]}
        try:
            model.fit(train[row_indices][:, feature_indices], label_codes[row_indices], **fit_kwargs)
        except TypeError:
            model.fit(train[row_indices][:, feature_indices], label_codes[row_indices])
        probabilities = _aligned_probabilities(model, test[:, feature_indices], classes=encoded_classes, epsilon=cfg.epsilon)
        members.append(
            RandomSubspaceMember(
                estimator_index=int(estimator_index),
                model=model,
                feature_indices=feature_indices.astype(int, copy=True),
                row_indices=row_indices.astype(int, copy=True),
            )
        )
        member_probabilities.append(probabilities)
    averaged = _normalize_probability_rows(np.mean(np.stack(member_probabilities, axis=0), axis=0), epsilon=cfg.epsilon)
    predictions = classes[np.argmax(averaged, axis=1)]
    metadata = _metadata(
        cfg,
        n_train_rows=train.shape[0],
        n_test_rows=test.shape[0],
        feature_dim=train.shape[1],
        n_classes=classes.shape[0],
        member_feature_counts=[member.feature_indices.shape[0] for member in members],
        member_row_counts=[member.row_indices.shape[0] for member in members],
    )
    return RandomSubspaceEnsembleResult(
        probabilities=averaged.astype(np.float32, copy=False),
        predictions=predictions,
        classes=classes,
        members=tuple(members),
        member_probabilities=tuple(probability.astype(np.float32, copy=False) for probability in member_probabilities),
        metadata=metadata,
    )


def random_subspace_ensemble_config(
    *,
    n_estimators: int | str = DEFAULT_N_ESTIMATORS,
    feature_fraction: float | str = DEFAULT_FEATURE_FRACTION,
    min_features: int | str = 1,
    bootstrap_rows: bool = False,
    row_fraction: float | str = 1.0,
    random_state: int | str | None = 13,
    epsilon: float | str = DEFAULT_EPSILON,
) -> RandomSubspaceEnsembleConfig:
    """Normalize public random-subspace ensemble options."""

    return RandomSubspaceEnsembleConfig(
        n_estimators=_positive_int(n_estimators, name="n_estimators"),
        feature_fraction=_unit_interval_float(feature_fraction, name="feature_fraction", include_zero=False),
        min_features=_positive_int(min_features, name="min_features"),
        bootstrap_rows=bool(bootstrap_rows),
        row_fraction=_unit_interval_float(row_fraction, name="row_fraction", include_zero=False),
        random_state=None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state"),
        epsilon=_positive_float(epsilon, name="epsilon"),
    )


def sample_feature_subspaces(
    *,
    n_features: int | str,
    n_estimators: int | str = DEFAULT_N_ESTIMATORS,
    feature_fraction: float | str = DEFAULT_FEATURE_FRACTION,
    min_features: int | str = 1,
    random_state: int | str | None = 13,
) -> tuple[np.ndarray, ...]:
    """Return deterministic random feature-index subsets."""

    total_features = _positive_int(n_features, name="n_features")
    n_members = _positive_int(n_estimators, name="n_estimators")
    fraction = _unit_interval_float(feature_fraction, name="feature_fraction", include_zero=False)
    minimum = min(_positive_int(min_features, name="min_features"), total_features)
    subset_size = min(total_features, max(minimum, int(round(total_features * fraction))))
    seed = None if random_state in {None, "", "none", "None"} else _nonnegative_int(random_state, name="random_state")
    rng = np.random.default_rng(seed)
    return tuple(np.sort(rng.choice(total_features, size=subset_size, replace=False)).astype(int, copy=False) for _ in range(n_members))


def _coerce_config(config: RandomSubspaceEnsembleConfig | Mapping[str, Any]) -> RandomSubspaceEnsembleConfig:
    if isinstance(config, RandomSubspaceEnsembleConfig):
        return config
    return random_subspace_ensemble_config(**dict(config))


def _row_indices(*, n_rows: int, row_fraction: float, bootstrap_rows: bool, rng: np.random.Generator) -> np.ndarray:
    if not bootstrap_rows and row_fraction >= 1.0:
        return np.arange(n_rows, dtype=int)
    n_selected = max(1, min(n_rows, int(round(n_rows * row_fraction))))
    return np.sort(rng.choice(n_rows, size=n_selected, replace=bootstrap_rows)).astype(int, copy=False)


def _aligned_probabilities(model: BaseEstimator, features: np.ndarray, *, classes: np.ndarray, epsilon: float) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(features), dtype=float)
        model_classes = np.asarray(getattr(model, "classes_", classes), dtype=object)
        aligned = np.full((features.shape[0], classes.shape[0]), epsilon, dtype=float)
        class_to_column = {class_label: index for index, class_label in enumerate(classes.tolist())}
        for source_column, class_label in enumerate(model_classes.tolist()):
            if class_label in class_to_column:
                aligned[:, class_to_column[class_label]] = raw[:, source_column]
        return _normalize_probability_rows(aligned, epsilon=epsilon)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        return _normalize_probability_rows(np.exp(np.clip(shifted, -50.0, 50.0)), epsilon=epsilon)
    predictions = np.asarray(model.predict(features), dtype=object)
    output = np.full((features.shape[0], classes.shape[0]), epsilon, dtype=float)
    class_to_column = {class_label: index for index, class_label in enumerate(classes.tolist())}
    for row, label in enumerate(predictions.tolist()):
        if label in class_to_column:
            output[row, class_to_column[label]] = 1.0
    return _normalize_probability_rows(output, epsilon=epsilon)


def _default_estimator(_cfg: RandomSubspaceEnsembleConfig) -> BaseEstimator:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=13))


def _metadata(
    cfg: RandomSubspaceEnsembleConfig,
    *,
    n_train_rows: int,
    n_test_rows: int,
    feature_dim: int,
    n_classes: int,
    member_feature_counts: Sequence[int],
    member_row_counts: Sequence[int],
) -> dict[str, Any]:
    return {
        "random_subspace_ensemble": True,
        "random_subspace_protocol": RANDOM_SUBSPACE_PROTOCOL,
        "random_subspace_protocol_category": RANDOM_SUBSPACE_CATEGORY,
        "random_subspace_uses_train_features": True,
        "random_subspace_uses_train_labels": True,
        "random_subspace_uses_test_features_for_fit": False,
        "random_subspace_uses_test_labels": False,
        "random_subspace_valid_for_strict_source_only": True,
        "random_subspace_valid_for_unlabeled_target_adaptation": True,
        "random_subspace_valid_for_benchmark": True,
        "random_subspace_n_train_rows": int(n_train_rows),
        "random_subspace_n_test_rows": int(n_test_rows),
        "random_subspace_feature_dim": int(feature_dim),
        "random_subspace_n_classes": int(n_classes),
        "random_subspace_n_estimators": int(cfg.n_estimators),
        "random_subspace_feature_fraction": float(cfg.feature_fraction),
        "random_subspace_min_features": int(cfg.min_features),
        "random_subspace_bootstrap_rows": bool(cfg.bootstrap_rows),
        "random_subspace_row_fraction": float(cfg.row_fraction),
        "random_subspace_random_state": "" if cfg.random_state is None else int(cfg.random_state),
        "random_subspace_mean_selected_features": float(np.mean(member_feature_counts)),
        "random_subspace_mean_selected_rows": float(np.mean(member_row_counts)),
    }


def _normalize_probability_rows(probabilities: np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("probabilities must be a finite two-dimensional matrix.")
    matrix = np.maximum(matrix, float(epsilon))
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return matrix / row_sums


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _sample_weight(values: Sequence[float] | np.ndarray, *, expected_length: int) -> np.ndarray:
    weights = np.asarray(values, dtype=float).reshape(-1)
    if weights.shape[0] != expected_length:
        raise ValueError(f"sample_weight must contain one value per train row: {weights.shape[0]} != {expected_length}.")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("sample_weight must contain finite non-negative values.")
    return weights


def _positive_int(value: int | str, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _nonnegative_int(value: int | str, *, name: str) -> int:
    integer = _integer(value, name=name)
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return integer


def _integer(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    parsed = _float_value(value, name=name)
    if parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed


def _unit_interval_float(value: float | str, *, name: str, include_zero: bool) -> float:
    parsed = _float_value(value, name=name)
    lower_ok = parsed >= 0.0 if include_zero else parsed > 0.0
    if not lower_ok or parsed > 1.0:
        bracket = "[0, 1]" if include_zero else "(0, 1]"
        raise ValueError(f"{name} must be in {bracket}.")
    return parsed


def _float_value(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed

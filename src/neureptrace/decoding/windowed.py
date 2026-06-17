from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score

from neureptrace.decoding.classifiers import prediction_scores


@dataclass(frozen=True)
class WindowedModelBundle:
    """Fitted model plus label-independent feature transform metadata."""

    model: Any
    train_window: tuple[float, float] | None
    train_labels: np.ndarray
    pca_coeff: np.ndarray | None
    train_features_mean: np.ndarray | None
    explained_variance_percent: float
    actual_components_pca: int


@dataclass(frozen=True)
class WindowedDecodingResult:
    """Predictions and null scores for one train/validation feature window."""

    model_bundle: WindowedModelBundle
    predictions: np.ndarray
    scores: np.ndarray
    accuracy: float
    permutation_accuracy: np.ndarray
    permutation_p_value: float
    balanced_accuracy: float = np.nan
    permutation_balanced_accuracy: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    balanced_accuracy_p_value: float = np.nan


FitModel = Callable[[np.ndarray, np.ndarray], Any]


def fit_window_model(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence | np.ndarray,
    *,
    fit_model: FitModel,
    components_pca: int | float | str | None = float("inf"),
    train_window: tuple[float, float] | None = None,
) -> WindowedModelBundle:
    """Fit one model for a precomputed feature window.

    Dataset-specific projects provide the windowed feature matrix and a model
    factory. NeuRepTrace owns the reusable PCA fit/transform bookkeeping.
    ``components_pca`` accepts a positive integer component cap, a float in
    ``(0, 1)`` interpreted as an explained-variance ratio, or ``None``/infinity
    to disable PCA.
    """

    train_features = _feature_matrix(train_features, name="train_features")
    train_labels = _label_vector(train_labels, expected_length=train_features.shape[0], name="train_labels")
    transformed_features, pca_coeff, feature_mean, explained_variance, actual_components = _fit_pca_transform(
        train_features,
        components_pca,
    )
    model = fit_model(transformed_features, train_labels)
    return WindowedModelBundle(
        model=model,
        train_window=train_window,
        train_labels=train_labels,
        pca_coeff=pca_coeff,
        train_features_mean=feature_mean,
        explained_variance_percent=explained_variance,
        actual_components_pca=actual_components,
    )


def transform_window_features(
    model_bundle: WindowedModelBundle,
    features: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Apply the fitted PCA transform from a windowed model bundle."""

    features = _feature_matrix(features, name="features")
    if model_bundle.pca_coeff is None:
        return features
    if model_bundle.train_features_mean is None:
        raise ValueError("PCA coefficients require train_features_mean.")
    return (features - model_bundle.train_features_mean) @ model_bundle.pca_coeff[:, : model_bundle.actual_components_pca]


def predict_window_model(
    model_bundle: WindowedModelBundle,
    features: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict labels and confidence-like scores for a precomputed feature window."""

    transformed_features = transform_window_features(model_bundle, features)
    predictions = np.asarray(model_bundle.model.predict(transformed_features))
    scores = prediction_scores(model_bundle.model, transformed_features)
    return predictions, scores


def score_windowed_decoding(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence | np.ndarray,
    validation_features: Sequence[Sequence[float]] | np.ndarray,
    validation_labels: Sequence | np.ndarray,
    *,
    fit_model: FitModel,
    components_pca: int | float | str | None = float("inf"),
    train_window: tuple[float, float] | None = None,
    n_permutations: int = 0,
    permutation_rng: np.random.Generator | None = None,
) -> WindowedDecodingResult:
    """Fit, predict, score accuracy, and optionally compute shuffled-label null scores."""

    n_permutations = _validate_permutation_count(n_permutations)
    train_features = _feature_matrix(train_features, name="train_features")
    train_labels = _label_vector(train_labels, expected_length=train_features.shape[0], name="train_labels")
    validation_features = _feature_matrix(validation_features, name="validation_features")
    validation_labels = _label_vector(
        validation_labels,
        expected_length=validation_features.shape[0],
        name="validation_labels",
    )
    model_bundle = fit_window_model(
        train_features,
        train_labels,
        fit_model=fit_model,
        components_pca=components_pca,
        train_window=train_window,
    )
    predictions, scores = predict_window_model(model_bundle, validation_features)
    accuracy = float(np.mean(predictions == validation_labels)) if len(validation_labels) else np.nan
    balanced_accuracy = _balanced_accuracy(predictions, validation_labels)

    permutation_accuracy = np.array([], dtype=float)
    permutation_balanced_accuracy = np.array([], dtype=float)
    permutation_p_value = np.nan
    balanced_accuracy_p_value = np.nan
    if n_permutations > 0:
        transformed_train = transform_window_features(model_bundle, train_features)
        transformed_validation = transform_window_features(model_bundle, validation_features)
        permutation_accuracy, permutation_balanced_accuracy = permutation_score_curves(
            transformed_train,
            validation_features=transformed_validation,
            validation_labels=validation_labels,
            train_labels=train_labels,
            fit_model=fit_model,
            n_permutations=n_permutations,
            permutation_rng=permutation_rng,
        )
        permutation_p_value = permutation_p_from_accuracy(
            accuracy,
            permutation_accuracy,
        )
        balanced_accuracy_p_value = permutation_p_from_accuracy(
            balanced_accuracy,
            permutation_balanced_accuracy,
        )

    return WindowedDecodingResult(
        model_bundle=model_bundle,
        predictions=predictions,
        scores=scores,
        accuracy=accuracy,
        permutation_accuracy=permutation_accuracy,
        permutation_p_value=permutation_p_value,
        balanced_accuracy=balanced_accuracy,
        permutation_balanced_accuracy=permutation_balanced_accuracy,
        balanced_accuracy_p_value=balanced_accuracy_p_value,
    )


def permutation_accuracy_curve(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    validation_features: Sequence[Sequence[float]] | np.ndarray,
    validation_labels: Sequence | np.ndarray,
    train_labels: Sequence | np.ndarray,
    fit_model: FitModel,
    n_permutations: int,
    permutation_rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Train shuffled-label models and return validation accuracies."""

    accuracy, _ = permutation_score_curves(
        train_features,
        validation_features=validation_features,
        validation_labels=validation_labels,
        train_labels=train_labels,
        fit_model=fit_model,
        n_permutations=n_permutations,
        permutation_rng=permutation_rng,
    )
    return accuracy


def permutation_score_curves(
    train_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    validation_features: Sequence[Sequence[float]] | np.ndarray,
    validation_labels: Sequence | np.ndarray,
    train_labels: Sequence | np.ndarray,
    fit_model: FitModel,
    n_permutations: int,
    permutation_rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Train shuffled-label models and return accuracy and balanced-accuracy null curves."""

    n_permutations = _validate_permutation_count(n_permutations)
    train_features = _feature_matrix(train_features, name="train_features")
    validation_features = _feature_matrix(validation_features, name="validation_features")
    train_labels = _label_vector(train_labels, expected_length=train_features.shape[0], name="train_labels")
    validation_labels = _label_vector(
        validation_labels,
        expected_length=validation_features.shape[0],
        name="validation_labels",
    )
    if permutation_rng is None:
        permutation_rng = np.random.default_rng()

    permuted_accuracy = []
    permuted_balanced_accuracy = []
    for _ in range(n_permutations):
        permuted_train_labels = np.array(train_labels, copy=True)
        permutation_rng.shuffle(permuted_train_labels)
        model = fit_model(train_features, permuted_train_labels)
        predictions = np.asarray(model.predict(validation_features))
        permuted_accuracy.append(float(np.mean(predictions == validation_labels)))
        permuted_balanced_accuracy.append(_balanced_accuracy(predictions, validation_labels))
    return np.asarray(permuted_accuracy, dtype=float), np.asarray(permuted_balanced_accuracy, dtype=float)


def permutation_p_from_accuracy(accuracy: float, permutation_accuracy: Sequence[float] | np.ndarray) -> float:
    """Return the one-sided permutation p-value with plus-one correction."""

    permutation_accuracy = np.asarray(permutation_accuracy, dtype=float)
    if permutation_accuracy.size == 0 or not np.isfinite(accuracy):
        return np.nan
    return float((np.sum(permutation_accuracy >= accuracy) + 1.0) / (permutation_accuracy.size + 1.0))


def _validate_permutation_count(n_permutations: int) -> int:
    if isinstance(n_permutations, (bool, np.bool_)):
        raise ValueError("n_permutations must be a non-negative integer.")
    try:
        numeric = float(n_permutations)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_permutations must be a non-negative integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 0.0:
        raise ValueError("n_permutations must be a non-negative integer.")
    return int(numeric)


def _balanced_accuracy(predictions: Sequence | np.ndarray, labels: Sequence | np.ndarray) -> float:
    labels = np.asarray(labels).ravel()
    predictions = np.asarray(predictions).ravel()
    if labels.size == 0:
        return np.nan
    return float(balanced_accuracy_score(labels, predictions))


def _fit_pca_transform(
    features: np.ndarray,
    components_pca: int | float | str | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float, int]:
    normalized_components = _normalize_pca_components(components_pca, features)
    if normalized_components is None:
        return features, None, None, np.nan, int(features.shape[1])

    feature_mean = np.mean(features, axis=0)
    centered = features - feature_mean
    pca = PCA(n_components=normalized_components)
    transformed = pca.fit_transform(centered)
    explained_variance = float(np.sum(pca.explained_variance_ratio_) * 100.0)
    actual_components = int(getattr(pca, "n_components_", transformed.shape[1]))
    return transformed, pca.components_.T, feature_mean, explained_variance, actual_components


def _actual_pca_components(components_pca: int | float | str | None, features: np.ndarray) -> int:
    normalized_components = _normalize_pca_components(components_pca, features)
    if normalized_components is None:
        return int(features.shape[1])
    if isinstance(normalized_components, float) and not normalized_components.is_integer():
        return _max_pca_components(features)
    return int(normalized_components)


def _normalize_pca_components(components_pca: int | float | str | None, features: np.ndarray) -> int | float | None:
    """Normalize PCA component configuration before constructing ``sklearn`` PCA."""

    if components_pca is None:
        return None
    if isinstance(components_pca, str):
        normalized = components_pca.strip().lower()
        if normalized in {"", "all", "inf", "infinity", "none"}:
            return None
        try:
            value = float(normalized)
        except ValueError as exc:
            raise ValueError(_pca_components_error_message()) from exc
    else:
        try:
            value = float(components_pca)
        except (TypeError, ValueError) as exc:
            raise ValueError(_pca_components_error_message()) from exc

    if np.isposinf(value):
        return None
    if not np.isfinite(value):
        raise ValueError(_pca_components_error_message())
    if 0.0 < value < 1.0:
        return value
    if value.is_integer() and value >= 1.0:
        return min(int(value), _max_pca_components(features))
    raise ValueError(_pca_components_error_message())


def _max_pca_components(features: np.ndarray) -> int:
    return min(int(features.shape[0]), int(features.shape[1]))


def _pca_components_error_message() -> str:
    return "components_pca must be a positive integer count, a float in (0, 1) explained-variance ratio, or None/infinity to disable PCA."


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column.")
    return matrix


def _label_vector(labels: Sequence | np.ndarray, *, expected_length: int, name: str) -> np.ndarray:
    vector = np.asarray(labels).ravel()
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match feature rows: {len(vector)} != {expected_length}.")
    return vector

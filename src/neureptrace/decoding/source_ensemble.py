"""Source-domain probability ensembles for cross-subject decoding."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCE_DOMAIN_ENSEMBLE_PROTOCOL = "source_domain_probability_ensemble"
SOURCE_DOMAIN_ENSEMBLE_CATEGORY_1 = "1_strict_source_only"
SOURCE_DOMAIN_ENSEMBLE_CATEGORY_2 = "2_unlabeled_target_adaptive"
ENSEMBLE_WEIGHTING_MODES = ("uniform", "target_confidence", "target_entropy", "target_feature_similarity")
DEFAULT_TEMPERATURE = 1.0
DEFAULT_EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class SourceDomainModel:
    """One fitted source-domain classifier and its provenance."""

    domain_id: Hashable
    model: BaseEstimator
    n_rows: int
    classes: np.ndarray


@dataclass(frozen=True, slots=True)
class SourceDomainEnsembleResult:
    """Target probabilities from a source-domain classifier ensemble."""

    probabilities: np.ndarray
    predictions: np.ndarray
    classes: np.ndarray
    domain_weights: Mapping[Hashable, float]
    domain_probabilities: Mapping[Hashable, np.ndarray]
    models: Mapping[Hashable, SourceDomainModel]
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_source_domain_probability_ensemble(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    source_domains: Sequence[Hashable] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    estimator: BaseEstimator | None = None,
    weighting: str = "uniform",
    temperature: float | str = DEFAULT_TEMPERATURE,
    min_classes_per_domain: int | str = 2,
    epsilon: float | str = DEFAULT_EPSILON,
) -> SourceDomainEnsembleResult:
    """Fit per-source-domain classifiers and ensemble their target probabilities."""

    source = _feature_matrix(source_features, name="source_features")
    target = _feature_matrix(target_features, name="target_features")
    if source.shape[1] != target.shape[1]:
        raise ValueError(f"source_features and target_features must have the same feature width: {source.shape[1]} != {target.shape[1]}.")

    labels = _label_vector(source_labels, name="source_labels")
    if labels.shape[0] != source.shape[0]:
        raise ValueError(f"source_labels must contain one value per source row: {labels.shape[0]} != {source.shape[0]}.")
    domains = _domain_vector(source_domains, expected_length=source.shape[0])
    classes = _unique_label_vector(labels)
    if classes.shape[0] < 2:
        raise ValueError("At least two source classes are required.")

    class_indices = {_label_key(label): index for index, label in enumerate(classes.tolist())}
    encoded_labels = np.asarray([class_indices[_label_key(label)] for label in labels.tolist()], dtype=int)
    mode = normalize_ensemble_weighting(weighting)
    temp = _positive_float(temperature, name="temperature")
    eps = _positive_float(epsilon, name="epsilon")
    min_classes = _positive_int(min_classes_per_domain, name="min_classes_per_domain")
    model_template = _default_estimator() if estimator is None else estimator

    models: dict[Hashable, SourceDomainModel] = {}
    domain_probabilities: dict[Hashable, np.ndarray] = {}
    for domain in tuple(dict.fromkeys(domains.tolist())):
        mask = domains == domain
        domain_classes = _unique_label_vector(labels[mask])
        if domain_classes.shape[0] < min_classes:
            continue
        model = clone(model_template)
        model.fit(source[mask], encoded_labels[mask])
        probabilities = _aligned_probabilities(model, target, classes=np.arange(classes.shape[0]), epsilon=eps)
        models[domain] = SourceDomainModel(domain_id=domain, model=model, n_rows=int(np.sum(mask)), classes=domain_classes)
        domain_probabilities[domain] = probabilities
    if not models:
        raise ValueError("No source domain had enough classes to train a domain classifier.")

    weights = _domain_weights(mode, domain_probabilities, source, domains, target, temperature=temp, epsilon=eps)
    probabilities = np.zeros((target.shape[0], classes.shape[0]), dtype=float)
    for domain, domain_probability in domain_probabilities.items():
        probabilities += weights[domain] * domain_probability
    probabilities = _normalize_probability_rows(probabilities, epsilon=eps)
    predictions = classes[np.argmax(probabilities, axis=1)]
    return SourceDomainEnsembleResult(
        probabilities=probabilities.astype(np.float32, copy=False),
        predictions=predictions,
        classes=classes,
        domain_weights=weights,
        domain_probabilities={domain: values.astype(np.float32, copy=False) for domain, values in domain_probabilities.items()},
        models=models,
        metadata=_metadata(
            mode=mode,
            n_source_rows=source.shape[0],
            n_target_rows=target.shape[0],
            feature_dim=source.shape[1],
            n_classes=classes.shape[0],
            n_source_domains=len(tuple(dict.fromkeys(domains.tolist()))),
            n_trained_domains=len(models),
            weights=weights,
            temperature=temp,
            min_classes=min_classes,
        ),
    )


def normalize_ensemble_weighting(value: str | None) -> str:
    """Normalize public source-domain ensemble weighting aliases."""

    normalized = "uniform" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "equal": "uniform",
        "mean": "uniform",
        "confidence": "target_confidence",
        "target_conf": "target_confidence",
        "low_entropy": "target_entropy",
        "entropy": "target_entropy",
        "feature_similarity": "target_feature_similarity",
        "target_similarity": "target_feature_similarity",
        "mean_covariance": "target_feature_similarity",
    }.get(normalized, normalized)
    if normalized not in ENSEMBLE_WEIGHTING_MODES:
        raise ValueError(f"Unknown source-domain ensemble weighting {value!r}. Available modes: {', '.join(ENSEMBLE_WEIGHTING_MODES)}.")
    return normalized


def _domain_weights(
    mode: str,
    domain_probabilities: Mapping[Hashable, np.ndarray],
    source: np.ndarray,
    domains: np.ndarray,
    target: np.ndarray,
    *,
    temperature: float,
    epsilon: float,
) -> dict[Hashable, float]:
    domain_ids = tuple(domain_probabilities)
    if mode == "uniform":
        return {domain: 1.0 / len(domain_ids) for domain in domain_ids}
    if mode == "target_confidence":
        scores = np.asarray([np.mean(np.max(domain_probabilities[domain], axis=1)) for domain in domain_ids], dtype=float)
        return _softmax_scores(domain_ids, scores, temperature=temperature)
    if mode == "target_entropy":
        scores = []
        for domain in domain_ids:
            probabilities = np.maximum(domain_probabilities[domain], epsilon)
            scores.append(-float(np.mean(-np.sum(probabilities * np.log(probabilities), axis=1))))
        return _softmax_scores(domain_ids, np.asarray(scores, dtype=float), temperature=temperature)
    if mode == "target_feature_similarity":
        distances = np.asarray([_feature_distribution_distance(source[domains == domain], target) for domain in domain_ids], dtype=float)
        return _softmax_scores(domain_ids, -distances, temperature=temperature)
    raise ValueError(f"Unhandled ensemble weighting mode {mode!r}.")


def _softmax_scores(domain_ids: Sequence[Hashable], scores: np.ndarray, *, temperature: float) -> dict[Hashable, float]:
    if scores.shape[0] != len(domain_ids):
        raise ValueError("scores must contain one value per domain.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Domain weighting scores must be finite.")
    weights = np.exp(np.clip((scores - np.max(scores)) / float(temperature), -50.0, 50.0))
    total = float(np.sum(weights))
    if total <= 0.0:
        return {domain: 1.0 / len(domain_ids) for domain in domain_ids}
    return {domain: float(weight) for domain, weight in zip(domain_ids, weights / total, strict=True)}


def _feature_distribution_distance(source_domain: np.ndarray, target: np.ndarray) -> float:
    mean_distance = float(np.linalg.norm(np.mean(source_domain, axis=0) - np.mean(target, axis=0)) / np.sqrt(source_domain.shape[1]))
    covariance_distance = float(np.linalg.norm(_covariance(source_domain) - _covariance(target), ord="fro") / max(1, source_domain.shape[1]))
    return mean_distance + covariance_distance


def _covariance(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 1:
        return np.zeros((matrix.shape[1], matrix.shape[1]), dtype=float)
    centered = matrix - np.mean(matrix, axis=0, keepdims=True)
    return centered.T @ centered / float(matrix.shape[0] - 1)


def _aligned_probabilities(model: BaseEstimator, features: np.ndarray, *, classes: np.ndarray, epsilon: float) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(features), dtype=float)
        model_classes = np.asarray(getattr(model, "classes_", classes), dtype=object)
    elif hasattr(model, "decision_function"):
        raw, model_classes = _decision_probabilities(model, features, classes)
    else:
        predictions = np.asarray(model.predict(features), dtype=object)
        output = np.full((features.shape[0], classes.shape[0]), epsilon, dtype=float)
        class_to_column = {class_label: index for index, class_label in enumerate(classes.tolist())}
        for row, label in enumerate(predictions.tolist()):
            if label in class_to_column:
                output[row, class_to_column[label]] = 1.0
        return _normalize_probability_rows(output, epsilon=epsilon)

    if raw.shape[0] != features.shape[0]:
        raise ValueError("probability output must contain one row per feature row.")
    if raw.shape[1] != model_classes.shape[0]:
        raise ValueError("probability output width must match model.classes_.")
    aligned = np.full((features.shape[0], classes.shape[0]), epsilon, dtype=float)
    class_to_column = {class_label: index for index, class_label in enumerate(classes.tolist())}
    for source_column, class_label in enumerate(model_classes.tolist()):
        if class_label in class_to_column:
            aligned[:, class_to_column[class_label]] = raw[:, source_column]
    return _normalize_probability_rows(aligned, epsilon=epsilon)


def _decision_probabilities(model: BaseEstimator, features: np.ndarray, classes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(model.decision_function(features), dtype=float)
    model_classes = np.asarray(getattr(model, "classes_", ()), dtype=object)
    if scores.ndim == 1:
        if model_classes.size == 0:
            if classes.shape[0] != 2:
                raise ValueError("Binary decision_function alignment requires model.classes_ when the global class set is not binary.")
            model_classes = classes
        if model_classes.shape[0] != 2:
            raise ValueError("One-dimensional decision_function output requires exactly two model classes.")
        scores = np.column_stack([-scores, scores])
    elif scores.ndim == 2:
        if model_classes.size == 0:
            if scores.shape[1] != classes.shape[0]:
                raise ValueError("Multiclass decision_function alignment requires model.classes_ when output width differs from the global class count.")
            model_classes = classes
    else:
        raise ValueError("decision_function output must be one- or two-dimensional.")
    if scores.shape[0] != features.shape[0]:
        raise ValueError("decision_function output must contain one row per feature row.")
    if scores.shape[1] != model_classes.shape[0]:
        raise ValueError("decision_function output width must match model.classes_.")
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    return np.exp(np.clip(shifted, -50.0, 50.0)), model_classes


def _normalize_probability_rows(probabilities: np.ndarray, *, epsilon: float) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("probabilities must be a finite two-dimensional matrix.")
    matrix = np.maximum(matrix, float(epsilon))
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return matrix / row_sums


def _default_estimator() -> BaseEstimator:
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=13))


def _metadata(
    *,
    mode: str,
    n_source_rows: int,
    n_target_rows: int,
    feature_dim: int,
    n_classes: int,
    n_source_domains: int,
    n_trained_domains: int,
    weights: Mapping[Hashable, float],
    temperature: float,
    min_classes: int,
) -> dict[str, Any]:
    adaptive = mode != "uniform"
    return {
        "source_domain_ensemble": True,
        "source_domain_ensemble_protocol": SOURCE_DOMAIN_ENSEMBLE_PROTOCOL,
        "source_domain_ensemble_protocol_category": SOURCE_DOMAIN_ENSEMBLE_CATEGORY_2 if adaptive else SOURCE_DOMAIN_ENSEMBLE_CATEGORY_1,
        "source_domain_ensemble_weighting": mode,
        "source_domain_ensemble_uses_source_features": True,
        "source_domain_ensemble_uses_source_labels": True,
        "source_domain_ensemble_uses_source_domains": True,
        "source_domain_ensemble_uses_target_features_for_weighting": bool(adaptive),
        "source_domain_ensemble_uses_target_labels": False,
        "source_domain_ensemble_valid_for_strict_source_only": not adaptive,
        "source_domain_ensemble_valid_for_unlabeled_target_adaptation": True,
        "source_domain_ensemble_valid_for_benchmark": not adaptive,
        "source_domain_ensemble_n_source_rows": int(n_source_rows),
        "source_domain_ensemble_n_target_rows": int(n_target_rows),
        "source_domain_ensemble_feature_dim": int(feature_dim),
        "source_domain_ensemble_n_classes": int(n_classes),
        "source_domain_ensemble_n_source_domains": int(n_source_domains),
        "source_domain_ensemble_n_trained_domains": int(n_trained_domains),
        "source_domain_ensemble_temperature": float(temperature),
        "source_domain_ensemble_min_classes_per_domain": int(min_classes),
        "source_domain_ensemble_domain_weights": "|".join(f"{domain}:{float(weight):.12g}" for domain, weight in weights.items()),
    }


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _label_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            raise ValueError(f"{name} must contain one value per source row.")
        items = values.tolist() if values.ndim == 1 else [_sequence_label_from_row(row) for row in values.reshape(values.shape[0], -1)]
    else:
        try:
            items = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must contain one value per source row.") from exc
    vector = np.empty(len(items), dtype=object)
    vector[:] = items
    return vector


def _sequence_label_from_row(row: np.ndarray) -> Any:
    if row.size == 1:
        value = row[0]
        return value.item() if isinstance(value, np.generic) else value
    return tuple(value.item() if isinstance(value, np.generic) else value for value in row.tolist())


def _unique_label_vector(labels: np.ndarray) -> np.ndarray:
    unique: list[Any] = []
    seen: set[tuple[Any, ...]] = set()
    for label in labels.tolist():
        key = _label_key(label)
        if key not in seen:
            seen.add(key)
            unique.append(label)
    vector = np.empty(len(unique), dtype=object)
    vector[:] = unique
    return vector


def _label_key(label: Any) -> tuple[Any, ...]:
    if isinstance(label, np.generic):
        return _label_key(label.item())
    if isinstance(label, np.ndarray):
        return ("sequence", tuple(_label_key(value) for value in label.reshape(-1).tolist()))
    if isinstance(label, (list, tuple)):
        return ("sequence", tuple(_label_key(value) for value in label))
    if isinstance(label, float) and np.isnan(label):
        return ("nan",)
    if isinstance(label, bytes):
        return ("bytes", label)
    if isinstance(label, str):
        return ("str", label)
    if isinstance(label, (bool, np.bool_)):
        return ("bool", bool(label))
    try:
        hash(label)
    except TypeError:
        return ("repr", type(label).__module__, type(label).__qualname__, repr(label))
    return ("scalar", label)


def _domain_vector(values: Sequence[Hashable] | np.ndarray, *, expected_length: int) -> np.ndarray:
    vector = _label_vector(values, name="source_domains")
    if vector.shape[0] != expected_length:
        raise ValueError(f"source_domains must contain one value per source row: {vector.shape[0]} != {expected_length}.")
    for value in vector.tolist():
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"source_domains must be hashable; got {value!r}.") from exc
    return vector


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


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

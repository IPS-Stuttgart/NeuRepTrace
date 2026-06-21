from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline


SOURCE_FREE_ADAPTATION_PROTOCOL = "source_free_unlabeled_target_adaptation"
_EPS = 1e-12


@dataclass(frozen=True, slots=True)
class SourceFreeAdaptationResult:
    """Batch-level source-free adaptation result.

    ``adapter`` contains the fitted target-batch adapter, ``probabilities`` are
    the adapted probabilities for the target features passed to ``fit``, and
    ``metadata`` records the protocol hygiene: target features are used, target
    labels and source samples are not used during the adaptation step.
    """

    adapter: "SourceFreeSubjectAdapter"
    probabilities: np.ndarray
    metadata: dict[str, Any]


class SourceFreeSubjectAdapter(BaseEstimator):
    """Adapt a fitted source model to an unlabeled target subject without source samples.

    The adapter implements a lightweight SHOT-style source-free protocol for
    NeuRepTrace experiments: a source-trained classifier is treated as fixed,
    unlabeled target features are pseudo-labeled by its predictions, and
    class prototypes are estimated from the target batch only. Final target
    probabilities blend the frozen source-model posterior with target-batch
    prototype posteriors. The method intentionally has no ``target_labels``
    argument; target labels must be reserved for scoring.

    This is Protocol 2 in the cross-subject taxonomy: it uses ``X_t`` but not
    ``y_t`` during adaptation. The original source rows and source labels are
    not accepted by ``fit``; only the fitted source model is required.
    """

    def __init__(
        self,
        source_model: Any | None = None,
        *,
        confidence_threshold: float = 0.75,
        max_iterations: int = 5,
        min_class_count: int = 1,
        min_active_classes: int = 2,
        prototype_weight: float = 0.5,
        prototype_temperature: float = 1.0,
        standardize_target: bool = True,
        feature_space: Literal["input", "model_preprocessor", "auto"] = "auto",
    ):
        self.source_model = source_model
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.min_class_count = min_class_count
        self.min_active_classes = min_active_classes
        self.prototype_weight = prototype_weight
        self.prototype_temperature = prototype_temperature
        self.standardize_target = standardize_target
        self.feature_space = feature_space

    def fit(self, target_features: np.ndarray, *, source_model: Any | None = None, classes: np.ndarray | list[Any] | tuple[Any, ...] | None = None):
        """Fit target-batch adaptation from unlabeled target features only."""

        model = self.source_model if source_model is None else source_model
        if model is None:
            raise ValueError("SourceFreeSubjectAdapter.fit requires a fitted source_model.")
        x_target = _as_2d_array(target_features, "target_features")
        classes_array = _resolve_classes(model, classes)
        source_probabilities = _predict_source_probabilities(model, x_target, classes_array)
        embedding, embedding_mode = _target_embedding(model, x_target, feature_space=self.feature_space)
        embedding = np.asarray(embedding, dtype=float)
        if embedding.shape[0] != x_target.shape[0]:
            raise ValueError("The source-model preprocessor returned a different number of target rows.")
        embedding, mean, scale = _standardize_embedding(embedding, enabled=bool(self.standardize_target))

        threshold = _bounded_float(self.confidence_threshold, "source_free_confidence_threshold", lower=0.0, upper=1.0, include_upper=True)
        max_iterations = _nonnegative_int(self.max_iterations, "source_free_max_iterations")
        min_class_count = _positive_int(self.min_class_count, "source_free_min_class_count")
        min_active_classes = _positive_int(self.min_active_classes, "source_free_min_active_classes")
        prototype_weight = _bounded_float(self.prototype_weight, "source_free_prototype_weight", lower=0.0, upper=1.0, include_upper=True)
        prototype_temperature = _positive_float(self.prototype_temperature, "source_free_prototype_temperature")

        probabilities = source_probabilities.copy()
        prototypes = np.full((classes_array.shape[0], embedding.shape[1]), np.nan, dtype=float)
        active_classes = np.zeros(classes_array.shape[0], dtype=bool)
        class_counts = np.zeros(classes_array.shape[0], dtype=int)
        stop_reason = "max_iterations"
        iterations = 0

        if max_iterations == 0 or prototype_weight == 0.0:
            stop_reason = "prototype_adaptation_disabled"
        else:
            seen_signatures: set[tuple[tuple[int, int], ...]] = set()
            for iteration in range(1, max_iterations + 1):
                pseudo_labels = probabilities.argmax(axis=1).astype(int)
                selected = probabilities.max(axis=1) >= threshold
                signature = _pseudo_label_signature(selected, pseudo_labels)
                if signature in seen_signatures:
                    stop_reason = "selection_repeated"
                    break
                seen_signatures.add(signature)

                prototypes, active_classes, class_counts = _fit_target_prototypes(
                    embedding,
                    probabilities,
                    selected,
                    pseudo_labels,
                    n_classes=classes_array.shape[0],
                    min_class_count=min_class_count,
                )
                n_active = int(np.count_nonzero(active_classes))
                if not np.any(selected):
                    stop_reason = "none_selected"
                    break
                if n_active < min_active_classes:
                    stop_reason = "insufficient_active_classes"
                    break

                prototype_probabilities = _prototype_probabilities(embedding, prototypes, active_classes, temperature=prototype_temperature)
                next_probabilities = _blend_probabilities(source_probabilities, prototype_probabilities, prototype_weight=prototype_weight)
                iterations = iteration
                if np.array_equal(next_probabilities.argmax(axis=1), pseudo_labels):
                    probabilities = next_probabilities
                    stop_reason = "selection_unchanged"
                    break
                probabilities = next_probabilities

        final_selected = probabilities.max(axis=1) >= threshold
        final_pseudo_labels = probabilities.argmax(axis=1).astype(int)
        self.source_model_ = model
        self.classes_ = classes_array
        self.n_features_in_ = x_target.shape[1]
        self.target_rows_ = x_target.shape[0]
        self.source_probabilities_ = source_probabilities
        self.probabilities_ = _normalize_probability_rows(probabilities)
        self.prototypes_ = prototypes
        self.active_classes_ = active_classes
        self.prototype_class_counts_ = class_counts
        self.target_embedding_mean_ = mean
        self.target_embedding_scale_ = scale
        self.embedding_mode_ = embedding_mode
        self.selected_ = final_selected
        self.pseudo_labels_ = final_pseudo_labels
        self.n_iterations_ = iterations
        self.stop_reason_ = stop_reason
        return self

    def predict_proba(self, target_features: np.ndarray) -> np.ndarray:
        """Return source-free-adapted probabilities for target features."""

        if not hasattr(self, "source_model_"):
            raise RuntimeError("SourceFreeSubjectAdapter must be fitted before prediction.")
        x_target = _as_2d_array(target_features, "target_features")
        source_probabilities = _predict_source_probabilities(self.source_model_, x_target, self.classes_)
        if not np.any(self.active_classes_) or float(self.prototype_weight) == 0.0:
            return source_probabilities
        embedding, _embedding_mode = _target_embedding(self.source_model_, x_target, feature_space=self.feature_space)
        embedding = (np.asarray(embedding, dtype=float) - self.target_embedding_mean_) / self.target_embedding_scale_
        prototype_probabilities = _prototype_probabilities(embedding, self.prototypes_, self.active_classes_, temperature=float(self.prototype_temperature))
        return _blend_probabilities(source_probabilities, prototype_probabilities, prototype_weight=float(self.prototype_weight))

    def predict(self, target_features: np.ndarray) -> np.ndarray:
        """Predict target classes after source-free adaptation."""

        return self.classes_[np.argmax(self.predict_proba(target_features), axis=1)]

    def metadata(self) -> dict[str, Any]:
        """Return benchmark/protocol metadata for the fitted adapter."""

        if not hasattr(self, "classes_"):
            raise RuntimeError("SourceFreeSubjectAdapter must be fitted before metadata is available.")
        return {
            "source_free_adaptation": True,
            "source_free_protocol": SOURCE_FREE_ADAPTATION_PROTOCOL,
            "source_free_uses_pretrained_source_model": True,
            "source_free_uses_source_features_during_adaptation": False,
            "source_free_uses_source_labels_during_adaptation": False,
            "source_free_uses_target_features": True,
            "source_free_uses_target_labels": False,
            "source_free_valid_for_benchmark": True,
            "source_free_confidence_threshold": float(self.confidence_threshold),
            "source_free_max_iterations": int(self.max_iterations),
            "source_free_iterations": int(self.n_iterations_),
            "source_free_min_class_count": int(self.min_class_count),
            "source_free_min_active_classes": int(self.min_active_classes),
            "source_free_prototype_weight": float(self.prototype_weight),
            "source_free_prototype_temperature": float(self.prototype_temperature),
            "source_free_standardize_target": bool(self.standardize_target),
            "source_free_feature_space": self.embedding_mode_,
            "source_free_target_rows": int(self.target_rows_),
            "source_free_n_selected": int(np.count_nonzero(self.selected_)),
            "source_free_selected_fraction": float(np.mean(self.selected_)) if self.selected_.size else 0.0,
            "source_free_active_classes": int(np.count_nonzero(self.active_classes_)),
            "source_free_class_counts": _format_class_counts(self.prototype_class_counts_, self.classes_),
            "source_free_stop_reason": self.stop_reason_,
        }


def fit_source_free_predict_proba(
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
) -> SourceFreeAdaptationResult:
    """Fit a source-free adapter and return adapted probabilities for a target batch."""

    adapter = SourceFreeSubjectAdapter(
        source_model=source_model,
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations,
        min_class_count=min_class_count,
        min_active_classes=min_active_classes,
        prototype_weight=prototype_weight,
        prototype_temperature=prototype_temperature,
        standardize_target=standardize_target,
        feature_space=feature_space,
    )
    adapter.fit(target_features, classes=classes)
    return SourceFreeAdaptationResult(adapter=adapter, probabilities=adapter.probabilities_.copy(), metadata=adapter.metadata())


def _as_2d_array(features: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature.")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain only finite values.")
    return x


def _resolve_classes(model: Any, classes: np.ndarray | list[Any] | tuple[Any, ...] | None) -> np.ndarray:
    if classes is not None:
        resolved = np.asarray(classes)
    elif hasattr(model, "classes_"):
        resolved = np.asarray(model.classes_)
    else:
        raise ValueError("classes must be supplied when source_model does not expose classes_.")
    if resolved.ndim != 1 or resolved.shape[0] < 2:
        raise ValueError("Source-free adaptation needs at least two classes.")
    if len(set(map(str, resolved))) != resolved.shape[0]:
        raise ValueError("classes must be unique.")
    return resolved


def _predict_source_probabilities(model: Any, features: np.ndarray, classes: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        probabilities = _softmax_rows(scores)
    else:
        raise ValueError("source_model must expose predict_proba or decision_function.")
    model_classes = np.asarray(getattr(model, "classes_", classes))
    return _align_probability_columns(probabilities, model_classes=model_classes, classes=classes)


def _align_probability_columns(probabilities: np.ndarray, *, model_classes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("source_model probabilities must be a two-dimensional matrix.")
    if probabilities.shape[1] != model_classes.shape[0]:
        raise ValueError("source_model probability columns do not match source_model.classes_.")
    aligned = np.zeros((probabilities.shape[0], classes.shape[0]), dtype=float)
    lookup = {class_label: index for index, class_label in enumerate(model_classes.tolist())}
    for output_index, class_label in enumerate(classes.tolist()):
        if class_label not in lookup:
            raise ValueError(f"source_model is missing requested class {class_label!r}.")
        aligned[:, output_index] = probabilities[:, lookup[class_label]]
    return _normalize_probability_rows(aligned)


def _target_embedding(model: Any, features: np.ndarray, *, feature_space: str) -> tuple[np.ndarray, str]:
    requested = "auto" if feature_space is None else str(feature_space).strip().lower().replace("-", "_")
    if requested not in {"input", "model_preprocessor", "auto"}:
        raise ValueError("source_free feature_space must be one of: input, model_preprocessor, auto.")
    if requested == "input":
        return features, "input"
    transformed = _pipeline_preprocessor_transform(model, features)
    if transformed is not None:
        return transformed, "model_preprocessor"
    if requested == "model_preprocessor":
        raise ValueError("feature_space='model_preprocessor' requires a fitted sklearn Pipeline with at least one transformer step.")
    return features, "input"


def _pipeline_preprocessor_transform(model: Any, features: np.ndarray) -> np.ndarray | None:
    steps = getattr(model, "steps", None)
    if not steps or len(steps) <= 1:
        return None
    transformer_steps = steps[:-1]
    try:
        transformed = Pipeline(transformer_steps).transform(features)
    except Exception:
        return None
    transformed = np.asarray(transformed, dtype=float)
    if transformed.ndim != 2:
        raise ValueError("The source-model preprocessor must return a two-dimensional feature matrix.")
    return transformed


def _standardize_embedding(embedding: np.ndarray, *, enabled: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not enabled:
        mean = np.zeros((1, embedding.shape[1]), dtype=float)
        scale = np.ones((1, embedding.shape[1]), dtype=float)
        return embedding, mean, scale
    mean = embedding.mean(axis=0, keepdims=True)
    scale = embedding.std(axis=0, keepdims=True)
    scale = np.where(scale < _EPS, 1.0, scale)
    return (embedding - mean) / scale, mean, scale


def _fit_target_prototypes(
    embedding: np.ndarray,
    probabilities: np.ndarray,
    selected: np.ndarray,
    pseudo_labels: np.ndarray,
    *,
    n_classes: int,
    min_class_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prototypes = np.full((n_classes, embedding.shape[1]), np.nan, dtype=float)
    active = np.zeros(n_classes, dtype=bool)
    counts = np.zeros(n_classes, dtype=int)
    selected = np.asarray(selected, dtype=bool)
    pseudo_labels = np.asarray(pseudo_labels, dtype=int)
    for class_index in range(n_classes):
        mask = selected & (pseudo_labels == class_index)
        counts[class_index] = int(np.count_nonzero(mask))
        if counts[class_index] < min_class_count:
            continue
        weights = np.clip(probabilities[mask, class_index], _EPS, None)
        prototypes[class_index] = np.average(embedding[mask], axis=0, weights=weights)
        active[class_index] = True
    return prototypes, active, counts


def _prototype_probabilities(embedding: np.ndarray, prototypes: np.ndarray, active_classes: np.ndarray, *, temperature: float) -> np.ndarray:
    active_indices = np.flatnonzero(active_classes)
    if active_indices.size == 0:
        raise ValueError("Cannot compute prototype probabilities without active prototypes.")
    distances = np.full((embedding.shape[0], prototypes.shape[0]), np.inf, dtype=float)
    for class_index in active_indices:
        difference = embedding - prototypes[class_index]
        distances[:, class_index] = np.mean(difference * difference, axis=1)
    logits = -distances / max(float(temperature), _EPS)
    return _softmax_rows(logits)


def _blend_probabilities(source_probabilities: np.ndarray, prototype_probabilities: np.ndarray, *, prototype_weight: float) -> np.ndarray:
    weight = float(prototype_weight)
    source_logits = np.log(np.clip(_normalize_probability_rows(source_probabilities), _EPS, 1.0))
    prototype_logits = np.log(np.clip(_normalize_probability_rows(prototype_probabilities), _EPS, 1.0))
    return _softmax_rows((1.0 - weight) * source_logits + weight * prototype_logits)


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[0] < 1 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional matrix with at least two classes.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("probabilities must be finite.")
    probabilities = np.clip(probabilities, 0.0, None)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("probability rows must have positive mass.")
    return probabilities / row_sums


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    if logits.ndim != 2:
        raise ValueError("logits must be two-dimensional.")
    finite = np.isfinite(logits)
    if not np.all(np.any(finite, axis=1)):
        raise ValueError("Each softmax row must contain at least one finite logit.")
    row_max = np.max(np.where(finite, logits, -np.inf), axis=1, keepdims=True)
    shifted = np.where(finite, logits - row_max, -np.inf)
    exp_logits = np.where(finite, np.exp(np.clip(shifted, -50.0, 50.0)), 0.0)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _pseudo_label_signature(selected: np.ndarray, pseudo_labels: np.ndarray) -> tuple[tuple[int, int], ...]:
    indices = np.flatnonzero(np.asarray(selected, dtype=bool))
    labels = np.asarray(pseudo_labels, dtype=int)
    return tuple((int(index), int(labels[index])) for index in indices)


def _format_class_counts(counts: np.ndarray, classes: np.ndarray) -> str:
    return "|".join(f"{class_label}:{int(count)}" for class_label, count in zip(classes.tolist(), np.asarray(counts, dtype=int), strict=True))


def _positive_int(value: Any, name: str) -> int:
    parsed = _integer(value, name)
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    parsed = _integer(value, name)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative.")
    return parsed


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    number = float(value)
    if not np.isfinite(number) or number % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(number)


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return number


def _bounded_float(value: Any, name: str, *, lower: float, upper: float, include_upper: bool) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite in [{lower}, {upper}{']' if include_upper else ')'}.")
    number = float(value)
    upper_ok = number <= upper if include_upper else number < upper
    if not np.isfinite(number) or number < lower or not upper_ok:
        raise ValueError(f"{name} must be finite in [{lower}, {upper}{']' if include_upper else ')'}.")
    return number

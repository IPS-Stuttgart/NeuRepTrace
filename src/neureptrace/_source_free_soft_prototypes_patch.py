from __future__ import annotations

from typing import Any

import numpy as np

import neureptrace.decoding.source_free as _sf


def _prototype_estimator_mode(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text in {"hard", "selected_hard", "pseudo_label", "pseudo_labels"}:
        return "hard"
    if text in {"soft_selected", "selected_soft", "posterior_selected"}:
        return "soft_selected"
    if text in {"soft_all", "all_soft", "posterior_all", "soft"}:
        return "soft_all"
    raise ValueError("source_free prototype_estimator must be one of: hard, soft_selected, soft_all.")


def _fit_target_prototypes(
    embedding: np.ndarray,
    probabilities: np.ndarray,
    selected: np.ndarray,
    pseudo_labels: np.ndarray,
    *,
    n_classes: int,
    min_class_count: int,
    prototype_estimator: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prototypes = np.full((n_classes, embedding.shape[1]), np.nan, dtype=float)
    active = np.zeros(n_classes, dtype=bool)
    counts = np.zeros(n_classes, dtype=int)
    selected = np.asarray(selected, dtype=bool)
    pseudo_labels = np.asarray(pseudo_labels, dtype=int)
    probabilities = _sf._normalize_probability_rows(probabilities)
    estimator = _prototype_estimator_mode(prototype_estimator)
    for class_index in range(n_classes):
        if estimator == "hard":
            mask = selected & (pseudo_labels == class_index)
            counts[class_index] = int(np.count_nonzero(mask))
            if counts[class_index] < min_class_count:
                continue
            weights = np.clip(probabilities[mask, class_index], _sf._EPS, None)
            prototypes[class_index] = np.average(embedding[mask], axis=0, weights=weights)
            active[class_index] = True
            continue
        row_mask = np.ones(embedding.shape[0], dtype=bool) if estimator == "soft_all" else selected
        weights = np.asarray(probabilities[row_mask, class_index], dtype=float)
        counts[class_index] = int(np.count_nonzero(weights > _sf._EPS))
        if weights.size == 0:
            continue
        mass = float(np.sum(weights))
        effective_count = (mass * mass) / max(float(np.sum(weights * weights)), _sf._EPS)
        if effective_count < float(min_class_count):
            continue
        prototypes[class_index] = np.average(embedding[row_mask], axis=0, weights=np.clip(weights, _sf._EPS, None))
        active[class_index] = True
    return prototypes, active, counts


def _init(
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
    feature_space: str = "auto",
    pseudo_label_selection: str = "confidence",
    balanced_topk_per_class: int | None = None,
    prototype_estimator: str = "hard",
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
    self.pseudo_label_selection = pseudo_label_selection
    self.balanced_topk_per_class = balanced_topk_per_class
    self.prototype_estimator = prototype_estimator


def _fit(self, target_features: np.ndarray, *, source_model: Any | None = None, classes: np.ndarray | list[Any] | tuple[Any, ...] | None = None):
    model = self.source_model if source_model is None else source_model
    if model is None:
        raise ValueError("SourceFreeSubjectAdapter.fit requires a fitted source_model.")
    x_target = _sf._as_2d_array(target_features, "target_features")
    classes_array = _sf._resolve_classes(model, classes)
    source_probabilities = _sf._predict_source_probabilities(model, x_target, classes_array)
    embedding, embedding_mode = _sf._target_embedding(model, x_target, feature_space=self.feature_space)
    embedding = np.asarray(embedding, dtype=float)
    if embedding.shape[0] != x_target.shape[0]:
        raise ValueError("The source-model preprocessor returned a different number of target rows.")
    standardize_target = _sf._boolean(self.standardize_target, "source_free_standardize_target")
    embedding, mean, scale = _sf._standardize_embedding(embedding, enabled=standardize_target)
    threshold = _sf._bounded_float(self.confidence_threshold, "source_free_confidence_threshold", lower=0.0, upper=1.0, include_upper=True)
    max_iterations = _sf._nonnegative_int(self.max_iterations, "source_free_max_iterations")
    min_class_count = _sf._positive_int(self.min_class_count, "source_free_min_class_count")
    min_active_classes = _sf._positive_int(self.min_active_classes, "source_free_min_active_classes")
    prototype_weight = _sf._bounded_float(self.prototype_weight, "source_free_prototype_weight", lower=0.0, upper=1.0, include_upper=True)
    prototype_temperature = _sf._positive_float(self.prototype_temperature, "source_free_prototype_temperature")
    pseudo_label_selection = _sf._pseudo_label_selection_mode(self.pseudo_label_selection)
    balanced_topk_per_class = _sf._optional_positive_int(self.balanced_topk_per_class, "source_free_balanced_topk_per_class")
    prototype_estimator = _prototype_estimator_mode(self.prototype_estimator)
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
            selected = _sf._select_pseudo_label_rows(
                probabilities,
                pseudo_labels,
                threshold=threshold,
                selection_mode=pseudo_label_selection,
                balanced_topk_per_class=balanced_topk_per_class,
                min_class_count=min_class_count,
            )
            signature = _sf._pseudo_label_signature(selected, pseudo_labels)
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
                prototype_estimator=prototype_estimator,
            )
            if not np.any(selected) and prototype_estimator != "soft_all":
                stop_reason = "none_selected"
                break
            if int(np.count_nonzero(active_classes)) < min_active_classes:
                stop_reason = "insufficient_active_classes"
                break
            prototype_probabilities = _sf._prototype_probabilities(embedding, prototypes, active_classes, temperature=prototype_temperature)
            next_probabilities = _sf._blend_probabilities(source_probabilities, prototype_probabilities, prototype_weight=prototype_weight)
            iterations = iteration
            if np.array_equal(next_probabilities.argmax(axis=1), pseudo_labels):
                probabilities = next_probabilities
                stop_reason = "selection_unchanged"
                break
            probabilities = next_probabilities
    final_pseudo_labels = probabilities.argmax(axis=1).astype(int)
    final_selected = _sf._select_pseudo_label_rows(
        probabilities,
        final_pseudo_labels,
        threshold=threshold,
        selection_mode=pseudo_label_selection,
        balanced_topk_per_class=balanced_topk_per_class,
        min_class_count=min_class_count,
    )
    self.source_model_ = model
    self.classes_ = classes_array
    self.n_features_in_ = x_target.shape[1]
    self.target_rows_ = x_target.shape[0]
    self.source_probabilities_ = source_probabilities
    self.probabilities_ = _sf._normalize_probability_rows(probabilities)
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
    self.standardize_target_ = standardize_target
    self.pseudo_label_selection_ = pseudo_label_selection
    self.balanced_topk_per_class_ = balanced_topk_per_class
    self.prototype_estimator_ = prototype_estimator
    return self


def _metadata(self) -> dict[str, Any]:
    metadata = _ORIGINAL_METADATA(self)
    metadata["source_free_prototype_estimator"] = getattr(self, "prototype_estimator_", _prototype_estimator_mode(getattr(self, "prototype_estimator", "hard")))
    return metadata


def _fit_source_free_predict_proba(
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
    feature_space: str = "auto",
    pseudo_label_selection: str = "confidence",
    balanced_topk_per_class: int | None = None,
    prototype_estimator: str = "hard",
) -> _sf.SourceFreeAdaptationResult:
    adapter = _sf.SourceFreeSubjectAdapter(
        source_model=source_model,
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
    adapter.fit(target_features, classes=classes)
    return _sf.SourceFreeAdaptationResult(adapter=adapter, probabilities=adapter.probabilities_.copy(), metadata=adapter.metadata())


_ORIGINAL_METADATA = _sf.SourceFreeSubjectAdapter.metadata
_INSTALLED = False


def install() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _sf.PrototypeEstimator = str
    _sf._prototype_estimator_mode = _prototype_estimator_mode
    _sf.SourceFreeSubjectAdapter.__init__ = _init
    _sf.SourceFreeSubjectAdapter.fit = _fit
    _sf.SourceFreeSubjectAdapter.metadata = _metadata
    _sf.fit_source_free_predict_proba = _fit_source_free_predict_proba
    _INSTALLED = True

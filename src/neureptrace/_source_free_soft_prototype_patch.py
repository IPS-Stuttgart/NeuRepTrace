"""Add soft target-prototype estimators to source-free adaptation.

The source-free adapter deliberately receives only a fitted source model and
unlabeled target features.  Hard pseudo-label prototypes can fail on small
OpenNeuro folds when every target row has the same argmax pseudo-label, even if
minority-class posterior mass is still present.  This patch adds soft prototype
estimators that use unlabeled target posterior weights without target labels.
"""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_free_soft_prototype_patch_installed"
_INIT_MARKER = "_neureptrace_source_free_soft_prototype_init_wrapped"
_FIT_MARKER = "_neureptrace_source_free_soft_prototype_fit_wrapped"
_METADATA_MARKER = "_neureptrace_source_free_soft_prototype_metadata_wrapped"
_FUNC_MARKER = "_neureptrace_source_free_soft_prototype_fit_func_wrapped"


def _prototype_estimator_mode(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_")
    if text in {"", "hard", "argmax", "pseudo_label", "pseudo_labels"}:
        return "hard"
    if text in {"soft_selected", "selected_soft", "posterior_selected"}:
        return "soft_selected"
    if text in {"soft_all", "all_soft", "posterior_all", "soft"}:
        return "soft_all"
    raise ValueError("source_free prototype_estimator must be one of: hard, soft_selected, soft_all.")


def _soft_effective_count(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    weight_sum = float(weights.sum())
    sq_sum = float(np.sum(weights * weights))
    if weight_sum <= 0.0 or sq_sum <= 0.0:
        return 0.0
    return (weight_sum * weight_sum) / sq_sum


def _fit_target_prototypes(
    source_free: Any,
    embedding: np.ndarray,
    probabilities: np.ndarray,
    selected: np.ndarray,
    pseudo_labels: np.ndarray,
    *,
    n_classes: int,
    min_class_count: int,
    prototype_estimator: str = "hard",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = _prototype_estimator_mode(prototype_estimator)
    if mode == "hard":
        return source_free._fit_target_prototypes(
            embedding,
            probabilities,
            selected,
            pseudo_labels,
            n_classes=n_classes,
            min_class_count=min_class_count,
        )

    embedding = np.asarray(embedding, dtype=float)
    probabilities = source_free._normalize_probability_rows(probabilities)
    selected = np.asarray(selected, dtype=bool)
    if selected.shape != (embedding.shape[0],):
        raise ValueError("selected must have one entry per target row.")
    if probabilities.shape[0] != embedding.shape[0]:
        raise ValueError("probabilities and embedding must have the same number of rows.")
    if probabilities.shape[1] != int(n_classes):
        raise ValueError("probabilities must have one column per class.")

    if mode == "soft_all":
        row_mask = np.ones(embedding.shape[0], dtype=bool)
    else:
        row_mask = selected
    row_indices = np.flatnonzero(row_mask)

    prototypes = np.full((int(n_classes), embedding.shape[1]), np.nan, dtype=float)
    active = np.zeros(int(n_classes), dtype=bool)
    counts = np.zeros(int(n_classes), dtype=int)
    if row_indices.size == 0:
        return prototypes, active, counts

    selected_embedding = embedding[row_indices]
    selected_probabilities = probabilities[row_indices]
    for class_index in range(int(n_classes)):
        weights = np.clip(selected_probabilities[:, class_index], 0.0, None)
        weight_sum = float(weights.sum())
        effective_count = _soft_effective_count(weights)
        counts[class_index] = int(np.floor(effective_count + 1e-12))
        if weight_sum <= 0.0 or effective_count < int(min_class_count):
            continue
        prototypes[class_index] = np.average(selected_embedding, axis=0, weights=weights)
        active[class_index] = True
    return prototypes, active, counts


def install() -> None:
    """Patch source-free adaptation with soft prototype estimators."""

    source_free = importlib.import_module("neureptrace.decoding.source_free")
    adapter_cls = source_free.SourceFreeSubjectAdapter

    original_init = adapter_cls.__init__
    if not getattr(original_init, _INIT_MARKER, False):

        @wraps(original_init)
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
            feature_space: str = "auto",
            pseudo_label_selection: str = "confidence",
            balanced_topk_per_class: int | None = None,
            prototype_estimator: str = "hard",
        ):
            original_init(
                self,
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
            )
            self.prototype_estimator = prototype_estimator

        setattr(__init__, _INIT_MARKER, True)
        adapter_cls.__init__ = __init__

    original_fit = adapter_cls.fit
    if not getattr(original_fit, _FIT_MARKER, False):

        @wraps(original_fit)
        def fit(self, target_features, *, source_model=None, classes=None):
            model = self.source_model if source_model is None else source_model
            if model is None:
                raise ValueError("SourceFreeSubjectAdapter.fit requires a fitted source_model.")
            x_target = source_free._as_2d_array(target_features, "target_features")
            classes_array = source_free._resolve_classes(model, classes)
            source_probabilities = source_free._predict_source_probabilities(model, x_target, classes_array)
            embedding, embedding_mode = source_free._target_embedding(model, x_target, feature_space=self.feature_space)
            embedding = np.asarray(embedding, dtype=float)
            if embedding.shape[0] != x_target.shape[0]:
                raise ValueError("The source-model preprocessor returned a different number of target rows.")
            standardize_target = source_free._boolean(self.standardize_target, "source_free_standardize_target")
            embedding, mean, scale = source_free._standardize_embedding(embedding, enabled=standardize_target)

            threshold = source_free._bounded_float(
                self.confidence_threshold,
                "source_free_confidence_threshold",
                lower=0.0,
                upper=1.0,
                include_upper=True,
            )
            max_iterations = source_free._nonnegative_int(self.max_iterations, "source_free_max_iterations")
            min_class_count = source_free._positive_int(self.min_class_count, "source_free_min_class_count")
            min_active_classes = source_free._positive_int(self.min_active_classes, "source_free_min_active_classes")
            prototype_weight = source_free._bounded_float(
                self.prototype_weight,
                "source_free_prototype_weight",
                lower=0.0,
                upper=1.0,
                include_upper=True,
            )
            prototype_temperature = source_free._positive_float(self.prototype_temperature, "source_free_prototype_temperature")
            pseudo_label_selection = source_free._pseudo_label_selection_mode(self.pseudo_label_selection)
            balanced_topk_per_class = source_free._optional_positive_int(
                self.balanced_topk_per_class,
                "source_free_balanced_topk_per_class",
            )
            prototype_estimator = _prototype_estimator_mode(getattr(self, "prototype_estimator", "hard"))

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
                    selected = source_free._select_pseudo_label_rows(
                        probabilities,
                        pseudo_labels,
                        threshold=threshold,
                        selection_mode=pseudo_label_selection,
                        balanced_topk_per_class=balanced_topk_per_class,
                        min_class_count=min_class_count,
                    )
                    prototype_rows = np.ones_like(selected, dtype=bool) if prototype_estimator == "soft_all" else selected
                    signature = source_free._pseudo_label_signature(prototype_rows, pseudo_labels)
                    if signature in seen_signatures:
                        stop_reason = "selection_repeated"
                        break
                    seen_signatures.add(signature)

                    prototypes, active_classes, class_counts = _fit_target_prototypes(
                        source_free,
                        embedding,
                        probabilities,
                        selected,
                        pseudo_labels,
                        n_classes=classes_array.shape[0],
                        min_class_count=min_class_count,
                        prototype_estimator=prototype_estimator,
                    )
                    n_active = int(np.count_nonzero(active_classes))
                    if not np.any(prototype_rows):
                        stop_reason = "none_selected"
                        break
                    if n_active < min_active_classes:
                        stop_reason = "insufficient_active_classes"
                        break

                    prototype_probabilities = source_free._prototype_probabilities(
                        embedding,
                        prototypes,
                        active_classes,
                        temperature=prototype_temperature,
                    )
                    next_probabilities = source_free._blend_probabilities(
                        source_probabilities,
                        prototype_probabilities,
                        prototype_weight=prototype_weight,
                    )
                    iterations = iteration
                    if np.array_equal(next_probabilities.argmax(axis=1), pseudo_labels):
                        probabilities = next_probabilities
                        stop_reason = "selection_unchanged"
                        break
                    probabilities = next_probabilities

            final_pseudo_labels = probabilities.argmax(axis=1).astype(int)
            final_selected = source_free._select_pseudo_label_rows(
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
            self.probabilities_ = source_free._normalize_probability_rows(probabilities)
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
            self.prototype_estimator = prototype_estimator
            return self

        setattr(fit, _FIT_MARKER, True)
        adapter_cls.fit = fit

    original_metadata = adapter_cls.metadata
    if not getattr(original_metadata, _METADATA_MARKER, False):

        @wraps(original_metadata)
        def metadata(self):
            result = dict(original_metadata(self))
            result["source_free_prototype_estimator"] = getattr(
                self,
                "prototype_estimator_",
                _prototype_estimator_mode(getattr(self, "prototype_estimator", "hard")),
            )
            return result

        setattr(metadata, _METADATA_MARKER, True)
        adapter_cls.metadata = metadata

    original_fit_source_free_predict_proba = source_free.fit_source_free_predict_proba
    if not getattr(original_fit_source_free_predict_proba, _FUNC_MARKER, False):

        @wraps(original_fit_source_free_predict_proba)
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
            feature_space: str = "auto",
            pseudo_label_selection: str = "confidence",
            balanced_topk_per_class: int | None = None,
            prototype_estimator: str = "hard",
        ):
            adapter = adapter_cls(
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
            return source_free.SourceFreeAdaptationResult(
                adapter=adapter,
                probabilities=adapter.probabilities_.copy(),
                metadata=adapter.metadata(),
            )

        setattr(fit_source_free_predict_proba, _FUNC_MARKER, True)
        source_free.fit_source_free_predict_proba = fit_source_free_predict_proba

    setattr(source_free, _PATCH_MARKER, True)


__all__ = ["install"]

"""Runtime patch for MEKT label, domain, and pseudo-label vector validation.

The MEKT implementation historically normalized several public vector-like inputs
with ``reshape(-1)`` or by checking only the first array dimension.  That could
turn malformed matrix-shaped labels into apparently valid per-row labels, or let
matrix-shaped domain arrays reach later NumPy masking operations.  This patch
keeps genuine label vectors accepted, including single-row/single-column CLI
vectors, while preserving row-wise composite source-domain identifiers.

It also keeps tuple-valued source-domain identifiers atomic when DTE source-domain
selection materializes the top-k domain list.  Without that guard, NumPy can coerce
``[(subject, run), ...]`` into a 2-D string array, causing matching and domain
selection to operate on flattened scalar cells instead of per-row domain tuples.

Finally, MEKT estimator fits are routed through a dense integer label wrapper.
Valid tuple-valued source classes remain exposed in MEKT results and predictions,
but scikit-learn estimators no longer see sequence-valued class labels as legacy
multilabel targets.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Hashable, Mapping
from dataclasses import replace
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

_TARGET_MODULE = "neureptrace.decoding.mekt"
_PATCH_MARKER = "_neureptrace_mekt_vector_validation_patch_installed"
_FINDER_MARKER = "_neureptrace_mekt_vector_validation_finder"


def _object_array(items: list[Any]) -> np.ndarray:
    if not any(isinstance(item, tuple) for item in items):
        return np.asarray(items)
    vector = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        vector[index] = item
    return vector


def _as_hashable_vector(values: Any, *, name: str) -> np.ndarray:
    """Return a 1-D object vector without silently flattening true matrices."""

    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            items = [array.item()]
        elif array.ndim == 1:
            items = array.tolist()
        elif array.ndim == 2 and 1 in array.shape:
            items = array.reshape(-1).tolist()
        else:
            raise ValueError(f"{name} must be a one-dimensional vector.")
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must be a one-dimensional vector.") from exc

    vector = _object_array(items)
    for value in vector.tolist():
        if not isinstance(value, Hashable):
            raise ValueError(f"{name} entries must be hashable scalar or tuple values.")
    return vector


def _matrix_rows_as_tuples(array: np.ndarray) -> list[tuple[Any, ...]]:
    rows = np.asarray(array, dtype=object).reshape(array.shape[0], -1)
    return [tuple(row.tolist()) for row in rows]


def _as_domain_vector(values: Any, *, name: str) -> np.ndarray:
    """Return a 1-D vector, preserving 2-D rows as composite domain IDs."""

    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim <= 1 or (array.ndim == 2 and 1 in array.shape):
            return _as_hashable_vector(array, name=name)
        items = _matrix_rows_as_tuples(array)
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            raw_items = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must be a one-dimensional vector or row-wise composite matrix.") from exc
        if raw_items and all(isinstance(item, (list, tuple, np.ndarray)) for item in raw_items):
            items = [tuple(np.asarray(item, dtype=object).reshape(-1).tolist()) for item in raw_items]
        else:
            items = raw_items

    vector = _object_array(items)
    for value in vector.tolist():
        if not isinstance(value, Hashable):
            raise ValueError(f"{name} entries must be hashable scalar or tuple values.")
    return vector


def _normalize_optional_vector(value: Any, *, name: str) -> np.ndarray | None:
    return None if value is None else _as_hashable_vector(value, name=name)


def _normalize_optional_domain_vector(value: Any, *, name: str) -> np.ndarray | None:
    return None if value is None else _as_domain_vector(value, name=name)


def _select_top_domains(scores: Mapping[Any, float], top_k: int) -> np.ndarray:
    ordered = sorted(scores, key=lambda key: (-float(scores[key]), str(key)))
    return _object_array(list(ordered[: min(int(top_k), len(ordered))]))


class _EncodedLabelEstimator(BaseEstimator, ClassifierMixin):
    """Fit a scikit-learn estimator on dense ids while exposing original labels."""

    def __init__(self, base_estimator: Any):
        self.base_estimator = base_estimator

    def fit(self, features: Any, labels: Any, **fit_params: Any):
        from neureptrace.decoding.classifiers import encode_classifier_labels

        self.classes_, encoded_labels = encode_classifier_labels(labels)
        self.estimator_ = clone(self.base_estimator)
        self.estimator_.fit(features, encoded_labels, **fit_params)
        return self

    def _require_fitted(self) -> Any:
        if not hasattr(self, "estimator_") or not hasattr(self, "classes_"):
            raise RuntimeError("MEKT label-encoded estimator must be fitted before prediction.")
        return self.estimator_

    def predict(self, features: Any) -> np.ndarray:
        estimator = self._require_fitted()
        encoded = np.asarray(estimator.predict(features), dtype=int).reshape(-1)
        if np.any(encoded < 0) or np.any(encoded >= self.classes_.shape[0]):
            raise ValueError("MEKT estimator returned an encoded label outside the fitted class range.")
        return self.classes_[encoded]

    def predict_proba(self, features: Any) -> np.ndarray:
        estimator = self._require_fitted()
        if not hasattr(estimator, "predict_proba"):
            raise AttributeError(f"{estimator.__class__.__name__!r} object has no attribute 'predict_proba'")
        return np.asarray(estimator.predict_proba(features), dtype=float)

    def decision_function(self, features: Any) -> np.ndarray:
        estimator = self._require_fitted()
        if hasattr(estimator, "decision_function"):
            scores = np.asarray(estimator.decision_function(features), dtype=float)
            if scores.ndim == 1 and self.classes_.shape[0] == 2:
                return np.column_stack((-0.5 * scores, 0.5 * scores))
            return scores
        if hasattr(estimator, "predict_proba"):
            return np.asarray(estimator.predict_proba(features), dtype=float)
        encoded = np.asarray(estimator.predict(features), dtype=int).reshape(-1)
        scores = np.zeros((encoded.shape[0], self.classes_.shape[0]), dtype=float)
        valid = (encoded >= 0) & (encoded < self.classes_.shape[0])
        scores[np.flatnonzero(valid), encoded[valid]] = 1.0
        return scores


def _wrap_estimator(module: ModuleType, estimator: Any | None) -> _EncodedLabelEstimator:
    base_estimator = module._default_estimator() if estimator is None else estimator
    if isinstance(base_estimator, _EncodedLabelEstimator):
        return base_estimator
    return _EncodedLabelEstimator(base_estimator)


def _active_classes_for_result(module: ModuleType, labels: np.ndarray, source_domains: Any, result: Any) -> np.ndarray:
    if result.source_domains.shape[0] == labels.shape[0]:
        return module._unique_value_array(labels)
    domains = module._domain_ids(labels.shape[0], source_domains, name="source_domains")
    keep_mask = module._values_in(domains, result.selected_source_domains)
    return module._unique_value_array(labels[keep_mask])


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_centroid = module.centroid_aligned_tangent_features
    original_transfer = module.mekt_transfer_features
    original_fit_predict = module.fit_predict_mekt_transfer
    original_scores = module.domain_transferability_scores
    original_initial_pseudo_labels = module._initial_pseudo_labels

    def _domain_ids(n_rows: int, source_domains: Any, *, name: str) -> np.ndarray:
        if source_domains is None:
            return np.zeros(n_rows, dtype=int)
        domains = _as_domain_vector(source_domains, name=name)
        if domains.shape[0] != n_rows:
            raise ValueError(f"{name} length must match source rows.")
        return domains

    @wraps(original_initial_pseudo_labels)
    def _initial_pseudo_labels(
        source_features: Any,
        source_labels: Any,
        target_features: Any,
        *,
        classes: Any,
        estimator: Any,
        initial_pseudo_labels: Any,
    ) -> np.ndarray:
        active_classes = module._unique_value_array(source_labels)
        return original_initial_pseudo_labels(
            source_features,
            source_labels,
            target_features,
            classes=active_classes,
            estimator=estimator,
            initial_pseudo_labels=initial_pseudo_labels,
        )

    @wraps(original_centroid)
    def centroid_aligned_tangent_features(
        source_covariances: Any,
        target_covariances: Any,
        *,
        source_domains: Any = None,
        epsilon: float = module.DEFAULT_RIEMANNIAN_EPSILON,
    ) -> Any:
        return original_centroid(
            source_covariances,
            target_covariances,
            source_domains=_normalize_optional_domain_vector(source_domains, name="source_domains"),
            epsilon=epsilon,
        )

    @wraps(original_transfer)
    def mekt_transfer_features(source_covariances: Any, source_labels: Any, target_covariances: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        if "source_domains" in kwargs:
            kwargs["source_domains"] = _normalize_optional_domain_vector(kwargs["source_domains"], name="source_domains")
        if kwargs.get("initial_pseudo_labels") is not None:
            kwargs["initial_pseudo_labels"] = _as_hashable_vector(kwargs["initial_pseudo_labels"], name="initial_pseudo_labels")
        kwargs["estimator"] = _wrap_estimator(module, kwargs.get("estimator"))
        labels = _as_hashable_vector(source_labels, name="source_labels")
        result = original_transfer(source_covariances, labels, target_covariances, **kwargs)
        active_classes = _active_classes_for_result(module, labels, kwargs.get("source_domains"), result)
        if not np.array_equal(result.classes, active_classes):
            result = replace(result, classes=active_classes)
        return result

    @wraps(original_fit_predict)
    def fit_predict_mekt_transfer(source_covariances: Any, source_labels: Any, target_covariances: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        if "source_domains" in kwargs:
            kwargs["source_domains"] = _normalize_optional_domain_vector(kwargs["source_domains"], name="source_domains")
        kwargs["estimator"] = _wrap_estimator(module, kwargs.get("estimator"))
        return original_fit_predict(source_covariances, _as_hashable_vector(source_labels, name="source_labels"), target_covariances, **kwargs)

    @wraps(original_scores)
    def domain_transferability_scores(source_features: Any, source_labels: Any, target_features: Any, source_domains: Any, **kwargs: Any) -> Any:
        return original_scores(
            source_features,
            _as_hashable_vector(source_labels, name="source_labels"),
            target_features,
            _as_domain_vector(source_domains, name="source_domains"),
            **kwargs,
        )

    module._domain_ids = _domain_ids
    module._select_top_domains = _select_top_domains
    module._initial_pseudo_labels = _initial_pseudo_labels
    module.centroid_aligned_tangent_features = centroid_aligned_tangent_features
    module.mekt_transfer_features = mekt_transfer_features
    module.fit_predict_mekt_transfer = fit_predict_mekt_transfer
    module.domain_transferability_scores = domain_transferability_scores
    setattr(module, _PATCH_MARKER, True)


class _MektVectorValidationPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader) -> None:
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):  # type: ignore[override]
        create_module = getattr(self.wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped_loader.exec_module(module)
        _patch_module(module)

    def get_code(self, fullname: str):
        get_code = getattr(self.wrapped_loader, "get_code", None)
        if get_code is None:
            raise ImportError(f"Loader for {fullname!r} does not provide executable code.")
        return get_code(fullname)

    def get_source(self, fullname: str):
        get_source = getattr(self.wrapped_loader, "get_source", None)
        if get_source is None:
            return None
        return get_source(fullname)

    def is_package(self, fullname: str) -> bool:
        is_package = getattr(self.wrapped_loader, "is_package", None)
        if is_package is None:
            return False
        return bool(is_package(fullname))


class _MektVectorValidationPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _MektVectorValidationPatchLoader):
            return spec
        spec.loader = _MektVectorValidationPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install MEKT input-vector validation."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _MektVectorValidationPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)


__all__ = ["install"]

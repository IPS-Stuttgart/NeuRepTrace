"""Runtime patch for MEKT label, domain, and pseudo-label vector validation.

The MEKT implementation historically normalized several public vector-like inputs
with ``reshape(-1)`` or by checking only the first array dimension.  That could
turn malformed matrix-shaped labels into apparently valid per-row labels, or let
matrix-shaped domain arrays reach later NumPy masking operations.  This patch
keeps genuine vectors accepted, including single-row/single-column CLI vectors
and explicit ``dtype=object`` matrices whose rows are composite tuple keys, while
rejecting true numeric/string matrices at the public boundary.

It also keeps tuple-valued source-domain identifiers atomic when DTE source-domain
selection materializes the top-k domain list.  Without that guard, NumPy can coerce
``[(subject, run), ...]`` into a 2-D string array, causing ``np.isin`` to reject
all matching tuple-valued domain rows.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Hashable, Mapping
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

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
        raw_array = np.asarray(values)
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            items = [array.item()]
        elif array.ndim == 1:
            items = array.tolist()
        elif array.ndim == 2 and 1 in array.shape:
            items = array.reshape(-1).tolist()
        elif array.ndim == 2 and raw_array.dtype == object:
            items = [tuple(np.asarray(row, dtype=object).reshape(-1).tolist()) for row in array]
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


def _normalize_optional_vector(value: Any, *, name: str) -> np.ndarray | None:
    return None if value is None else _as_hashable_vector(value, name=name)


def _select_top_domains(scores: Mapping[Any, float], top_k: int) -> np.ndarray:
    ordered = sorted(scores, key=lambda key: (-float(scores[key]), str(key)))
    return _object_array(list(ordered[: min(int(top_k), len(ordered))]))


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_centroid = module.centroid_aligned_tangent_features
    original_transfer = module.mekt_transfer_features
    original_fit_predict = module.fit_predict_mekt_transfer
    original_scores = module.domain_transferability_scores

    def _domain_ids(n_rows: int, source_domains: Any, *, name: str) -> np.ndarray:
        if source_domains is None:
            return np.zeros(n_rows, dtype=int)
        domains = _as_hashable_vector(source_domains, name=name)
        if domains.shape[0] != n_rows:
            raise ValueError(f"{name} length must match source rows.")
        return domains

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
            source_domains=_normalize_optional_vector(source_domains, name="source_domains"),
            epsilon=epsilon,
        )

    @wraps(original_transfer)
    def mekt_transfer_features(source_covariances: Any, source_labels: Any, target_covariances: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        if "source_domains" in kwargs:
            kwargs["source_domains"] = _normalize_optional_vector(kwargs["source_domains"], name="source_domains")
        if kwargs.get("initial_pseudo_labels") is not None:
            kwargs["initial_pseudo_labels"] = _as_hashable_vector(kwargs["initial_pseudo_labels"], name="initial_pseudo_labels")
        return original_transfer(source_covariances, _as_hashable_vector(source_labels, name="source_labels"), target_covariances, **kwargs)

    @wraps(original_fit_predict)
    def fit_predict_mekt_transfer(source_covariances: Any, source_labels: Any, target_covariances: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        if "source_domains" in kwargs:
            kwargs["source_domains"] = _normalize_optional_vector(kwargs["source_domains"], name="source_domains")
        return original_fit_predict(source_covariances, _as_hashable_vector(source_labels, name="source_labels"), target_covariances, **kwargs)

    @wraps(original_scores)
    def domain_transferability_scores(source_features: Any, source_labels: Any, target_features: Any, source_domains: Any, **kwargs: Any) -> Any:
        return original_scores(
            source_features,
            _as_hashable_vector(source_labels, name="source_labels"),
            target_features,
            _as_hashable_vector(source_domains, name="source_domains"),
            **kwargs,
        )

    module._domain_ids = _domain_ids
    module._select_top_domains = _select_top_domains
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

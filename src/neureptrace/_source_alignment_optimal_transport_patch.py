"""Runtime patch adding unlabeled Sinkhorn optimal-transport alignment.

The core source-alignment module owns the public LOSO API.  This extension adds a
Category-2 method that transports source-subject feature distributions toward the
held-out target feature distribution without reading target labels.  The target
rows stay in their native feature space; each source subject is barycentrically
mapped onto the unlabeled target support with an entropic Sinkhorn coupling.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from collections.abc import Hashable, Mapping
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.decoding.source_alignment"
_PATCH_MARKER = "_neureptrace_source_alignment_optimal_transport_patch_installed"
_FINDER_MARKER = "_neureptrace_source_alignment_optimal_transport_finder"
_OT_METHOD = "sinkhorn_transport"
_OT_METHODS = frozenset({_OT_METHOD})
_OT_ALIASES = {
    "barycentric_ot": _OT_METHOD,
    "barycentric_transport": _OT_METHOD,
    "optimal_transport": _OT_METHOD,
    "ot": _OT_METHOD,
    "sinkhorn": _OT_METHOD,
    "sinkhorn_alignment": _OT_METHOD,
    "sinkhorn_barycentric": _OT_METHOD,
    "sinkhorn_barycentric_transport": _OT_METHOD,
    "sinkhorn_ot": _OT_METHOD,
    "sinkhorn_transport": _OT_METHOD,
    "wasserstein": _OT_METHOD,
    "wasserstein_alignment": _OT_METHOD,
    "wasserstein_barycentric": _OT_METHOD,
    "wasserstein_transport": _OT_METHOD,
}
_OT_PROTOCOL = "unlabeled_target_optimal_transport_alignment"
_OT_PROTOCOL_NOTE = (
    "uses unlabeled target feature distribution with Sinkhorn barycentric optimal transport; "
    "report separately from strict source-only alignment"
)
_OT_TRANSFORM_TYPE = "unlabeled_target_sinkhorn_barycentric_transport"
_OT_ESTIMATOR = "sinkhorn_barycentric"
_OT_MAX_ITERATIONS = 100
_OT_TOLERANCE = 1e-6
_OT_EPSILON = 0.08
_OT_MIN_SCALE = 1e-12


def _normalize_ot_method(method: str | None) -> str | None:
    if method is None:
        return None
    return _OT_ALIASES.get(str(method).strip().lower().replace("-", "_"), None)


def _patch_source_alignment(source_alignment: ModuleType) -> None:
    if getattr(source_alignment, _PATCH_MARKER, False):
        return

    original_methods = tuple(getattr(source_alignment, "SOURCE_ALIGNMENT_METHODS"))
    original_unsupervised = tuple(getattr(source_alignment, "SOURCE_ALIGNMENT_UNSUPERVISED_METHODS"))
    if _OT_METHOD not in original_unsupervised:
        source_alignment.SOURCE_ALIGNMENT_UNSUPERVISED_METHODS = (*original_unsupervised, _OT_METHOD)
    if _OT_METHOD not in original_methods:
        source_alignment.SOURCE_ALIGNMENT_METHODS = (*original_methods, _OT_METHOD)

    original_normalize_method = source_alignment.normalize_source_alignment_method
    original_uses_unlabeled = source_alignment._uses_unlabeled_covariance_alignment
    original_transform_by_subject = source_alignment._transform_unsupervised_covariance_alignment_by_subject
    original_static_metadata = source_alignment.SourceAlignmentConfig.static_metadata
    original_align_train_test = source_alignment.align_train_test_features
    original_anchor_availability = source_alignment.source_alignment_anchor_availability

    def normalize_source_alignment_method(method: str | None) -> str:
        ot_method = _normalize_ot_method(method)
        if ot_method is not None:
            return ot_method
        return original_normalize_method(method)

    def _uses_unlabeled_covariance_alignment(method: str) -> bool:
        return method in _OT_METHODS or bool(original_uses_unlabeled(method))

    def _transform_unsupervised_covariance_alignment_by_subject(
        features_by_subject: Mapping[Hashable, np.ndarray],
        test_features: np.ndarray,
        *,
        method: str,
    ) -> tuple[dict[Hashable, np.ndarray], np.ndarray, dict[str, Any]]:
        if method not in _OT_METHODS:
            return original_transform_by_subject(features_by_subject, test_features, method=method)
        return _transform_sinkhorn_transport_by_subject(source_alignment, features_by_subject, test_features)

    def static_metadata(self) -> dict[str, Any]:
        metadata = original_static_metadata(self)
        if self.method in _OT_METHODS:
            metadata.update(
                {
                    "alignment_uses_unlabeled_target_data": True,
                    "alignment_strict_source_only": False,
                    "alignment_valid_for_benchmark": False,
                    "alignment_valid_for_strict_source_only": False,
                    "alignment_protocol": _OT_PROTOCOL,
                    "alignment_protocol_note": _OT_PROTOCOL_NOTE,
                    "alignment_uses_class_labels": False,
                }
            )
        return metadata

    def align_train_test_features(*args, **kwargs):
        result = original_align_train_test(*args, **kwargs)
        config = kwargs.get("config")
        if config is None:
            return result
        if getattr(config, "method", None) not in _OT_METHODS:
            return result
        _mark_ot_result(result)
        return result

    def source_alignment_anchor_availability(*args, **kwargs):
        row = original_anchor_availability(*args, **kwargs)
        config = kwargs.get("config")
        if config is None and len(args) >= 3:
            # The public function is keyword-only today; this keeps the wrapper
            # robust if an older positional call path exists in downstream code.
            config = args[2]
        if getattr(config, "method", None) in _OT_METHODS:
            row.update(
                {
                    "sample_mode": "unlabeled_optimal_transport",
                    "source_anchor_value_source": "unlabeled_target_distribution",
                    "alignment_protocol": _OT_PROTOCOL,
                }
            )
        return row

    source_alignment.normalize_source_alignment_method = normalize_source_alignment_method
    source_alignment._uses_unlabeled_covariance_alignment = _uses_unlabeled_covariance_alignment
    source_alignment._transform_unsupervised_covariance_alignment_by_subject = _transform_unsupervised_covariance_alignment_by_subject
    source_alignment.SourceAlignmentConfig.static_metadata = static_metadata
    source_alignment.align_train_test_features = align_train_test_features
    source_alignment.source_alignment_anchor_availability = source_alignment_anchor_availability
    source_alignment.SINKHORN_TRANSPORT_ALIGNMENT = _OT_METHOD
    setattr(source_alignment, _PATCH_MARKER, True)


def _mark_ot_result(result: Any) -> None:
    result.metadata.update(
        {
            "alignment_anchor_value_source": "unlabeled_target_distribution",
            "alignment_protocol": _OT_PROTOCOL,
            "alignment_protocol_note": _OT_PROTOCOL_NOTE,
            "alignment_covariance_method": _OT_METHOD,
            "covariance_alignment_estimator": _OT_ESTIMATOR,
            "target_transform_type": _OT_TRANSFORM_TYPE,
            "alignment_target_projection_fit": _OT_TRANSFORM_TYPE,
            "alignment_uses_unlabeled_target_data": True,
            "alignment_uses_class_labels": False,
            "alignment_valid_for_benchmark": False,
            "alignment_valid_for_strict_source_only": False,
            "alignment_strict_source_only": False,
        }
    )
    result.diagnostics.update(
        {
            "sample_mode": "unlabeled_optimal_transport",
            "requested_components": "unlabeled_optimal_transport",
            "uses_unlabeled_target_data": True,
            "covariance_alignment_estimator": _OT_ESTIMATOR,
            "target_transform_type": _OT_TRANSFORM_TYPE,
            "source_inner_validation_type": "strict_source_loso_nearest_centroid_unlabeled_target_optimal_transport"
            if result.diagnostics.get("source_inner_validation_type")
            else "",
        }
    )


def _transform_sinkhorn_transport_by_subject(
    source_alignment: ModuleType,
    features_by_subject: Mapping[Hashable, np.ndarray],
    test_features: np.ndarray,
) -> tuple[dict[Hashable, np.ndarray], np.ndarray, dict[str, Any]]:
    target = source_alignment._feature_matrix(test_features, name="test_features")
    transformed_by_subject = {
        subject_id: _sinkhorn_barycentric_transport(
            source_alignment._feature_matrix(features, name="source_subject_features"),
            target,
        )
        for subject_id, features in features_by_subject.items()
    }
    return (
        transformed_by_subject,
        np.asarray(target, dtype=float),
        {
            "alignment_uses_unlabeled_target_data": True,
            "alignment_covariance_method": _OT_METHOD,
            "covariance_alignment_estimator": _OT_ESTIMATOR,
            "target_transform_type": _OT_TRANSFORM_TYPE,
        },
    )


def _sinkhorn_barycentric_transport(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("Sinkhorn transport expects two-dimensional feature matrices.")
    if source.shape[1] != target.shape[1]:
        raise ValueError("Source and target feature matrices must have the same feature width.")
    if source.shape[0] == 0 or target.shape[0] == 0:
        raise ValueError("Sinkhorn transport requires non-empty source and target feature matrices.")
    if target.shape[0] == 1:
        return np.repeat(target, source.shape[0], axis=0)

    cost = _scaled_squared_euclidean_cost(source, target)
    epsilon = max(_OT_EPSILON, _OT_MIN_SCALE)
    log_kernel = -cost / epsilon
    log_a = np.full(source.shape[0], -np.log(source.shape[0]), dtype=float)
    log_b = np.full(target.shape[0], -np.log(target.shape[0]), dtype=float)
    log_u = np.zeros(source.shape[0], dtype=float)
    log_v = np.zeros(target.shape[0], dtype=float)

    for _iteration in range(_OT_MAX_ITERATIONS):
        previous_log_u = log_u.copy()
        log_u = log_a - _logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - _logsumexp(log_kernel.T + log_u[None, :], axis=1)
        if np.max(np.abs(log_u - previous_log_u)) < _OT_TOLERANCE:
            break

    log_plan = log_u[:, None] + log_kernel + log_v[None, :]
    # Row-normalize before exponentiation to avoid underflow in very separated
    # subject/target distributions.  This preserves the barycentric map because
    # only the conditional target weights per source row matter.
    log_plan = log_plan - _logsumexp(log_plan, axis=1)[:, None]
    weights = np.exp(log_plan)
    transported = weights @ target
    return np.asarray(transported, dtype=float)


def _scaled_squared_euclidean_cost(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_norm = np.sum(source * source, axis=1)[:, None]
    target_norm = np.sum(target * target, axis=1)[None, :]
    cost = np.maximum(source_norm + target_norm - 2.0 * source @ target.T, 0.0)
    finite = cost[np.isfinite(cost)]
    if finite.size == 0:
        raise ValueError("Sinkhorn transport encountered non-finite pairwise distances.")
    positive = finite[finite > 0.0]
    scale = float(np.median(positive)) if positive.size else float(np.mean(finite))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return cost / max(scale, _OT_MIN_SCALE)


def _logsumexp(values: np.ndarray, *, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    stable = np.exp(values - max_values)
    summed = np.sum(stable, axis=axis, keepdims=True)
    return np.squeeze(max_values + np.log(summed), axis=axis)


class _SourceAlignmentOptimalTransportPatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader) -> None:
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):  # type: ignore[override]
        create_module = getattr(self.wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped_loader.exec_module(module)
        _patch_source_alignment(module)


class _SourceAlignmentOptimalTransportPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _SourceAlignmentOptimalTransportPatchLoader):
            return spec
        spec.loader = _SourceAlignmentOptimalTransportPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install unlabeled Sinkhorn optimal-transport source alignment."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_source_alignment(loaded)
        return

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _SourceAlignmentOptimalTransportPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)

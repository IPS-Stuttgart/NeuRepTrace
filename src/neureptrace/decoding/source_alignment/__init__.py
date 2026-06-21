"""Riemannian Procrustes extension for ``decoding.source_alignment``.

This package intentionally shadows the historical ``source_alignment.py`` module,
loads that implementation under a private module name, and then installs one
additional unlabeled target-adaptive covariance alignment method.  Keeping the
legacy file intact lets this change stay small while preserving all existing
public and private imports from ``neureptrace.decoding.source_alignment``.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from types import ModuleType

import numpy as np

_LEGACY_MODULE_NAME = "_neureptrace_decoding_source_alignment_legacy"
_LEGACY_PATH = Path(__file__).resolve().parent.parent / "source_alignment.py"

if not _LEGACY_PATH.exists():  # pragma: no cover - repository packaging guardrail
    raise ImportError(f"Could not locate legacy source_alignment.py at {_LEGACY_PATH}.")

_spec = importlib.util.spec_from_file_location(_LEGACY_MODULE_NAME, _LEGACY_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - importlib guardrail
    raise ImportError(f"Could not load legacy source_alignment.py from {_LEGACY_PATH}.")
_legacy = importlib.util.module_from_spec(_spec)
sys.modules[_LEGACY_MODULE_NAME] = _legacy
_spec.loader.exec_module(_legacy)

RIEMANNIAN_PROCRUSTES_METHOD = "riemannian_procrustes"
_RIEMANNIAN_PROCRUSTES_ALIASES = {
    "rpa",
    "full_rpa",
    "riemannian_procrustes",
    "riemannian_procrustes_alignment",
    "riemannian_procrustes_analysis",
    "riemannian_procrustes_covariance",
    "riemannian_covariance_procrustes",
    "procrustes_covariance",
    "covariance_procrustes",
}
_ORIGINAL_NORMALIZE_SOURCE_ALIGNMENT_METHOD = _legacy.normalize_source_alignment_method
_ORIGINAL_TRANSFORM_UNSUPERVISED_COVARIANCE_ALIGNMENT_BY_SUBJECT = (
    _legacy._transform_unsupervised_covariance_alignment_by_subject
)


@dataclass(frozen=True, slots=True)
class _RiemannianProcrustesFeatureStats:
    """Label-free distribution summary used by RPA-style feature alignment."""

    mean: np.ndarray
    basis: np.ndarray
    dispersion: float
    estimator: str


def _canonicalize_eigenvector_signs(vectors: np.ndarray) -> np.ndarray:
    """Make eigenspace signs deterministic for reproducible Procrustes rotations."""

    basis = np.asarray(vectors, dtype=float).copy()
    for column in range(basis.shape[1]):
        pivot = int(np.argmax(np.abs(basis[:, column])))
        if basis[pivot, column] < 0.0:
            basis[:, column] *= -1.0
    return basis


def _riemannian_procrustes_feature_stats(features: np.ndarray) -> _RiemannianProcrustesFeatureStats:
    """Estimate a feature-space mean, principal frame, and scalar dispersion.

    The original Riemannian Procrustes Analysis literature acts on SPD covariance
    points.  NeuRepTrace's source-alignment hook receives a two-dimensional
    feature matrix, so this method implements the label-free RPA steps that are
    available at this level: translation to the target mean, isotropic dispersion
    scaling, and an orthogonal principal-frame rotation.  No decoder labels,
    target labels, class prototypes, or target accuracy are used.
    """

    matrix = _legacy._feature_matrix(features, name="riemannian_procrustes_features")
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    n_rows, n_features = centered.shape
    if n_features <= _legacy.MAX_FULL_COVARIANCE_FEATURES and n_rows > 1:
        covariance = centered.T @ centered / max(1, n_rows - 1)
        trace_scale = float(np.trace(covariance) / max(1, n_features))
        floor = _legacy.MIN_COVARIANCE_EIGENVALUE * max(trace_scale, 1.0)
        values, vectors = np.linalg.eigh(covariance)
        order = np.argsort(values)[::-1]
        values = np.maximum(values[order], floor)
        basis = _canonicalize_eigenvector_signs(vectors[:, order])
        dispersion = float(np.sqrt(np.mean(values)))
        return _RiemannianProcrustesFeatureStats(
            mean=mean,
            basis=basis,
            dispersion=dispersion,
            estimator="full",
        )

    variance = np.var(centered, axis=0, ddof=1 if n_rows > 1 else 0)
    trace_scale = float(np.mean(variance)) if variance.size else 1.0
    floor = _legacy.MIN_COVARIANCE_EIGENVALUE * max(trace_scale, 1.0)
    dispersion = float(np.sqrt(np.mean(np.maximum(variance, floor)))) if variance.size else 1.0
    return _RiemannianProcrustesFeatureStats(
        mean=mean,
        basis=np.eye(n_features, dtype=float),
        dispersion=dispersion,
        estimator="diagonal",
    )


def _riemannian_procrustes_transform_to_target(
    source_features: np.ndarray,
    *,
    source_stats: _RiemannianProcrustesFeatureStats,
    target_stats: _RiemannianProcrustesFeatureStats,
) -> tuple[np.ndarray, bool, float]:
    """Map source rows into the unlabeled target subject's feature distribution."""

    matrix = _legacy._feature_matrix(source_features, name="riemannian_procrustes_source_features")
    centered = matrix - source_stats.mean
    use_rotation = source_stats.estimator == "full" and target_stats.estimator == "full"
    if use_rotation:
        rotation = source_stats.basis @ target_stats.basis.T
        centered = centered @ rotation
    denom = max(float(source_stats.dispersion), np.finfo(float).eps)
    scale = float(target_stats.dispersion) / denom
    transformed = centered * scale + target_stats.mean
    return transformed, use_rotation, scale


def _transform_riemannian_procrustes_alignment_by_subject(
    features_by_subject: Mapping[Hashable, np.ndarray],
    test_features: np.ndarray,
) -> tuple[dict[Hashable, np.ndarray], np.ndarray, dict[str, Any]]:
    """Apply unlabeled target-adaptive RPA-style feature alignment.

    Source subjects are translated, isotropically scaled, and optionally rotated
    into the held-out target feature distribution.  The target rows themselves are
    left in their native coordinates.  Because the target distribution is used,
    this is Category 2 rather than strict source-only; because no target labels
    or anchors are used, it is not calibrated Category 3.
    """

    subject_ids = tuple(features_by_subject)
    target_stats = _riemannian_procrustes_feature_stats(np.asarray(test_features, dtype=float))
    transformed_by_subject: dict[Hashable, np.ndarray] = {}
    estimators = {target_stats.estimator}
    used_rotation = False
    scales: list[float] = []
    for subject_id in subject_ids:
        source_stats = _riemannian_procrustes_feature_stats(np.asarray(features_by_subject[subject_id], dtype=float))
        estimators.add(source_stats.estimator)
        transformed, subject_used_rotation, scale = _riemannian_procrustes_transform_to_target(
            np.asarray(features_by_subject[subject_id], dtype=float),
            source_stats=source_stats,
            target_stats=target_stats,
        )
        transformed_by_subject[subject_id] = transformed
        used_rotation = used_rotation or subject_used_rotation
        scales.append(scale)

    transform_steps = "recenter_scale_rotate" if used_rotation else "recenter_scale"
    transformed_test = np.asarray(test_features, dtype=float)
    return (
        transformed_by_subject,
        transformed_test,
        {
            "alignment_uses_unlabeled_target_data": True,
            "alignment_covariance_method": RIEMANNIAN_PROCRUSTES_METHOD,
            "covariance_alignment_estimator": "|".join(sorted(estimators)),
            "target_transform_type": f"unlabeled_target_riemannian_procrustes_{transform_steps}",
            "riemannian_procrustes_rotation_used": bool(used_rotation),
            "riemannian_procrustes_mean_scale": float(np.mean(scales)) if scales else "",
        },
    )


def normalize_source_alignment_method(method: str | None) -> str:
    """Normalize source-alignment names, including RPA aliases."""

    normalized = "none" if method is None else str(method).strip().lower().replace("-", "_")
    if normalized in _RIEMANNIAN_PROCRUSTES_ALIASES:
        return RIEMANNIAN_PROCRUSTES_METHOD
    return _ORIGINAL_NORMALIZE_SOURCE_ALIGNMENT_METHOD(method)


def _uses_unlabeled_covariance_alignment(method: str) -> bool:
    return method in _legacy.SOURCE_ALIGNMENT_UNSUPERVISED_METHODS


def _transform_unsupervised_covariance_alignment_by_subject(
    features_by_subject: Mapping[Hashable, np.ndarray],
    test_features: np.ndarray,
    *,
    method: str,
) -> tuple[dict[Hashable, np.ndarray], np.ndarray, dict[str, Any]]:
    normalized = normalize_source_alignment_method(method)
    if normalized == RIEMANNIAN_PROCRUSTES_METHOD:
        return _transform_riemannian_procrustes_alignment_by_subject(features_by_subject, test_features)
    return _ORIGINAL_TRANSFORM_UNSUPERVISED_COVARIANCE_ALIGNMENT_BY_SUBJECT(
        features_by_subject,
        test_features,
        method=normalized,
    )


def _install_patches() -> None:
    _legacy.RIEMANNIAN_PROCRUSTES_METHOD = RIEMANNIAN_PROCRUSTES_METHOD
    _legacy.SOURCE_ALIGNMENT_UNSUPERVISED_METHODS = tuple(
        dict.fromkeys((*_legacy.SOURCE_ALIGNMENT_UNSUPERVISED_METHODS, RIEMANNIAN_PROCRUSTES_METHOD))
    )
    _legacy.SOURCE_ALIGNMENT_METHODS = tuple(
        dict.fromkeys(("none", *_legacy.SOURCE_ALIGNMENT_CLASS_ANCHORED_METHODS, *_legacy.SOURCE_ALIGNMENT_UNSUPERVISED_METHODS))
    )
    _legacy.normalize_source_alignment_method = normalize_source_alignment_method
    _legacy._uses_unlabeled_covariance_alignment = _uses_unlabeled_covariance_alignment
    _legacy._transform_unsupervised_covariance_alignment_by_subject = _transform_unsupervised_covariance_alignment_by_subject
    _legacy._transform_riemannian_procrustes_alignment_by_subject = _transform_riemannian_procrustes_alignment_by_subject
    _legacy._riemannian_procrustes_feature_stats = _riemannian_procrustes_feature_stats
    _legacy._riemannian_procrustes_transform_to_target = _riemannian_procrustes_transform_to_target
    _legacy.__all__ = list(dict.fromkeys((*getattr(_legacy, "__all__", ()), "RIEMANNIAN_PROCRUSTES_METHOD")))


_install_patches()

for _name, _value in vars(_legacy).items():
    if _name in {"__builtins__", "__cached__", "__file__", "__loader__", "__name__", "__package__", "__spec__"}:
        continue
    globals()[_name] = _value


class _SourceAlignmentModule(ModuleType):
    def __setattr__(self, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if hasattr(_legacy, name):
            setattr(_legacy, name, value)


sys.modules[__name__].__class__ = _SourceAlignmentModule

__all__ = list(_legacy.__all__)

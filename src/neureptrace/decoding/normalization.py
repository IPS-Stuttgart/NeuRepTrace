"""Feature-normalization helpers for decoding benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

NORMALIZATION_MODES = (
    "none",
    "subject_z",
    "subject_trial_z",
    "subject_baseline_z",
    "subject_baseline_whiten",
)
DEFAULT_BASELINE_WHITENING_SHRINKAGE = 0.1
DEFAULT_BASELINE_WHITENING_EIGENVALUE_FLOOR = 1e-6
_STD_FLOOR = 1e-12


@dataclass(frozen=True)
class FeatureNormalizer:
    """Fitted feature normalizer for row-wise decoding feature matrices.

    The normalizer is deliberately dataset-agnostic: callers provide already
    extracted feature matrices.  Baseline statistics can either be supplied
    directly or estimated from baseline feature rows.  For block-structured
    features, for example flattened time-by-sensor windows, baseline statistics
    and whitening matrices may describe one feature block; ``transform`` tiles
    statistics or applies whitening block-by-block when the transformed feature
    width is an integer multiple of ``block_size``.
    """

    mode: str = "none"
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    whitening_matrix_: np.ndarray | None = None
    block_size_: int | None = None
    shrinkage_: float = DEFAULT_BASELINE_WHITENING_SHRINKAGE
    eigenvalue_floor_: float = DEFAULT_BASELINE_WHITENING_EIGENVALUE_FLOOR

    def transform(self, features: Any) -> np.ndarray:
        """Return normalized copies of ``features`` without modifying the input."""

        features = _as_feature_matrix(features)
        mode = normalize_feature_normalization(self.mode)
        if mode == "none":
            return features.copy()
        if mode == "subject_trial_z":
            return trial_zscore_features(features)
        if mode in {"subject_z", "subject_baseline_z"}:
            if self.mean_ is None or self.scale_ is None:
                raise ValueError(f"{mode} requires fitted mean and scale statistics.")
            mean = _broadcast_feature_stat(self.mean_, features.shape[1], block_size=self.block_size_)
            scale = _nonzero_std(_broadcast_feature_stat(self.scale_, features.shape[1], block_size=self.block_size_))
            return (features - mean) / scale
        if mode == "subject_baseline_whiten":
            if self.mean_ is None or self.whitening_matrix_ is None:
                raise ValueError("subject_baseline_whiten requires a fitted mean and whitening matrix.")
            mean = _broadcast_feature_stat(self.mean_, features.shape[1], block_size=self.block_size_)
            return baseline_whiten_features(features - mean, self.whitening_matrix_, block_size=self.block_size_)
        raise ValueError(f"Unsupported normalization mode: {self.mode}")


def fit_feature_normalizer(
    features: Any,
    *,
    mode: str = "none",
    baseline_features: Any | None = None,
    baseline_mean: Any | None = None,
    baseline_scale: Any | None = None,
    whitening_matrix: Any | None = None,
    block_size: int | None = None,
    shrinkage: float = DEFAULT_BASELINE_WHITENING_SHRINKAGE,
    eigenvalue_floor: float = DEFAULT_BASELINE_WHITENING_EIGENVALUE_FLOOR,
) -> FeatureNormalizer:
    """Fit a normalizer for a decoding feature matrix.

    Parameters
    ----------
    features:
        Feature matrix used to infer feature width and, for ``subject_z``, to fit
        feature-wise mean and scale.  Pass only train/subject-local rows here to
        avoid leakage.
    mode:
        One of ``NORMALIZATION_MODES``.
    baseline_features:
        Optional baseline feature rows.  These are used to fit mean/scale for
        ``subject_baseline_z`` and mean/covariance for
        ``subject_baseline_whiten``.
    baseline_mean, baseline_scale, whitening_matrix:
        Optional precomputed statistics.  These let dataset-specific loaders
        compute baseline statistics from richer objects while still using the
        generic transform code.
    block_size:
        Optional width of one sensor/feature block.  If omitted for whitening it
        defaults to the whitening-matrix width; otherwise statistics whose width
        is smaller than the feature width are tiled when possible.
    """

    features = _as_feature_matrix(features)
    mode = normalize_feature_normalization(mode)
    block_size = _normalize_optional_positive_int(block_size, name="block_size")
    shrinkage = _normalize_fraction(shrinkage, name="shrinkage")
    eigenvalue_floor = _normalize_positive_float(eigenvalue_floor, name="eigenvalue_floor")

    if mode == "none":
        return FeatureNormalizer(mode=mode, block_size_=block_size, shrinkage_=shrinkage, eigenvalue_floor_=eigenvalue_floor)

    if mode == "subject_trial_z":
        return FeatureNormalizer(mode=mode, block_size_=block_size, shrinkage_=shrinkage, eigenvalue_floor_=eigenvalue_floor)

    if mode == "subject_z":
        mean = np.mean(features, axis=0, keepdims=True)
        scale = _nonzero_std(np.std(features, axis=0, keepdims=True))
        return FeatureNormalizer(mode=mode, mean_=mean, scale_=scale, block_size_=block_size, shrinkage_=shrinkage, eigenvalue_floor_=eigenvalue_floor)

    if mode == "subject_baseline_z":
        mean, scale = _baseline_mean_and_scale(
            baseline_features,
            baseline_mean=baseline_mean,
            baseline_scale=baseline_scale,
        )
        return FeatureNormalizer(mode=mode, mean_=mean, scale_=_nonzero_std(scale), block_size_=block_size, shrinkage_=shrinkage, eigenvalue_floor_=eigenvalue_floor)

    if mode == "subject_baseline_whiten":
        baseline_matrix = None if baseline_features is None else _as_feature_matrix(baseline_features)
        mean = _baseline_mean(baseline_matrix, baseline_mean=baseline_mean)
        whitening = _baseline_whitening_matrix(
            baseline_matrix,
            whitening_matrix=whitening_matrix,
            shrinkage=shrinkage,
            eigenvalue_floor=eigenvalue_floor,
        )
        if block_size is None:
            block_size = int(whitening.shape[0])
        _validate_square_matrix(whitening, name="whitening_matrix")
        if whitening.shape[0] != block_size:
            raise ValueError("block_size must match whitening_matrix width.")
        return FeatureNormalizer(
            mode=mode,
            mean_=mean,
            whitening_matrix_=whitening,
            block_size_=block_size,
            shrinkage_=shrinkage,
            eigenvalue_floor_=eigenvalue_floor,
        )

    raise ValueError(f"Unsupported normalization mode: {mode}")


def normalize_features(features: Any, **kwargs: Any) -> np.ndarray:
    """Fit a normalizer with ``kwargs`` and return transformed ``features``."""

    return fit_feature_normalizer(features, **kwargs).transform(features)


def normalize_feature_normalization(value: str) -> str:
    """Normalize and validate a feature-normalization mode name."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in NORMALIZATION_MODES:
        raise ValueError(f"normalization must be one of {NORMALIZATION_MODES}.")
    return normalized


def trial_zscore_features(features: Any) -> np.ndarray:
    """Z-score each trial/row independently."""

    features = _as_feature_matrix(features)
    mean = np.mean(features, axis=1, keepdims=True)
    scale = _nonzero_std(np.std(features, axis=1, keepdims=True))
    return (features - mean) / scale


def baseline_whiten_features(features: Any, whitening_matrix: Any, *, block_size: int | None = None) -> np.ndarray:
    """Apply a whitening matrix to a feature matrix, optionally per block."""

    features = _as_feature_matrix(features)
    whitening_matrix = _validate_square_matrix(whitening_matrix, name="whitening_matrix")
    if block_size is None:
        block_size = int(whitening_matrix.shape[0])
    block_size = _normalize_optional_positive_int(block_size, name="block_size")
    if int(whitening_matrix.shape[0]) != block_size:
        raise ValueError("block_size must match whitening_matrix width.")
    if features.shape[1] == block_size:
        return features @ whitening_matrix.T
    if features.shape[1] % block_size:
        raise ValueError("Feature width must equal block_size or be an integer multiple of block_size.")
    n_blocks = int(features.shape[1] // block_size)
    blocks = features.reshape(features.shape[0], n_blocks, block_size)
    whitened = blocks @ whitening_matrix.T
    return whitened.reshape(features.shape[0], -1)


def covariance_matrix(features: Any) -> np.ndarray:
    """Return a symmetric feature covariance matrix with safe one-row handling."""

    features = _as_feature_matrix(features)
    n_features = int(features.shape[1])
    if features.shape[0] < 2:
        return np.eye(n_features, dtype=float)
    covariance = np.asarray(np.cov(features, rowvar=False), dtype=float)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    return 0.5 * (covariance + covariance.T)


def shrink_covariance(covariance: Any, *, shrinkage: float = DEFAULT_BASELINE_WHITENING_SHRINKAGE) -> np.ndarray:
    """Shrink off-diagonal covariance entries toward a diagonal covariance."""

    covariance = _validate_square_matrix(covariance, name="covariance")
    shrinkage = _normalize_fraction(shrinkage, name="shrinkage")
    diagonal = np.diag(np.diag(covariance))
    return (1.0 - shrinkage) * covariance + shrinkage * diagonal


def whitening_matrix(covariance: Any, *, eigenvalue_floor: float = DEFAULT_BASELINE_WHITENING_EIGENVALUE_FLOOR) -> np.ndarray:
    """Return a symmetric inverse-square-root whitening matrix."""

    covariance = _validate_square_matrix(covariance, name="covariance")
    eigenvalue_floor = _normalize_positive_float(eigenvalue_floor, name="eigenvalue_floor")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = max(float(np.max(eigenvalues)) * eigenvalue_floor, _STD_FLOOR)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, floor))
    whitening = (eigenvectors * inverse_sqrt) @ eigenvectors.T
    return 0.5 * (whitening + whitening.T)


def _baseline_mean_and_scale(baseline_features, *, baseline_mean=None, baseline_scale=None) -> tuple[np.ndarray, np.ndarray]:
    baseline_matrix = None if baseline_features is None else _as_feature_matrix(baseline_features)
    mean = _baseline_mean(baseline_matrix, baseline_mean=baseline_mean)
    if baseline_scale is None:
        if baseline_matrix is None:
            raise ValueError("baseline_features or baseline_scale is required for subject_baseline_z.")
        scale = np.std(baseline_matrix, axis=0, keepdims=True)
    else:
        scale = _as_stat_row(baseline_scale, name="baseline_scale")
    return mean, scale


def _baseline_mean(baseline_matrix: np.ndarray | None, *, baseline_mean=None) -> np.ndarray:
    if baseline_mean is not None:
        return _as_stat_row(baseline_mean, name="baseline_mean")
    if baseline_matrix is None:
        raise ValueError("baseline_features or baseline_mean is required for baseline normalization.")
    return np.mean(baseline_matrix, axis=0, keepdims=True)


def _baseline_whitening_matrix(
    baseline_matrix: np.ndarray | None,
    *,
    whitening_matrix=None,
    shrinkage: float,
    eigenvalue_floor: float,
) -> np.ndarray:
    if whitening_matrix is not None:
        return _validate_square_matrix(whitening_matrix, name="whitening_matrix")
    if baseline_matrix is None:
        raise ValueError("baseline_features or whitening_matrix is required for subject_baseline_whiten.")
    covariance = covariance_matrix(baseline_matrix)
    return whitening_matrix_from_baseline_covariance(covariance, shrinkage=shrinkage, eigenvalue_floor=eigenvalue_floor)


def whitening_matrix_from_baseline_covariance(
    covariance: Any,
    *,
    shrinkage: float = DEFAULT_BASELINE_WHITENING_SHRINKAGE,
    eigenvalue_floor: float = DEFAULT_BASELINE_WHITENING_EIGENVALUE_FLOOR,
) -> np.ndarray:
    """Shrink a baseline covariance matrix and return its whitening transform."""

    return whitening_matrix(shrink_covariance(covariance, shrinkage=shrinkage), eigenvalue_floor=eigenvalue_floor)


def _broadcast_feature_stat(stat: Any, n_features: int, *, block_size: int | None = None) -> np.ndarray:
    stat = _as_stat_row(stat, name="feature statistic")
    if stat.shape[1] == n_features:
        return stat
    if block_size is None:
        block_size = int(stat.shape[1])
    block_size = _normalize_optional_positive_int(block_size, name="block_size")
    if stat.shape[1] != block_size:
        raise ValueError("Feature statistic width must match feature width or block_size.")
    if n_features % block_size:
        raise ValueError("Feature width must be an integer multiple of the statistic width.")
    return np.tile(stat, int(n_features // block_size))


def _as_feature_matrix(features: Any) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix.")
    if matrix.shape[1] == 0:
        raise ValueError("features must contain at least one column.")
    return matrix


def _as_stat_row(values: Any, *, name: str) -> np.ndarray:
    row = np.asarray(values, dtype=float)
    if row.ndim == 1:
        row = row[None, :]
    if row.ndim != 2 or row.shape[0] != 1 or row.shape[1] == 0:
        raise ValueError(f"{name} must be a one-dimensional vector or a single-row matrix.")
    return row


def _validate_square_matrix(matrix: Any, *, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix.")
    return 0.5 * (matrix + matrix.T)


def _nonzero_std(scale: Any) -> np.ndarray:
    return np.where(np.asarray(scale, dtype=float) < _STD_FLOOR, 1.0, scale)


def _normalize_optional_positive_int(value: int | None, *, name: str) -> int | None:
    if value is None:
        return None
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive or None.")
    return value


def _normalize_positive_float(value: float, *, name: str) -> float:
    value = float(value)
    if value <= 0.0 or not np.isfinite(value):
        raise ValueError(f"{name} must be a positive finite value.")
    return value


def _normalize_fraction(value: float, *, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return value

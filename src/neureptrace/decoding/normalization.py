"""Subject-level feature normalization and baseline whitening utilities."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

NormalizationMode = Literal[
    "none",
    "subject_z",
    "subject_trial_z",
    "subject_baseline_z",
    "subject_baseline_whiten",
]

NORMALIZATION_MODES: tuple[str, ...] = (
    "none",
    "subject_z",
    "subject_trial_z",
    "subject_baseline_z",
    "subject_baseline_whiten",
)
BASELINE_WHITENING_SHRINKAGE = 0.1
BASELINE_WHITENING_EIGENVALUE_FLOOR = 1e-6
STD_EPS = 1e-12


def normalize_normalization(value: str) -> str:
    """Normalize a feature-normalization mode name."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized not in NORMALIZATION_MODES:
        raise ValueError(f"normalization must be one of {NORMALIZATION_MODES}.")
    return normalized


def nonzero_std(std: ArrayLike, *, eps: float = STD_EPS) -> NDArray[np.float64]:
    """Replace near-zero standard deviations by one to avoid division blowups."""

    std_array = np.asarray(std, dtype=float)
    return np.where(std_array < float(eps), 1.0, std_array)


def subject_zscore_features(features: ArrayLike, *, reference_features: ArrayLike | None = None, copy: bool = True) -> NDArray[np.float64]:
    """Z-score columns using the subject's own feature distribution by default."""

    output = _feature_matrix(features, copy=copy)
    reference = output if reference_features is None else _feature_matrix(reference_features, copy=False)
    mean = np.mean(reference, axis=0, keepdims=True)
    std = nonzero_std(np.std(reference, axis=0, keepdims=True))
    output -= mean
    output /= std
    return output


def trial_zscore_features(features: ArrayLike, *, copy: bool = True) -> NDArray[np.float64]:
    """Z-score each trial row independently across its feature dimensions."""

    output = _feature_matrix(features, copy=copy)
    mean = np.mean(output, axis=1, keepdims=True)
    std = nonzero_std(np.std(output, axis=1, keepdims=True))
    output -= mean
    output /= std
    return output


def baseline_zscore_features(features: ArrayLike, baseline_feature_mean: ArrayLike, baseline_feature_std: ArrayLike, *, copy: bool = True) -> NDArray[np.float64]:
    """Z-score features using baseline-derived feature statistics."""

    output = _feature_matrix(features, copy=copy)
    mean = _row_vector(baseline_feature_mean, name="baseline_feature_mean")
    std = nonzero_std(_row_vector(baseline_feature_std, name="baseline_feature_std"))
    _require_matching_feature_width(output, mean, "baseline_feature_mean")
    _require_matching_feature_width(output, std, "baseline_feature_std")
    output -= mean
    output /= std
    return output


def normalize_subject_features(
    features: ArrayLike,
    normalization: str,
    *,
    baseline_feature_mean: ArrayLike | None = None,
    baseline_feature_std: ArrayLike | None = None,
    baseline_whitening_matrix: ArrayLike | None = None,
    feature_mode: str = "sensor_mean",
    reference_features: ArrayLike | None = None,
    copy: bool = True,
) -> NDArray[np.float64]:
    """Apply one of the supported subject-level feature normalizations."""

    normalization = normalize_normalization(normalization)
    if normalization == "none":
        return _feature_matrix(features, copy=copy)
    if normalization == "subject_z":
        return subject_zscore_features(features, reference_features=reference_features, copy=copy)
    if normalization == "subject_trial_z":
        return trial_zscore_features(features, copy=copy)
    if normalization == "subject_baseline_z":
        if baseline_feature_mean is None or baseline_feature_std is None:
            raise ValueError("subject_baseline_z requires baseline feature statistics.")
        return baseline_zscore_features(features, baseline_feature_mean, baseline_feature_std, copy=copy)
    if normalization == "subject_baseline_whiten":
        if baseline_feature_mean is None or baseline_whitening_matrix is None:
            raise ValueError("subject_baseline_whiten requires baseline feature statistics and a whitening matrix.")
        return baseline_whiten_features(
            features,
            baseline_feature_mean,
            baseline_whitening_matrix,
            feature_mode=feature_mode,
            copy=copy,
        )
    raise ValueError(f"Unsupported normalization: {normalization}")


def covariance_matrix(features: ArrayLike) -> NDArray[np.float64]:
    """Return a symmetric feature covariance matrix with robust single-sample handling."""

    feature_matrix = _feature_matrix(features, copy=False)
    n_features = int(feature_matrix.shape[1])
    if feature_matrix.shape[0] < 2:
        return np.eye(n_features, dtype=float)
    covariance = np.cov(feature_matrix, rowvar=False)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    return 0.5 * (covariance + covariance.T)


def shrink_covariance(covariance: ArrayLike, *, shrinkage: float = BASELINE_WHITENING_SHRINKAGE) -> NDArray[np.float64]:
    """Shrink covariance toward its diagonal while preserving feature variances."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    shrinkage = float(shrinkage)
    if shrinkage < 0.0 or shrinkage > 1.0 or not np.isfinite(shrinkage):
        raise ValueError("shrinkage must be finite and between 0 and 1.")
    diagonal = np.diag(np.diag(covariance))
    return (1.0 - shrinkage) * covariance + shrinkage * diagonal


def whitening_matrix(covariance: ArrayLike, *, eigenvalue_floor: float = BASELINE_WHITENING_EIGENVALUE_FLOOR) -> NDArray[np.float64]:
    """Return a symmetric inverse-square-root whitening matrix."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance must be a square matrix.")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    eigen_floor = max(float(np.max(eigenvalues)) * float(eigenvalue_floor), 1e-12)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, eigen_floor))
    matrix = (eigenvectors * inverse_sqrt) @ eigenvectors.T
    return 0.5 * (matrix + matrix.T)


def baseline_channel_whitening_matrix_from_features(
    baseline_features: ArrayLike,
    *,
    shrinkage: float = BASELINE_WHITENING_SHRINKAGE,
    eigenvalue_floor: float = BASELINE_WHITENING_EIGENVALUE_FLOOR,
) -> NDArray[np.float64]:
    """Fit a channel-space whitening matrix from baseline feature rows."""

    covariance = covariance_matrix(baseline_features)
    covariance = shrink_covariance(covariance, shrinkage=shrinkage)
    return whitening_matrix(covariance, eigenvalue_floor=eigenvalue_floor)


def baseline_whiten_features(
    features: ArrayLike,
    baseline_feature_mean: ArrayLike,
    baseline_whitening_matrix: ArrayLike,
    *,
    feature_mode: str = "sensor_mean",
    copy: bool = True,
) -> NDArray[np.float64]:
    """Center features by a baseline mean and apply channel-space whitening."""

    centered = _feature_matrix(features, copy=copy)
    mean = _row_vector(baseline_feature_mean, name="baseline_feature_mean")
    _require_matching_feature_width(centered, mean, "baseline_feature_mean")
    centered -= mean
    whitening = np.asarray(baseline_whitening_matrix, dtype=float)
    if whitening.ndim != 2 or whitening.shape[0] != whitening.shape[1]:
        raise ValueError("baseline_whitening_matrix must be a square matrix.")

    normalized_feature_mode = str(feature_mode).strip().lower().replace("-", "_")
    if normalized_feature_mode == "sensor_mean":
        if centered.shape[1] != whitening.shape[0]:
            raise ValueError("sensor_mean feature width must match the whitening matrix width.")
        return centered @ whitening.T
    if normalized_feature_mode in {
        "sensor_flat",
        "sensor_mean_slope",
        "sensor_mean_slope_std",
        "sensor_mean_slope_std_halves",
    }:
        return baseline_whiten_feature_blocks(centered, whitening, copy=False)
    raise ValueError(f"Unsupported feature_mode: {feature_mode}")


def baseline_whiten_feature_blocks(features: ArrayLike, whitening_matrix_: ArrayLike, *, copy: bool = True) -> NDArray[np.float64]:
    """Apply the same channel whitening matrix to each channel-sized feature block."""

    output = _feature_matrix(features, copy=copy)
    whitening = np.asarray(whitening_matrix_, dtype=float)
    if whitening.ndim != 2 or whitening.shape[0] != whitening.shape[1]:
        raise ValueError("whitening_matrix must be a square matrix.")
    n_channels = int(whitening.shape[0])
    if output.shape[1] % n_channels:
        raise ValueError("feature width must be a multiple of the number of whitening channels.")
    n_feature_blocks = int(output.shape[1] // n_channels)
    matrices = output.reshape(output.shape[0], n_feature_blocks, n_channels)
    whitened = matrices @ whitening.T
    return whitened.reshape(output.shape[0], -1)


def _feature_matrix(features: ArrayLike, *, copy: bool) -> NDArray[np.float64]:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("features must be a two-dimensional array.")
    return matrix.copy() if copy else matrix


def _row_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    vector = np.asarray(values, dtype=float)
    if vector.ndim == 1:
        vector = vector[None, :]
    if vector.ndim != 2 or vector.shape[0] != 1:
        raise ValueError(f"{name} must be a one-dimensional vector or a single-row matrix.")
    return vector


def _require_matching_feature_width(features: NDArray[np.float64], values: NDArray[np.float64], name: str) -> None:
    if features.shape[1] != values.shape[1]:
        raise ValueError(f"{name} width must match feature width.")

"""Reusable covariance feature extraction for M/EEG decoding workflows.

The helpers in this module convert trial-level ``channels x time`` arrays into
flat covariance-derived feature vectors. They are intentionally independent of
any dataset loader so project-specific repositories such as PyMEGDec can reuse
the same representations without importing BUSH-MEG command-line workflows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

COVARIANCE_FEATURE_MODES = (
    "logeuclidean_covariance",
    "covariance_upper",
    "correlation_upper",
    "variance",
)
DEFAULT_COVARIANCE_FEATURE_MODE = "logeuclidean_covariance"
DEFAULT_COVARIANCE_SHRINKAGE = 0.1
DEFAULT_COVARIANCE_EPSILON = 1.0e-6
DEFAULT_COVARIANCE_MAX_CHANNELS = 64


@dataclass(frozen=True, slots=True)
class CovarianceWindow:
    """Absolute-time window used for covariance feature extraction."""

    name: str
    start: float
    stop: float


def normalize_covariance_feature_mode(value: Any) -> str:
    """Normalize covariance-feature mode names and PyMEGDec aliases."""

    normalized = DEFAULT_COVARIANCE_FEATURE_MODE if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "logeig_covariance": "logeuclidean_covariance",
        "log_covariance": "logeuclidean_covariance",
        "covariance_logeuclidean": "logeuclidean_covariance",
        "covariance": "covariance_upper",
        "correlation": "correlation_upper",
        "diag_variance": "variance",
        "logvariance": "variance",
        "log_variance": "variance",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in COVARIANCE_FEATURE_MODES:
        raise ValueError(f"covariance_feature_mode must be one of {COVARIANCE_FEATURE_MODES}, got {value!r}.")
    return normalized


def normalize_covariance_shrinkage(value: Any) -> float:
    """Return a finite Ledoit-style diagonal shrinkage weight in ``[0, 1]``."""

    shrinkage = float(value)
    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("covariance_shrinkage must be a finite value in [0, 1].")
    return shrinkage


def normalize_covariance_epsilon(value: Any) -> float:
    """Return a positive eigenvalue/jitter floor factor."""

    epsilon = float(value)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("covariance_epsilon must be a positive finite value.")
    return epsilon


def sample_indices_for_window(times: Sequence[float] | np.ndarray, window: CovarianceWindow) -> np.ndarray:
    """Return inclusive sample indices overlapping ``window``."""

    times = np.asarray(times, dtype=float).reshape(-1)
    if times.size == 0 or not np.all(np.isfinite(times)):
        raise ValueError("times must contain at least one finite sample time.")
    if not np.isfinite([window.start, window.stop]).all() or window.stop < window.start:
        raise ValueError("CovarianceWindow stop must be finite and greater than or equal to start.")
    tolerance = 1e-12
    indices = np.flatnonzero((times >= float(window.start) - tolerance) & (times <= float(window.stop) + tolerance))
    if indices.size == 0:
        raise ValueError(
            f"Covariance window '{window.name}' [{window.start:.6g}, {window.stop:.6g}] "
            f"does not overlap available times [{times[0]:.6g}, {times[-1]:.6g}]."
        )
    return indices


def channel_subset_indices(n_channels: int, max_channels: int = DEFAULT_COVARIANCE_MAX_CHANNELS) -> np.ndarray:
    """Return deterministic channel indices capped at ``max_channels``."""

    n_channels = int(n_channels)
    max_channels = int(max_channels)
    if n_channels < 1:
        raise ValueError("n_channels must be positive.")
    if max_channels < 1:
        raise ValueError("max_channels must be positive.")
    if n_channels <= max_channels:
        return np.arange(n_channels, dtype=int)
    return np.unique(np.linspace(0, n_channels - 1, max_channels, dtype=int))


def trial_covariance(
    signal: Sequence[Sequence[float]] | np.ndarray,
    *,
    shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    epsilon: float = DEFAULT_COVARIANCE_EPSILON,
) -> np.ndarray:
    """Return a shrunken SPD covariance matrix for one ``channels x time`` trial."""

    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("Trial signal must be a channels x time matrix.")
    if signal.shape[0] < 1 or signal.shape[1] < 1:
        raise ValueError("Trial signal must contain at least one channel and one sample.")
    if not np.all(np.isfinite(signal)):
        raise ValueError("Trial signal must contain only finite values.")
    centered = signal - np.mean(signal, axis=1, keepdims=True)
    denominator = max(1, int(centered.shape[1]) - 1)
    covariance = (centered @ centered.T) / float(denominator)
    covariance = 0.5 * (covariance + covariance.T)
    n_channels = covariance.shape[0]
    trace_mean = float(np.trace(covariance) / max(1, n_channels))
    if not np.isfinite(trace_mean) or trace_mean <= 0.0:
        trace_mean = 1.0
    shrinkage = normalize_covariance_shrinkage(shrinkage)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * trace_mean * np.eye(n_channels)
    covariance = covariance + normalize_covariance_epsilon(epsilon) * trace_mean * np.eye(n_channels)
    return 0.5 * (covariance + covariance.T)


def eigen_floor(matrix: Sequence[Sequence[float]] | np.ndarray, epsilon: float = DEFAULT_COVARIANCE_EPSILON) -> float:
    """Return a scale-aware positive floor for covariance eigenvalues."""

    matrix = np.asarray(matrix, dtype=float)
    scale = float(np.trace(matrix) / max(1, matrix.shape[0]))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return normalize_covariance_epsilon(epsilon) * scale


def matrix_log_spd(matrix: Sequence[Sequence[float]] | np.ndarray, epsilon: float = DEFAULT_COVARIANCE_EPSILON) -> np.ndarray:
    """Return the log-Euclidean matrix logarithm of a symmetric SPD matrix."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix_log_spd expects a square matrix.")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    floor = eigen_floor(matrix, epsilon)
    log_values = np.log(np.maximum(eigenvalues, floor))
    return (eigenvectors * log_values[None, :]) @ eigenvectors.T


def covariance_to_correlation(covariance: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a symmetric correlation matrix."""

    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance_to_correlation expects a square matrix.")
    std = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
    correlation = covariance / np.outer(std, std)
    np.fill_diagonal(correlation, 1.0)
    return 0.5 * (correlation + correlation.T)


def vectorize_symmetric(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    *,
    scale_off_diagonal: bool = True,
) -> np.ndarray:
    """Vectorize the upper triangle of a symmetric matrix."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("vectorize_symmetric expects a square matrix.")
    rows, cols = np.triu_indices(matrix.shape[0])
    values = np.asarray(matrix[rows, cols], dtype=float)
    if scale_off_diagonal:
        values = values.copy()
        values[rows != cols] *= np.sqrt(2.0)
    return values


def covariance_feature_vector(
    signal: Sequence[Sequence[float]] | np.ndarray,
    mode: str = DEFAULT_COVARIANCE_FEATURE_MODE,
    *,
    shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    epsilon: float = DEFAULT_COVARIANCE_EPSILON,
) -> np.ndarray:
    """Return one PyMEGDec-compatible covariance feature vector."""

    mode = normalize_covariance_feature_mode(mode)
    covariance = trial_covariance(signal, shrinkage=shrinkage, epsilon=epsilon)
    if mode == "variance":
        return np.log(np.maximum(np.diag(covariance), eigen_floor(covariance, epsilon)))
    if mode == "correlation_upper":
        return vectorize_symmetric(covariance_to_correlation(covariance), scale_off_diagonal=True)
    if mode == "covariance_upper":
        return vectorize_symmetric(covariance, scale_off_diagonal=True)
    if mode == "logeuclidean_covariance":
        return vectorize_symmetric(matrix_log_spd(covariance, epsilon), scale_off_diagonal=True)
    raise ValueError(f"Unsupported covariance feature mode: {mode}")  # pragma: no cover - guarded above


def window_covariance_features(
    data: Sequence[Sequence[Sequence[float]]] | np.ndarray,
    times: Sequence[float] | np.ndarray,
    window: CovarianceWindow,
    *,
    mode: str = DEFAULT_COVARIANCE_FEATURE_MODE,
    shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    epsilon: float = DEFAULT_COVARIANCE_EPSILON,
    max_channels: int = DEFAULT_COVARIANCE_MAX_CHANNELS,
) -> np.ndarray:
    """Return trial x feature covariance representations for one time window."""

    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("Covariance features expect data shaped trials x channels x time.")
    if data.shape[0] < 1:
        raise ValueError("At least one trial is required for covariance features.")
    time_indices = sample_indices_for_window(times, window)
    channel_indices = channel_subset_indices(data.shape[1], max_channels)
    window_data = data[:, channel_indices][:, :, time_indices]
    rows = [
        covariance_feature_vector(
            window_data[trial_index],
            mode,
            shrinkage=shrinkage,
            epsilon=epsilon,
        )
        for trial_index in range(window_data.shape[0])
    ]
    return np.vstack(rows).astype(np.float32, copy=False)

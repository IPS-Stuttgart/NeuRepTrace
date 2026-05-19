"""Dataset-neutral epoch-window feature extraction utilities.

The helpers in this module operate on already-loaded M/EEG-like arrays with
shape ``(trials, channels, time)``.  Dataset-specific projects such as
PyMEGDec can keep their MATLAB/FieldTrip readers downstream and adapt their
loaded arrays into :class:`EpochArray`, while NeuRepTrace owns the reusable
windowing, flattening, baseline-window matching, and feature normalization
bookkeeping.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

FeatureOrder = Literal["channel_time", "time_channel"]
NormalizationMode = Literal["none", "center", "zscore"]
DEFAULT_FEATURE_ORDER: FeatureOrder = "channel_time"

__all__ = [
    "DEFAULT_FEATURE_ORDER",
    "EpochArray",
    "FeatureNormalizer",
    "FeatureOrder",
    "NormalizationMode",
    "ResolvedTimeWindow",
    "WindowedFeatureMatrix",
    "as_epoch_array",
    "extract_matching_baseline_features",
    "extract_window_features",
    "fit_feature_normalizer",
    "normalize_feature_order",
    "normalize_normalization_mode",
    "normalize_window_features",
    "resolve_time_window",
]


@dataclass(frozen=True)
class EpochArray:
    """In-memory trial array for generic M/EEG window extraction.

    Parameters
    ----------
    data
        Trial data with shape ``(n_trials, n_channels, n_times)``.
    times
        One-dimensional time vector in seconds with one entry per sample.
    labels
        Optional trial labels.  When provided, length must equal ``n_trials``.
    groups
        Optional group/session/run identifiers.  When provided, length must
        equal ``n_trials``.
    channel_names
        Optional channel names.  When provided, length must equal
        ``n_channels``.
    sensor_positions
        Optional sensor coordinates.  The first dimension must equal
        ``n_channels``; the remaining dimension is left unconstrained so CTF,
        Neuromag, or synthetic coordinate layouts can be represented.
    """

    data: np.ndarray
    times: np.ndarray
    labels: np.ndarray | None = None
    groups: np.ndarray | None = None
    channel_names: tuple[str, ...] | None = None
    sensor_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        data = np.asarray(self.data, dtype=float)
        if data.ndim != 3:
            raise ValueError("data must have shape (trials, channels, time).")
        if data.shape[0] == 0:
            raise ValueError("data must contain at least one trial.")
        if data.shape[1] == 0:
            raise ValueError("data must contain at least one channel.")
        if data.shape[2] == 0:
            raise ValueError("data must contain at least one time sample.")

        times = np.asarray(self.times, dtype=float)
        if times.ndim != 1:
            raise ValueError("times must be one-dimensional.")
        if times.size != data.shape[2]:
            raise ValueError(
                "times length must match the last data dimension: "
                f"{times.size} != {data.shape[2]}."
            )
        if times.size > 1 and np.any(np.diff(times) <= 0):
            raise ValueError("times must be strictly increasing.")

        labels = _optional_vector(self.labels, expected_length=data.shape[0], name="labels")
        groups = _optional_vector(self.groups, expected_length=data.shape[0], name="groups")

        channel_names = None
        if self.channel_names is not None:
            channel_names = tuple(str(name) for name in self.channel_names)
            if len(channel_names) != data.shape[1]:
                raise ValueError(
                    "channel_names length must match the channel dimension: "
                    f"{len(channel_names)} != {data.shape[1]}."
                )

        sensor_positions = None
        if self.sensor_positions is not None:
            sensor_positions = np.asarray(self.sensor_positions, dtype=float)
            if sensor_positions.ndim != 2:
                raise ValueError("sensor_positions must be a two-dimensional array.")
            if sensor_positions.shape[0] != data.shape[1]:
                raise ValueError(
                    "sensor_positions rows must match the channel dimension: "
                    f"{sensor_positions.shape[0]} != {data.shape[1]}."
                )

        object.__setattr__(self, "data", data)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "channel_names", channel_names)
        object.__setattr__(self, "sensor_positions", sensor_positions)

    @property
    def n_trials(self) -> int:
        """Number of trials."""
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return int(self.data.shape[1])

    @property
    def n_times(self) -> int:
        """Number of time samples."""
        return int(self.data.shape[2])


@dataclass(frozen=True)
class ResolvedTimeWindow:
    """Time-window bounds and selected sample indices.

    ``sample_start`` is inclusive and ``sample_stop`` is exclusive, matching
    Python slicing.  ``start`` and ``stop`` store the requested time bounds;
    ``center`` is the empirical mean of the selected sample times.
    """

    start: float
    stop: float
    center: float
    sample_start: int
    sample_stop: int

    @property
    def n_samples(self) -> int:
        """Number of selected samples."""
        return int(self.sample_stop - self.sample_start)

    @property
    def sample_slice(self) -> slice:
        """Slice selecting the window samples."""
        return slice(self.sample_start, self.sample_stop)


@dataclass(frozen=True)
class WindowedFeatureMatrix:
    """Flattened feature matrix extracted from an :class:`EpochArray`.

    The feature rows are trials.  Columns represent ``channels × samples`` in
    either ``channel_time`` order, matching common MNE flattening, or
    ``time_channel`` order, matching MATLAB/FieldTrip column-major flattening.
    """

    features: np.ndarray
    labels: np.ndarray | None
    groups: np.ndarray | None
    window: ResolvedTimeWindow
    n_channels: int
    n_window_samples: int
    feature_order: FeatureOrder = DEFAULT_FEATURE_ORDER
    channel_names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        features = _feature_matrix(self.features, name="features")
        order = normalize_feature_order(self.feature_order)
        n_channels = int(self.n_channels)
        n_window_samples = int(self.n_window_samples)
        if n_channels <= 0:
            raise ValueError("n_channels must be positive.")
        if n_window_samples <= 0:
            raise ValueError("n_window_samples must be positive.")
        expected_columns = n_channels * n_window_samples
        if features.shape[1] != expected_columns:
            raise ValueError(
                "features columns must equal n_channels * n_window_samples: "
                f"{features.shape[1]} != {expected_columns}."
            )
        labels = _optional_vector(self.labels, expected_length=features.shape[0], name="labels")
        groups = _optional_vector(self.groups, expected_length=features.shape[0], name="groups")
        channel_names = None if self.channel_names is None else tuple(str(name) for name in self.channel_names)
        if channel_names is not None and len(channel_names) != n_channels:
            raise ValueError(
                "channel_names length must match n_channels: "
                f"{len(channel_names)} != {n_channels}."
            )
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "feature_order", order)
        object.__setattr__(self, "channel_names", channel_names)


@dataclass(frozen=True)
class FeatureNormalizer:
    """Column-wise feature normalizer fitted on a reference matrix."""

    mode: NormalizationMode
    mean: np.ndarray | None
    scale: np.ndarray | None
    reference_rows: int

    def transform(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Normalize a feature matrix using the fitted reference statistics."""
        matrix = _feature_matrix(features, name="features")
        if self.mode == "none":
            return matrix.copy()
        if self.mean is None:
            raise ValueError("Normalizer mean is missing.")
        if matrix.shape[1] != self.mean.shape[0]:
            raise ValueError(
                "features columns must match fitted normalizer width: "
                f"{matrix.shape[1]} != {self.mean.shape[0]}."
            )
        transformed = matrix - self.mean
        if self.mode == "zscore":
            if self.scale is None:
                raise ValueError("Z-score normalizer scale is missing.")
            transformed = transformed / self.scale
        return transformed

    def fit_transform(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Normalize a matrix after checking it matches the fitted width."""
        return self.transform(features)


def as_epoch_array(
    data: EpochArray | Sequence[Sequence[Sequence[float]]] | np.ndarray,
    times: Sequence[float] | np.ndarray | None = None,
    *,
    labels: Sequence | np.ndarray | None = None,
    groups: Sequence | np.ndarray | None = None,
    channel_names: Sequence[str] | None = None,
    sensor_positions: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> EpochArray:
    """Return ``data`` as an :class:`EpochArray`.

    Passing an existing :class:`EpochArray` is a no-op.  Otherwise, ``times`` is
    required so loaders remain explicit about the temporal axis.
    """
    if isinstance(data, EpochArray):
        if any(value is not None for value in (times, labels, groups, channel_names, sensor_positions)):
            raise ValueError("Extra metadata cannot be supplied when data is already an EpochArray.")
        return data
    if times is None:
        raise ValueError("times must be provided when constructing an EpochArray from raw data.")
    return EpochArray(
        data=np.asarray(data, dtype=float),
        times=np.asarray(times, dtype=float),
        labels=None if labels is None else np.asarray(labels),
        groups=None if groups is None else np.asarray(groups),
        channel_names=None if channel_names is None else tuple(channel_names),
        sensor_positions=None if sensor_positions is None else np.asarray(sensor_positions, dtype=float),
    )


def resolve_time_window(
    times: Sequence[float] | np.ndarray,
    window: tuple[float, float] | list[float] | ResolvedTimeWindow | None = None,
    *,
    start: float | None = None,
    stop: float | None = None,
    center: float | None = None,
    size: float | None = None,
    name: str = "window",
) -> ResolvedTimeWindow:
    """Resolve requested time bounds to nearest inclusive sample indices.

    Windows may be provided either as explicit ``(start, stop)`` bounds or as a
    ``center``/``size`` pair.  The nearest samples to the requested bounds are
    selected and the stop index is made exclusive.
    """
    times_array = _time_vector(times)
    if isinstance(window, ResolvedTimeWindow):
        _validate_resolved_window_against_times(window, times_array, name=name)
        return window
    requested_start, requested_stop = _requested_bounds(
        window=window,
        start=start,
        stop=stop,
        center=center,
        size=size,
        name=name,
    )
    _require_window_supported(times_array, requested_start, requested_stop, name=name)
    sample_start = int(np.argmin(np.abs(times_array - requested_start)))
    sample_stop_inclusive = int(np.argmin(np.abs(times_array - requested_stop)))
    if sample_stop_inclusive < sample_start:
        raise ValueError(f"{name} is empty after nearest-sample resolution.")
    sample_stop = sample_stop_inclusive + 1
    selected_times = times_array[sample_start:sample_stop]
    return ResolvedTimeWindow(
        start=float(requested_start),
        stop=float(requested_stop),
        center=float(np.mean(selected_times)),
        sample_start=sample_start,
        sample_stop=sample_stop,
    )


def extract_window_features(
    epochs: EpochArray | Sequence[Sequence[Sequence[float]]] | np.ndarray,
    times: Sequence[float] | np.ndarray | None = None,
    *,
    labels: Sequence | np.ndarray | None = None,
    groups: Sequence | np.ndarray | None = None,
    channel_names: Sequence[str] | None = None,
    sensor_positions: Sequence[Sequence[float]] | np.ndarray | None = None,
    window: tuple[float, float] | list[float] | ResolvedTimeWindow | None = None,
    start: float | None = None,
    stop: float | None = None,
    center: float | None = None,
    size: float | None = None,
    feature_order: FeatureOrder = DEFAULT_FEATURE_ORDER,
) -> WindowedFeatureMatrix:
    """Extract a flattened feature matrix for one time window.

    ``epochs`` may be an :class:`EpochArray` or a raw ``(trials, channels,
    time)`` array.  The returned matrix is ready for downstream decoders while
    preserving labels, groups, channel count, selected sample count, and feature
    order metadata.
    """
    epoch_array = as_epoch_array(
        epochs,
        times,
        labels=labels,
        groups=groups,
        channel_names=channel_names,
        sensor_positions=sensor_positions,
    )
    resolved = resolve_time_window(
        epoch_array.times,
        window,
        start=start,
        stop=stop,
        center=center,
        size=size,
        name="window",
    )
    order = normalize_feature_order(feature_order)
    window_data = epoch_array.data[:, :, resolved.sample_slice]
    features = _flatten_window(window_data, feature_order=order)
    return WindowedFeatureMatrix(
        features=features,
        labels=epoch_array.labels,
        groups=epoch_array.groups,
        window=resolved,
        n_channels=epoch_array.n_channels,
        n_window_samples=resolved.n_samples,
        feature_order=order,
        channel_names=epoch_array.channel_names,
    )


def extract_matching_baseline_features(
    epochs: EpochArray | Sequence[Sequence[Sequence[float]]] | np.ndarray,
    target_window: WindowedFeatureMatrix | ResolvedTimeWindow,
    times: Sequence[float] | np.ndarray | None = None,
    *,
    baseline_start: float,
    labels: Sequence | np.ndarray | None = None,
    groups: Sequence | np.ndarray | None = None,
    channel_names: Sequence[str] | None = None,
    sensor_positions: Sequence[Sequence[float]] | np.ndarray | None = None,
    feature_order: FeatureOrder | None = None,
    require_disjoint: bool = True,
) -> WindowedFeatureMatrix:
    """Extract a baseline window with the same sample count as ``target_window``.

    The baseline start is resolved to its nearest supported sample.  The window
    then spans exactly ``target_window.n_samples`` samples, which is useful for
    PyMEGDec-like null-window comparisons and baseline z-scoring.
    """
    epoch_array = as_epoch_array(
        epochs,
        times,
        labels=labels,
        groups=groups,
        channel_names=channel_names,
        sensor_positions=sensor_positions,
    )
    if isinstance(target_window, WindowedFeatureMatrix):
        target_resolved = target_window.window
        n_samples = target_window.n_window_samples
        order = target_window.feature_order if feature_order is None else normalize_feature_order(feature_order)
    else:
        target_resolved = target_window
        n_samples = target_window.n_samples
        order = DEFAULT_FEATURE_ORDER if feature_order is None else normalize_feature_order(feature_order)

    sample_start, sample_stop = _matching_sample_slice(
        epoch_array.times,
        baseline_start,
        n_samples,
        name="baseline window",
    )
    if require_disjoint and _slices_overlap(sample_start, sample_stop, target_resolved.sample_start, target_resolved.sample_stop):
        raise ValueError("baseline window selects samples that overlap the target window.")

    selected_times = epoch_array.times[sample_start:sample_stop]
    resolved = ResolvedTimeWindow(
        start=float(selected_times[0]),
        stop=float(selected_times[-1]),
        center=float(np.mean(selected_times)),
        sample_start=sample_start,
        sample_stop=sample_stop,
    )
    features = _flatten_window(epoch_array.data[:, :, resolved.sample_slice], feature_order=order)
    return WindowedFeatureMatrix(
        features=features,
        labels=epoch_array.labels,
        groups=epoch_array.groups,
        window=resolved,
        n_channels=epoch_array.n_channels,
        n_window_samples=resolved.n_samples,
        feature_order=order,
        channel_names=epoch_array.channel_names,
    )


def fit_feature_normalizer(
    reference_features: Sequence[Sequence[float]] | np.ndarray,
    *,
    mode: NormalizationMode = "zscore",
    ddof: int = 0,
    eps: float = 1e-12,
) -> FeatureNormalizer:
    """Fit a column-wise feature normalizer on a reference matrix.

    Use ``mode='center'`` to subtract reference means and ``mode='zscore'`` to
    divide by reference standard deviations.  Constant or numerically degenerate
    columns receive scale ``1.0`` so transformed values remain finite.
    """
    mode = normalize_normalization_mode(mode)
    matrix = _feature_matrix(reference_features, name="reference_features")
    if mode == "none":
        return FeatureNormalizer(mode="none", mean=None, scale=None, reference_rows=matrix.shape[0])
    mean = np.mean(matrix, axis=0)
    scale = None
    if mode == "zscore":
        if ddof < 0:
            raise ValueError("ddof must be non-negative.")
        scale = np.std(matrix, axis=0, ddof=ddof)
        scale = np.where(np.isfinite(scale) & (scale > eps), scale, 1.0)
    return FeatureNormalizer(mode=mode, mean=mean, scale=scale, reference_rows=matrix.shape[0])


def normalize_window_features(
    features: Sequence[Sequence[float]] | np.ndarray,
    *,
    reference_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    mode: NormalizationMode = "zscore",
    ddof: int = 0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, FeatureNormalizer]:
    """Normalize ``features`` and return the fitted normalizer.

    When ``reference_features`` is provided, its column statistics are used.
    This supports baseline-window normalization without hardcoding a baseline
    convention into NeuRepTrace.
    """
    matrix = _feature_matrix(features, name="features")
    reference = matrix if reference_features is None else _feature_matrix(reference_features, name="reference_features")
    if reference.shape[1] != matrix.shape[1]:
        raise ValueError(
            "reference_features columns must match features columns: "
            f"{reference.shape[1]} != {matrix.shape[1]}."
        )
    normalizer = fit_feature_normalizer(reference, mode=mode, ddof=ddof, eps=eps)
    return normalizer.transform(matrix), normalizer


def normalize_feature_order(feature_order: FeatureOrder | str) -> FeatureOrder:
    """Normalize feature-order aliases."""
    normalized = str(feature_order).lower().replace("-", "_")
    if normalized in {"channel_time", "channels_time", "channel_sample", "channels_samples"}:
        return "channel_time"
    if normalized in {"time_channel", "time_channels", "sample_channel", "samples_channels"}:
        return "time_channel"
    raise ValueError("feature_order must be 'channel_time' or 'time_channel'.")


def normalize_normalization_mode(mode: NormalizationMode | str) -> NormalizationMode:
    """Normalize feature-normalization aliases."""
    normalized = str(mode).lower().replace("-", "_")
    if normalized in {"none", "identity", "raw"}:
        return "none"
    if normalized in {"center", "centered", "demean", "demeaned"}:
        return "center"
    if normalized in {"zscore", "z_score", "standard", "standardize", "standardized"}:
        return "zscore"
    raise ValueError("mode must be 'none', 'center', or 'zscore'.")


def _flatten_window(window_data: np.ndarray, *, feature_order: FeatureOrder) -> np.ndarray:
    if window_data.ndim != 3:
        raise ValueError("window_data must have shape (trials, channels, samples).")
    if feature_order == "channel_time":
        return np.asarray(window_data, dtype=float).reshape(window_data.shape[0], -1)
    return np.asarray(window_data, dtype=float).transpose(0, 2, 1).reshape(window_data.shape[0], -1)


def _requested_bounds(
    *,
    window: tuple[float, float] | list[float] | None,
    start: float | None,
    stop: float | None,
    center: float | None,
    size: float | None,
    name: str,
) -> tuple[float, float]:
    explicit_bounds = window is not None or start is not None or stop is not None
    center_size = center is not None or size is not None
    if explicit_bounds and center_size:
        raise ValueError(f"{name} must be defined by bounds or center/size, not both.")
    if window is not None:
        if len(window) != 2:
            raise ValueError(f"{name} must contain exactly two bounds.")
        start, stop = float(window[0]), float(window[1])
    elif start is not None or stop is not None:
        if start is None or stop is None:
            raise ValueError(f"{name} requires both start and stop.")
        start, stop = float(start), float(stop)
    else:
        if center is None or size is None:
            raise ValueError(f"{name} requires either start/stop or center/size.")
        center = float(center)
        size = float(size)
        if not np.isfinite(size) or size <= 0.0:
            raise ValueError(f"{name} size must be finite and positive.")
        start, stop = center - size / 2.0, center + size / 2.0
    if not (np.isfinite(start) and np.isfinite(stop)):
        raise ValueError(f"{name} bounds must be finite.")
    if stop < start:
        raise ValueError(f"{name} stop must be greater than or equal to start.")
    return start, stop


def _optional_vector(value: Sequence | np.ndarray | None, *, expected_length: int, name: str) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.asarray(value).ravel()
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if len(vector) != expected_length:
        raise ValueError(f"{name} length must match trials: {len(vector)} != {expected_length}.")
    return vector


def _feature_matrix(value: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one column.")
    return matrix


def _time_vector(times: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(times, dtype=float)
    if vector.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    if vector.size == 0:
        raise ValueError("times must contain at least one sample.")
    if vector.size > 1 and np.any(np.diff(vector) <= 0):
        raise ValueError("times must be strictly increasing.")
    return vector


def _validate_resolved_window_against_times(window: ResolvedTimeWindow, times: np.ndarray, *, name: str) -> None:
    if window.sample_start < 0 or window.sample_stop > times.size or window.sample_stop <= window.sample_start:
        raise ValueError(f"{name} sample indices are outside the supplied times vector.")


def _require_window_supported(times: np.ndarray, start: float, stop: float, *, name: str) -> None:
    _require_time_supported(times, start, name=name)
    _require_time_supported(times, stop, name=name)


def _require_time_supported(times: np.ndarray, value: float, *, name: str) -> None:
    tolerance = _time_support_tolerance(times)
    if value < times[0] - tolerance or value > times[-1] + tolerance:
        raise ValueError(f"{name} is outside the time support.")


def _time_support_tolerance(times: np.ndarray) -> float:
    if times.size < 2:
        return 1e-12
    return 0.5 * float(np.median(np.diff(times))) + 1e-12


def _matching_sample_slice(times: np.ndarray, start: float, sample_count: int, *, name: str) -> tuple[int, int]:
    if sample_count <= 0:
        raise ValueError(f"{name} sample count must be positive.")
    _require_time_supported(times, float(start), name=name)
    sample_start = int(np.argmin(np.abs(times - float(start))))
    sample_stop = sample_start + int(sample_count)
    if sample_stop > times.size:
        raise ValueError(f"{name} extends beyond the time support.")
    return sample_start, sample_stop


def _slices_overlap(start_a: int, stop_a: int, start_b: int, stop_b: int) -> bool:
    return start_a < stop_b and start_b < stop_a

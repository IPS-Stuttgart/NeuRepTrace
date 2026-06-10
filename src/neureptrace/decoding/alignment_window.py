"""Feature-window adaptation helpers for cross-window alignment projections.

These utilities support decoding workflows that fit an alignment projection on
one feature window, then apply that projection to features extracted from a
possibly different decoding window. When the feature widths differ, the
projection and centering vector can be collapsed to channel space and reused
across the decoding window samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

FeatureOrder = Literal["channel_time", "time_channel"]
DEFAULT_FEATURE_ORDER: FeatureOrder = "channel_time"


class WindowedFeatureSet(Protocol):
    """Minimal feature-set interface needed for alignment-window adaptation.

    Flattened MNE epoch arrays use the default ``channel_time`` layout because
    ``data[:, :, start:stop].reshape(n_trials, -1)`` stores all time samples of
    channel 0 first, then all time samples of channel 1, and so on. Legacy or
    synthetic feature sets that are flattened as ``[t0c0, t0c1, t1c0, ...]`` can
    set ``feature_order = "time_channel"``.
    """

    features: np.ndarray
    labels: np.ndarray
    n_channels: int
    n_window_samples: int
    # Concrete feature sets may also expose timing metadata such as
    # window_center/window_size or window_start/window_stop variants.


@dataclass(frozen=True)
class AlignmentWindow:
    """Resolved alignment-window parameters."""

    center: float
    size: float

    @property
    def start(self) -> float:
        """Window start time, using center-size convention."""

        return self.center - self.size / 2.0

    @property
    def stop(self) -> float:
        """Window stop time, using center-size convention."""

        return self.center + self.size / 2.0


def resolved_alignment_window(config) -> AlignmentWindow:
    """Return explicit alignment-window values, defaulting to the decoding window.

    The ``config`` object is expected to expose ``window_center`` and
    ``window_size`` attributes. Optional ``alignment_window_center`` and
    ``alignment_window_size`` attributes override the decoding window when they
    are not ``None``.
    """

    center = config.window_center if getattr(config, "alignment_window_center", None) is None else config.alignment_window_center
    size = config.window_size if getattr(config, "alignment_window_size", None) is None else config.alignment_window_size
    return AlignmentWindow(center=float(center), size=float(size))


def uses_separate_alignment_window(config) -> bool:
    """Return whether alignment and decoding windows differ."""

    alignment_window = resolved_alignment_window(config)
    return not (np.isclose(alignment_window.center, float(config.window_center)) and np.isclose(alignment_window.size, float(config.window_size)))


def validate_paired_feature_sets(decode_set: WindowedFeatureSet, alignment_set: WindowedFeatureSet, *, participant: int | None = None) -> None:
    """Validate that two feature sets refer to the same trial rows.

    The decoding and alignment feature matrices may have different column counts
    because they can represent different windows. They must, however, have the
    same row count, labels, and number of channels.
    """

    if decode_set.features.shape[0] != alignment_set.features.shape[0]:
        context = "" if participant is None else f" for participant {participant}"
        raise ValueError(f"Decoding and alignment feature rows differ{context}.")
    if not np.array_equal(np.asarray(decode_set.labels), np.asarray(alignment_set.labels)):
        context = "" if participant is None else f" for participant {participant}"
        raise ValueError(f"Decoding and alignment labels differ{context}.")
    if int(decode_set.n_channels) != int(alignment_set.n_channels):
        context = "" if participant is None else f" for participant {participant}"
        raise ValueError(f"Decoding and alignment channel counts differ{context}.")


def transform_with_alignment_projection(
    features: np.ndarray,
    *,
    decode_feature_set: WindowedFeatureSet,
    projection: np.ndarray,
    projection_feature_mean: np.ndarray,
    projection_feature_set: WindowedFeatureSet,
    feature_mean: np.ndarray | None = None,
    feature_mean_set: WindowedFeatureSet | None = None,
) -> np.ndarray:
    """Apply an alignment projection to features from a possibly different window.

    When feature widths and window metadata match, this is the standard centered
    linear projection. When widths differ, or when timing metadata show that the
    projection was fitted on a different same-width window, the projection and
    centering vector are collapsed to channel space by averaging across the
    alignment-window samples, then applied independently to each decoding-window
    sample.
    """

    matrix = _feature_matrix(features, name="features")
    projection = _feature_matrix(projection, name="projection")
    projection_mean = np.asarray(projection_feature_mean, dtype=float).ravel()
    feature_mean_was_explicit = feature_mean is not None
    mean = projection_mean if not feature_mean_was_explicit else np.asarray(feature_mean, dtype=float).ravel()
    mean_set = _resolve_feature_mean_set(
        mean,
        decode_feature_set=decode_feature_set,
        projection_feature_set=projection_feature_set,
        explicit_feature_mean_set=(
            feature_mean_set
            if feature_mean_was_explicit
            else projection_feature_set
        ),
        feature_mean_was_explicit=feature_mean_was_explicit,
    )

    if matrix.shape[1] == projection.shape[0] and _feature_windows_match(
        decode_feature_set,
        projection_feature_set,
    ):
        if mean.shape[0] != matrix.shape[1]:
            raise ValueError(f"feature_mean length must match features columns: {mean.shape[0]} != {matrix.shape[1]}.")
        return (matrix - mean) @ projection

    channel_projection = _projection_to_channel_space(projection, projection_feature_set)
    channel_mean = _feature_mean_to_channel_space(mean, mean_set)
    return _apply_channel_projection(matrix, decode_feature_set, channel_projection, channel_mean)


def _resolve_feature_mean_set(
    mean: np.ndarray,
    *,
    decode_feature_set: WindowedFeatureSet,
    projection_feature_set: WindowedFeatureSet,
    explicit_feature_mean_set: WindowedFeatureSet | None,
    feature_mean_was_explicit: bool,
) -> WindowedFeatureSet:
    """Infer which feature window a supplied centering vector belongs to.

    Cross-window alignment can apply a projection fitted on an alignment window to
    features from a different decoding window.  Callers may pass an unsupervised
    target mean via ``feature_mean``; historically, if ``feature_mean_set`` was
    omitted we assumed that mean belonged to the decoding window.  That is unsafe:
    target-calibrated alignment commonly estimates the mean from the alignment
    rows/window, and interpreting an alignment-window mean as a decode-window mean
    can either raise a false shape error or collapse channels with the wrong time
    geometry.  Prefer the explicit set; otherwise infer from vector length.
    """

    if explicit_feature_mean_set is not None:
        return explicit_feature_mean_set

    mean_width = int(np.asarray(mean).shape[0])
    projection_widths = _compatible_feature_mean_widths(projection_feature_set)
    decode_widths = _compatible_feature_mean_widths(decode_feature_set)
    projection_compatible = mean_width in projection_widths
    decode_compatible = mean_width in decode_widths
    projection_channel_width = int(projection_feature_set.n_channels)
    decode_channel_width = int(decode_feature_set.n_channels)
    if mean_width == projection_channel_width == decode_channel_width:
        # A channel-space centering vector is already independent of the time
        # window.  Cross-window alignment collapses both full-window means and
        # projections to channel space before scoring, so requiring callers to
        # provide ``feature_mean_set`` for an explicit channel mean creates a
        # false ambiguity and can block otherwise valid target-calibrated
        # alignment adapters.
        return projection_feature_set

    if (
        feature_mean_was_explicit
        and projection_compatible
        and decode_compatible
        and not _feature_windows_match(decode_feature_set, projection_feature_set)
    ):
        raise ValueError(
            "feature_mean is compatible with both the projection/alignment window and the decoding window. "
            "Pass feature_mean_set explicitly so cross-window alignment does not silently center with the wrong time window."
        )
    if projection_compatible:
        return projection_feature_set
    if decode_compatible:
        return decode_feature_set
    # Return the projection set so the eventual shape error reports the alignment
    # geometry that the projection was fitted with.
    return projection_feature_set


def _compatible_feature_mean_widths(feature_set: WindowedFeatureSet) -> set[int]:
    n_channels = int(feature_set.n_channels)
    n_window_samples = int(feature_set.n_window_samples)
    return {n_channels, n_channels * n_window_samples}


def _feature_matrix(value: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    return matrix


def _feature_order(feature_set: WindowedFeatureSet) -> FeatureOrder:
    order = getattr(feature_set, "feature_order", DEFAULT_FEATURE_ORDER)
    if order not in {"channel_time", "time_channel"}:
        raise ValueError(f"feature_order must be 'channel_time' or 'time_channel', got {order!r}.")
    return order


def _feature_windows_match(left: WindowedFeatureSet, right: WindowedFeatureSet) -> bool:
    """Return whether two flattened feature sets refer to the same time window."""

    if left is right:
        return True

    if int(left.n_channels) != int(right.n_channels) or int(left.n_window_samples) != int(
        right.n_window_samples
    ):
        return False
    if _feature_order(left) != _feature_order(right):
        return False

    left_window = _feature_window_bounds(left)
    right_window = _feature_window_bounds(right)
    # Legacy/smoke feature-set objects often omit timing metadata. When shape
    # and flattening order match and neither side advertises a window, keep the
    # exact fitted projection instead of collapsing to channel space.
    if left_window is None and right_window is None:
        return True
    if left_window is None or right_window is None:
        return False
    return np.allclose(left_window, right_window)


def _feature_window_bounds(feature_set: WindowedFeatureSet) -> tuple[float, float] | None:
    """Return normalized ``(start, stop)`` timing metadata when present."""

    start = _first_present_float(feature_set, ("window_start_s", "window_start"))
    stop = _first_present_float(feature_set, ("window_stop_s", "window_stop"))
    if start is not None or stop is not None:
        if start is None or stop is None:
            return None
        return (start, stop)

    center = _first_present_float(feature_set, ("window_center_s", "window_center"))
    size = _first_present_float(feature_set, ("window_size_s", "window_size"))
    if center is not None or size is not None:
        if center is None or size is None:
            return None
        return (center - size / 2.0, center + size / 2.0)

    return None


def _first_present_float(feature_set: WindowedFeatureSet, attrs: tuple[str, ...]) -> float | None:
    for attr in attrs:
        value = getattr(feature_set, attr, None)
        if value is not None:
            return float(value)
    return None


def _projection_to_channel_space(projection: np.ndarray, feature_set: WindowedFeatureSet) -> np.ndarray:
    n_channels = int(feature_set.n_channels)
    if projection.shape[0] == n_channels:
        return projection
    n_window_samples = int(feature_set.n_window_samples)
    expected = n_window_samples * n_channels
    if projection.shape[0] != expected:
        raise ValueError(f"Projection rows are incompatible with the alignment feature shape: {projection.shape[0]} != {expected}.")
    if _feature_order(feature_set) == "channel_time":
        return projection.reshape(n_channels, n_window_samples, projection.shape[1]).mean(axis=1)
    return projection.reshape(n_window_samples, n_channels, projection.shape[1]).mean(axis=0)


def _feature_mean_to_channel_space(mean: np.ndarray, feature_set: WindowedFeatureSet) -> np.ndarray:
    n_channels = int(feature_set.n_channels)
    if mean.shape[0] == n_channels:
        return mean
    n_window_samples = int(feature_set.n_window_samples)
    expected = n_window_samples * n_channels
    if mean.shape[0] != expected:
        raise ValueError(f"Feature mean is incompatible with the feature shape: {mean.shape[0]} != {expected}.")
    if _feature_order(feature_set) == "channel_time":
        return mean.reshape(n_channels, n_window_samples).mean(axis=1)
    return mean.reshape(n_window_samples, n_channels).mean(axis=0)


def _features_to_time_channel(matrix: np.ndarray, feature_set: WindowedFeatureSet) -> np.ndarray:
    n_channels = int(feature_set.n_channels)
    n_window_samples = int(feature_set.n_window_samples)
    expected = n_window_samples * n_channels
    if matrix.shape[1] != expected:
        raise ValueError(f"Feature columns are incompatible with the decoding feature shape: {matrix.shape[1]} != {expected}.")
    if _feature_order(feature_set) == "channel_time":
        return matrix.reshape(matrix.shape[0], n_channels, n_window_samples).transpose(0, 2, 1)
    return matrix.reshape(matrix.shape[0], n_window_samples, n_channels)


def _apply_channel_projection(matrix: np.ndarray, feature_set: WindowedFeatureSet, channel_projection: np.ndarray, channel_mean: np.ndarray) -> np.ndarray:
    n_channels = int(feature_set.n_channels)
    if channel_projection.shape[0] != n_channels:
        raise ValueError(
            "Channel-space projection rows must match the feature-set channel count: "
            f"{channel_projection.shape[0]} != {n_channels}."
        )
    if channel_mean.shape[0] != n_channels:
        raise ValueError(
            "Channel-space feature mean length must match the feature-set channel count: "
            f"{channel_mean.shape[0]} != {n_channels}."
        )

    if matrix.shape[1] == n_channels:
        return (matrix - channel_mean) @ channel_projection
    trial_channel = _features_to_time_channel(matrix, feature_set)
    transformed = (trial_channel - channel_mean[None, None, :]) @ channel_projection
    if _feature_order(feature_set) == "channel_time":
        # Preserve the input flattening convention.  `_features_to_time_channel`
        # converts both supported layouts to (trial, time, channel) for the
        # linear algebra above.  MNE-style flattened features are channel-major,
        # so the aligned component axis must be moved back in front of time
        # before flattening.  Otherwise same-window and cross-window aligned
        # features silently use different feature orders.
        return transformed.transpose(0, 2, 1).reshape(matrix.shape[0], -1)
    return transformed.reshape(matrix.shape[0], -1)

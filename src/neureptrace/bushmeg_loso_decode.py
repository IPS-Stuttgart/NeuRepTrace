"""Memory-bounded source-only LOSO decoding for BUSH-MEG main-task files.

The generic time-resolved decoders are intentionally dataset-independent.  This
module is a thin BUSH-MEG runner for the strict comparison that matters for the
Part*Data.mat files: train only on source-subject main-task trials and evaluate
single held-out-subject main-task trials.  Cue files are not read.

The implementation is deliberately conservative about memory.  Each subject is
loaded, cropped, normalized, and written to a small cache once.  LOSO folds then
open those cached arrays with mmap and build only the requested time-window
feature matrix.  The default training representation uses source-subject
class-balanced pseudo-trials, which is often a better cross-subject visual MEG
bias/variance trade-off than fitting on every noisy single source trial.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from neureptrace.dataset_config import parse_participant_ids
from neureptrace.decoding import (
    EMISSION_MODE_CHOICES,
    FEATURE_PREPROCESSOR_CHOICES,
    make_decoder,
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    normalize_pca_components,
    predict_emission_probabilities,
    time_windows,
)
from neureptrace.fieldtrip_mat import DEFAULT_ROOT_PATH, load_fieldtrip_raw_mat, parse_path_tokens
from neureptrace.metrics import brier_score_multiclass, expected_calibration_error
from neureptrace.mne_time_decode import (
    DEFAULT_BASELINE_WINDOW,
    EPOCH_NORMALIZATION_RUN_CHOICES,
    RESULT_SELECTION_METRIC_CHOICES,
    _align_probability_columns,
    _apply_epoch_normalization,
    _normalize_baseline_window,
    _top_k_accuracy,
    normalize_epoch_normalization,
)
from neureptrace.observations import stable_hash

DEFAULT_BUSHMEG_PARTICIPANTS = "1-4,6,8-10,13-27"
DEFAULT_MAIN_TEMPLATE = "Part{participant}Data.mat"
DEFAULT_DECODE_WINDOW = (0.125, 0.225)
DEFAULT_DECODERS = ("logistic", "linear_svm", "correlation-prototype")
WINDOW_FEATURE_MODE_CHOICES = (
    "sensor_flat",
    "bin_means",
    "mean_slope",
    "dct",
    "stats",
    "sensor_flat_plus_stats",
)
DEFAULT_WINDOW_FEATURE_MODE = "sensor_flat"
DEFAULT_TEMPORAL_BINS = 4
EPSILON = 1e-12
PROBABILITY_SUM_TOLERANCE = 1.0e-3
TimeWindow = tuple[int, int, float]


@dataclass(frozen=True)
class CachedSubject:
    """Cached, cropped, subject-normalized BUSH-MEG main-task data."""

    participant: str
    source_path: Path
    data_path: Path
    labels_path: Path
    times_path: Path


def _participant_tokens(value: str | Iterable[str | int] | None) -> tuple[str, ...]:
    raw = DEFAULT_BUSHMEG_PARTICIPANTS if value is None else value
    parsed = parse_participant_ids(raw)
    if not parsed:
        raise ValueError("At least one participant is required for LOSO decoding.")
    return tuple(str(participant) for participant in parsed)


def _split_csv_tokens(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",")]
    else:
        tokens = [str(token).strip() for token in value]
    tokens = [token for token in tokens if token]
    if not tokens:
        raise ValueError("At least one decoder is required.")
    return tuple(tokens)


def _decode_window_tuple(value: Sequence[float] | None) -> tuple[float, float] | None:
    if value is None:
        return DEFAULT_DECODE_WINDOW
    if len(value) != 2:
        raise ValueError("decode_window must contain exactly START and STOP.")
    start, stop = map(float, value)
    if stop < start:
        raise ValueError("decode_window stop must be greater than or equal to start.")
    return start, stop


def normalize_window_feature_mode(value: str | None) -> str:
    """Normalize source-only BUSH-MEG window feature representations."""

    normalized = DEFAULT_WINDOW_FEATURE_MODE if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "flat": "sensor_flat",
        "sensor": "sensor_flat",
        "sensor_flattened": "sensor_flat",
        "sensorflat": "sensor_flat",
        "raw": "sensor_flat",
        "bin_mean": "bin_means",
        "binmeans": "bin_means",
        "evoked": "bin_means",
        "evoked_bin_means": "bin_means",
        "temporal_bin_means": "bin_means",
        "slope": "mean_slope",
        "temporal_slope": "mean_slope",
        "evoked_slope": "mean_slope",
        "mean_slope": "mean_slope",
        "meanslope": "mean_slope",
        "temporal_dct": "dct",
        "evoked_dct": "dct",
        "dct_coefficients": "dct",
        "summary": "stats",
        "summaries": "stats",
        "temporal_stats": "stats",
        "evoked_stats": "stats",
        "statistical_summary": "stats",
        "flat_stats": "sensor_flat_plus_stats",
        "flat_plus_stats": "sensor_flat_plus_stats",
        "sensor_flat_stats": "sensor_flat_plus_stats",
        "sensor_flat_plus_summary": "sensor_flat_plus_stats",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in WINDOW_FEATURE_MODE_CHOICES:
        raise ValueError(
            f"Unknown window feature mode '{value}'. Available modes: {', '.join(WINDOW_FEATURE_MODE_CHOICES)}."
        )
    return normalized


def _normalize_temporal_bins(value: int | str) -> int:
    try:
        bins = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("temporal_bins must be a positive integer.") from exc
    if bins < 1:
        raise ValueError("temporal_bins must be a positive integer.")
    return bins


def _crop_time_axis(data: np.ndarray, times: np.ndarray, *, tmin: float | None, tmax: float | None) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(times.shape[0], dtype=bool)
    if tmin is not None:
        mask &= times >= float(tmin) - EPSILON
    if tmax is not None:
        mask &= times <= float(tmax) + EPSILON
    if not np.any(mask):
        raise ValueError(f"Crop [{tmin}, {tmax}] does not overlap the subject time axis.")
    return data[:, :, mask], times[mask]


def _metadata_labels(metadata: pd.DataFrame, label_column: str) -> np.ndarray:
    if label_column not in metadata.columns:
        raise ValueError(f"Label column '{label_column}' not found in FieldTrip metadata.")
    labels = metadata[label_column].to_numpy()
    if labels.dtype == object:
        labels = labels.astype(str)
    keep = pd.notna(labels)
    if not np.all(keep):
        raise ValueError("BUSH-MEG LOSO decoding requires non-missing labels for every trial.")
    return labels


def _cache_key(
    *,
    source_path: Path,
    participant: str,
    fieldtrip_root_path: tuple[str | int, ...],
    label_column: str,
    label_base: float | None,
    trialinfo_column: int,
    tmin: float | None,
    tmax: float | None,
    normalization: str,
    baseline_window: tuple[float, float],
    trim_overlong_labels: bool,
) -> str:
    stat = source_path.stat()
    return stable_hash(
        {
            "source_path": str(source_path.resolve()),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "participant": participant,
            "fieldtrip_root_path": fieldtrip_root_path,
            "label_column": label_column,
            "label_base": label_base,
            "trialinfo_column": trialinfo_column,
            "tmin": tmin,
            "tmax": tmax,
            "normalization": normalization,
            "baseline_window": baseline_window,
            "trim_overlong_labels": trim_overlong_labels,
        }
    ).replace(":", "_")


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    np.save(temporary, values)
    written = temporary.with_suffix(temporary.suffix + ".npy")
    if not written.exists():  # np.save keeps the suffix only when it already ends with .npy
        written = temporary
    written.replace(path)


def _write_cache_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _prepare_subject_cache(
    *,
    participant: str,
    source_path: Path,
    cache_dir: Path,
    fieldtrip_root_path: tuple[str | int, ...],
    label_column: str,
    label_base: float | None,
    trialinfo_column: int,
    tmin: float | None,
    tmax: float | None,
    normalization: str,
    baseline_window: tuple[float, float],
    trim_overlong_labels: bool,
) -> CachedSubject:
    if not source_path.exists():
        raise FileNotFoundError(f"BUSH-MEG main-task file not found: {source_path}")

    normalization = normalize_epoch_normalization(normalization)
    key = _cache_key(
        source_path=source_path,
        participant=participant,
        fieldtrip_root_path=fieldtrip_root_path,
        label_column=label_column,
        label_base=label_base,
        trialinfo_column=trialinfo_column,
        tmin=tmin,
        tmax=tmax,
        normalization=normalization,
        baseline_window=baseline_window,
        trim_overlong_labels=trim_overlong_labels,
    )
    prefix = cache_dir / f"participant-{participant}_{key}"
    data_path = prefix.with_suffix(".data.npy")
    labels_path = prefix.with_suffix(".labels.npy")
    times_path = prefix.with_suffix(".times.npy")
    metadata_path = prefix.with_suffix(".json")
    if data_path.exists() and labels_path.exists() and times_path.exists():
        return CachedSubject(participant=participant, source_path=source_path, data_path=data_path, labels_path=labels_path, times_path=times_path)

    fieldtrip = load_fieldtrip_raw_mat(
        source_path,
        root_path=fieldtrip_root_path,
        label_column=label_column,
        label_base=label_base,
        trialinfo_column=trialinfo_column,
        trim_overlong_labels=trim_overlong_labels,
    )
    data = np.asarray(fieldtrip.trials, dtype=np.float32)
    times = np.asarray(fieldtrip.times[0], dtype=float)
    data, times = _crop_time_axis(data, times, tmin=tmin, tmax=tmax)
    data = _apply_epoch_normalization(data, times, normalization, baseline_window=baseline_window).astype(np.float32, copy=False)
    labels = _metadata_labels(fieldtrip.metadata, label_column)
    _atomic_save_npy(data_path, data)
    _atomic_save_npy(labels_path, labels)
    _atomic_save_npy(times_path, times)
    _write_cache_metadata(
        metadata_path,
        {
            "participant": participant,
            "source_path": str(source_path),
            "shape": list(data.shape),
            "n_labels": int(labels.shape[0]),
            "time_start": float(times[0]),
            "time_stop": float(times[-1]),
            "normalization": normalization,
            "baseline_window": baseline_window,
        },
    )
    return CachedSubject(participant=participant, source_path=source_path, data_path=data_path, labels_path=labels_path, times_path=times_path)


def _subject_data(subject: CachedSubject) -> np.ndarray:
    return np.load(subject.data_path, mmap_mode="r", allow_pickle=False)


def _subject_labels(subject: CachedSubject) -> np.ndarray:
    return np.load(subject.labels_path, allow_pickle=False)


def _subject_times(subject: CachedSubject) -> np.ndarray:
    return np.load(subject.times_path, allow_pickle=False)


def _window_sample_indices(window: TimeWindow) -> np.ndarray:
    start, stop, _center = window
    if stop <= start:
        raise ValueError("Time windows must contain at least one sample.")
    return np.arange(int(start), int(stop), dtype=int)


def _split_window_indices(window: TimeWindow, temporal_bins: int) -> list[np.ndarray]:
    bins = _normalize_temporal_bins(temporal_bins)
    indices = _window_sample_indices(window)
    split = [np.asarray(bin_indices, dtype=int) for bin_indices in np.array_split(indices, bins)]
    if any(bin_indices.size == 0 for bin_indices in split):
        raise ValueError(
            f"Window has only {indices.size} samples, not enough for {bins} temporal bins."
        )
    return split


def _window_times_for_indices(times: np.ndarray | None, indices: np.ndarray) -> np.ndarray:
    if times is None:
        return indices.astype(float, copy=False)
    time_values = np.asarray(times, dtype=float)
    if time_values.ndim != 1:
        raise ValueError("times must be a one-dimensional array when extracting temporal window features.")
    if int(indices[-1]) >= time_values.shape[0]:
        raise ValueError("Window indices extend past the provided time axis.")
    return time_values[indices]


def _linear_temporal_contrast(segment: np.ndarray, bin_times: np.ndarray) -> np.ndarray:
    if segment.shape[2] < 2:
        return np.zeros(segment.shape[:2], dtype=np.float64)
    weights = np.asarray(bin_times, dtype=np.float64) - float(np.mean(bin_times))
    norm = float(np.linalg.norm(weights))
    if norm <= EPSILON:
        return np.zeros(segment.shape[:2], dtype=np.float64)
    weights = weights / norm
    return np.tensordot(segment, weights, axes=([2], [0]))


def _sensor_flat_features(data: np.ndarray, window: TimeWindow) -> np.ndarray:
    start, stop, _center = window
    # Materialize only one window.  mmap slices can be non-contiguous and many
    # sklearn estimators copy them anyway, so an explicit float32 matrix keeps
    # peak memory predictable.
    return np.asarray(data[:, :, start:stop].reshape(data.shape[0], -1), dtype=np.float32)


def _window_bin_mean_features(data: np.ndarray, window: TimeWindow, *, temporal_bins: int) -> np.ndarray:
    features: list[np.ndarray] = []
    for bin_indices in _split_window_indices(window, temporal_bins):
        segment = np.asarray(data[:, :, bin_indices], dtype=np.float64)
        features.append(segment.mean(axis=2))
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _window_mean_slope_features(
    data: np.ndarray,
    window: TimeWindow,
    *,
    times: np.ndarray | None,
    temporal_bins: int,
) -> np.ndarray:
    mean_features: list[np.ndarray] = []
    slope_features: list[np.ndarray] = []
    for bin_indices in _split_window_indices(window, temporal_bins):
        segment = np.asarray(data[:, :, bin_indices], dtype=np.float64)
        mean_features.append(segment.mean(axis=2))
        slope_features.append(_linear_temporal_contrast(segment, _window_times_for_indices(times, bin_indices)))
    return np.concatenate([*mean_features, *slope_features], axis=1).astype(np.float32, copy=False)


def _window_dct_features(data: np.ndarray, window: TimeWindow, *, temporal_bins: int) -> np.ndarray:
    n_coefficients = _normalize_temporal_bins(temporal_bins)
    start, stop, _center = window
    n_samples = int(stop) - int(start)
    if n_samples < n_coefficients:
        raise ValueError(
            f"Window has only {n_samples} samples, not enough for {n_coefficients} DCT coefficients."
        )
    segment = np.asarray(data[:, :, start:stop], dtype=np.float64)
    coefficients = dct(segment, type=2, norm="ortho", axis=2)[:, :, :n_coefficients]
    return (
        np.transpose(coefficients, (0, 2, 1))
        .reshape(segment.shape[0], -1)
        .astype(np.float32, copy=False)
    )


def _window_stat_features(
    data: np.ndarray,
    window: TimeWindow,
    *,
    times: np.ndarray | None,
    temporal_bins: int,
) -> np.ndarray:
    features: list[np.ndarray] = []
    for bin_indices in _split_window_indices(window, temporal_bins):
        segment = np.asarray(data[:, :, bin_indices], dtype=np.float64)
        mean = segment.mean(axis=2)
        std = segment.std(axis=2, ddof=1 if segment.shape[2] > 1 else 0)
        minimum = segment.min(axis=2)
        maximum = segment.max(axis=2)
        slope = _linear_temporal_contrast(segment, _window_times_for_indices(times, bin_indices))
        features.extend([mean, std, minimum, maximum, slope])
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _features_for_window(
    data: np.ndarray,
    window: TimeWindow,
    *,
    times: np.ndarray | None = None,
    window_feature_mode: str = DEFAULT_WINDOW_FEATURE_MODE,
    temporal_bins: int = DEFAULT_TEMPORAL_BINS,
) -> np.ndarray:
    """Return a trial-by-feature matrix for one temporal decoding window.

    ``sensor_flat`` preserves the historical representation.  The compact modes
    summarize each channel inside a few temporal bins, reducing sample-level
    noise and peak memory while keeping the transform strictly source-only.
    """

    mode = normalize_window_feature_mode(window_feature_mode)
    if mode == "sensor_flat":
        return _sensor_flat_features(data, window)
    if mode == "bin_means":
        return _window_bin_mean_features(data, window, temporal_bins=temporal_bins)
    if mode == "mean_slope":
        return _window_mean_slope_features(data, window, times=times, temporal_bins=temporal_bins)
    if mode == "dct":
        return _window_dct_features(data, window, temporal_bins=temporal_bins)
    if mode == "stats":
        return _window_stat_features(data, window, times=times, temporal_bins=temporal_bins)
    if mode == "sensor_flat_plus_stats":
        stats = _window_stat_features(data, window, times=times, temporal_bins=temporal_bins)
        return np.concatenate([_sensor_flat_features(data, window), stats], axis=1).astype(np.float32, copy=False)
    raise AssertionError(f"Unhandled window feature mode: {mode}")


def make_source_pseudotrials(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    classes: np.ndarray,
    pseudotrials_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class-balanced source pseudo-trials for one subject.

    Each class contributes up to ``pseudotrials_per_class`` averaged rows.  When
    a class has fewer available rows than requested pseudo-trials, it contributes
    one row per available trial rather than oversampling.
    """

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels).ravel()
    if pseudotrials_per_class <= 0:
        return features, labels
    blocks: list[np.ndarray] = []
    block_labels: list[np.ndarray] = []
    for class_label in classes:
        indices = np.flatnonzero(labels == class_label)
        if indices.size == 0:
            continue
        shuffled = rng.permutation(indices)
        n_chunks = min(int(pseudotrials_per_class), int(indices.size))
        for chunk in np.array_split(shuffled, n_chunks):
            if chunk.size == 0:
                continue
            blocks.append(features[chunk].mean(axis=0, dtype=np.float64).astype(np.float32))
            block_labels.append(np.asarray([class_label], dtype=labels.dtype))
    if not blocks:
        raise ValueError("No pseudo-trials could be built from the source subject.")
    return np.vstack(blocks), np.concatenate(block_labels)


def _safe_pca_components(pca_components: int | float | str | None, *, feature_preprocessor: str, n_samples: int, n_features: int) -> int | float | str | None:
    normalized = normalize_feature_preprocessor(feature_preprocessor)
    if normalized == "none":
        return None
    if normalized == "anova_select":
        # This value is interpreted as a percentile by make_decoder(), not as a
        # PCA rank, so do not cap it by the number of pseudo-trials.
        return pca_components
    value = normalize_pca_components(pca_components)
    if isinstance(value, int):
        return max(1, min(value, n_samples - 1, n_features))
    return value


def _average_probabilities(probabilities: Sequence[np.ndarray], *, mode: str) -> np.ndarray:
    if not probabilities:
        raise ValueError("At least one probability matrix is required for ensembling.")
    stack = np.stack(probabilities, axis=0)
    if stack.ndim != 3:
        raise ValueError("Probability ensemble inputs must have shape (n_sources, n_samples, n_classes).")
    if not np.all(np.isfinite(stack)):
        raise ValueError("Probability ensemble inputs must be finite.")
    if np.any(stack < 0.0):
        raise ValueError("Probability ensemble inputs must be non-negative.")
    if np.any(stack > 1.0):
        raise ValueError("Probability ensemble inputs must not exceed 1.0.")
    row_sums = stack.sum(axis=2)
    bad_rows = np.flatnonzero(np.abs(row_sums.ravel() - 1.0) > PROBABILITY_SUM_TOLERANCE)
    if len(bad_rows):
        examples = [float(row_sums.ravel()[index]) for index in bad_rows[:5]]
        raise ValueError(
            "Probability ensemble rows must sum to 1.0 within tolerance "
            f"{PROBABILITY_SUM_TOLERANCE:g}; example row sums: {examples}"
        )
    mode = mode.lower().replace("-", "_")
    if mode in {"mean", "arithmetic"}:
        averaged = np.mean(stack, axis=0)
    elif mode in {"log", "log_mean", "geometric"}:
        log_average = np.mean(np.log(np.clip(stack, EPSILON, 1.0)), axis=0)
        shifted = log_average - np.max(log_average, axis=1, keepdims=True)
        averaged = np.exp(np.clip(shifted, -50.0, 50.0))
    else:
        raise ValueError("ensemble_mode must be 'mean' or 'log'.")
    row_sums = averaged.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Ensembled probabilities must have positive row sums.")
    return averaged / row_sums


def _fit_predict_window(
    *,
    source_subjects: Sequence[CachedSubject],
    heldout_subject: CachedSubject,
    window: TimeWindow,
    times: np.ndarray,
    encoder: LabelEncoder,
    classes: np.ndarray,
    decoders: Sequence[str],
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: int | float | str | None,
    window_feature_mode: str,
    temporal_bins: int,
    max_iter: int,
    pseudotrials_per_class: int,
    pseudotrial_seed: int,
    ensemble_mode: str,
) -> tuple[np.ndarray, int]:
    train_features: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    for source_index, subject in enumerate(source_subjects):
        source_data = _subject_data(subject)
        source_labels = encoder.transform(_subject_labels(subject))
        source_features = _features_for_window(
            source_data,
            window,
            times=times,
            window_feature_mode=window_feature_mode,
            temporal_bins=temporal_bins,
        )
        rng = np.random.default_rng(int(pseudotrial_seed) + 1009 * source_index + 9173 * int(round(window[2] * 1000.0)))
        subject_features, subject_labels = make_source_pseudotrials(
            source_features,
            source_labels,
            classes=classes,
            pseudotrials_per_class=pseudotrials_per_class,
            rng=rng,
        )
        train_features.append(subject_features)
        train_labels.append(subject_labels)

    x_train = np.vstack(train_features)
    y_train = np.concatenate(train_labels)
    heldout_features = _features_for_window(
        _subject_data(heldout_subject),
        window,
        times=times,
        window_feature_mode=window_feature_mode,
        temporal_bins=temporal_bins,
    )
    effective_pca = _safe_pca_components(
        pca_components,
        feature_preprocessor=feature_preprocessor,
        n_samples=x_train.shape[0],
        n_features=x_train.shape[1],
    )

    probability_matrices: list[np.ndarray] = []
    for decoder_name in decoders:
        model = make_decoder(
            normalize_decoder_name(decoder_name),
            max_iter=max_iter,
            emission_mode=emission_mode,
            feature_preprocessor=feature_preprocessor,
            pca_components=effective_pca,
        )
        model.fit(x_train, y_train)
        probabilities = _align_probability_columns(
            predict_emission_probabilities(model, heldout_features, emission_mode=emission_mode),
            model=model,
            classes=classes,
        )
        probability_matrices.append(probabilities)
    return _average_probabilities(probability_matrices, mode=ensemble_mode), int(y_train.shape[0])


def _metric_row(
    *,
    analysis: str,
    heldout_subject: CachedSubject,
    source_subjects: Sequence[CachedSubject],
    probabilities: np.ndarray,
    y_true: np.ndarray,
    classes: np.ndarray,
    class_names: np.ndarray,
    window: TimeWindow | None,
    selected_windows: Sequence[TimeWindow],
    times: np.ndarray,
    n_train: int,
    decoders: Sequence[str],
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: int | float | str | None,
    window_feature_mode: str,
    temporal_bins: int,
    normalization: str,
    baseline_window: tuple[float, float],
    pseudotrials_per_class: int,
    ensemble_mode: str,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    if window is None:
        starts = [candidate[0] for candidate in selected_windows]
        stops = [candidate[1] for candidate in selected_windows]
        centers = [candidate[2] for candidate in selected_windows]
        time = float(np.mean(centers))
        window_start = float(times[min(starts)])
        window_stop = float(times[max(stops) - 1])
        n_windows = len(selected_windows)
    else:
        start, stop, center = window
        time = center
        window_start = float(times[start])
        window_stop = float(times[stop - 1])
        n_windows = 1
    return {
        "analysis": analysis,
        "heldout_subject": heldout_subject.participant,
        "source_subjects": "|".join(subject.participant for subject in source_subjects),
        "n_source_subjects": len(source_subjects),
        "decoder": ",".join(decoders),
        "emission_mode": emission_mode,
        "feature_preprocessor": feature_preprocessor,
        "pca_components": "" if pca_components is None else pca_components,
        "window_feature_mode": window_feature_mode,
        "temporal_bins": temporal_bins,
        "normalization": normalization,
        "baseline_window_start": baseline_window[0],
        "baseline_window_stop": baseline_window[1],
        "pseudotrials_per_class": pseudotrials_per_class,
        "ensemble_mode": ensemble_mode,
        "time": time,
        "window_start": window_start,
        "window_stop": window_stop,
        "n_ensemble_windows": n_windows,
        "accuracy": accuracy_score(y_true, predictions),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "top2_accuracy": _top_k_accuracy(probabilities, y_true, k=2),
        "top3_accuracy": _top_k_accuracy(probabilities, y_true, k=3),
        "log_loss": log_loss(y_true, probabilities, labels=classes),
        "brier": brier_score_multiclass(probabilities, y_true),
        "ece": expected_calibration_error(probabilities, y_true),
        "n_train": n_train,
        "n_test": int(y_true.shape[0]),
        "n_classes": int(classes.shape[0]),
        "class_names": "|".join(map(str, class_names)),
    }


def _write_results_incremental(out_path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    results = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    return results


def _load_existing_rows(out_path: Path, *, resume: bool) -> list[dict[str, Any]]:
    if not resume or not out_path.exists():
        return []
    existing = pd.read_csv(out_path)
    return existing.to_dict(orient="records")


def _completed_ensemble_subjects(rows: Sequence[dict[str, Any]]) -> set[str]:
    completed: set[str] = set()
    for row in rows:
        if row.get("analysis") == "temporal_ensemble":
            completed.add(str(row.get("heldout_subject")))
    return completed


def _drop_incomplete_resume_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only fully completed held-out subjects when resuming.

    If a previous process was killed after writing per-window rows but before the
    temporal ensemble row, recomputing that held-out subject should replace the
    partial rows rather than append duplicates.
    """

    completed = _completed_ensemble_subjects(rows)
    return [row for row in rows if str(row.get("heldout_subject")) in completed]


def _resume_rows_for_window_feature_config(
    rows: Sequence[dict[str, Any]],
    *,
    window_feature_mode: str,
    temporal_bins: int,
) -> list[dict[str, Any]]:
    """Keep resume rows that match the current window-feature extraction mode."""

    kept: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("window_feature_mode", "")) != window_feature_mode:
            continue
        try:
            row_bins = int(row.get("temporal_bins"))
        except (TypeError, ValueError):
            continue
        if row_bins == int(temporal_bins):
            kept.append(row)
    return kept


def _select_centered_windows(windows: Sequence[TimeWindow], decode_window: tuple[float, float] | None) -> list[TimeWindow]:
    if decode_window is None:
        selected = list(windows)
    else:
        start, stop = decode_window
        selected = [window for window in windows if start <= window[2] <= stop]
    if not selected:
        centers = [window[2] for window in windows]
        raise ValueError(
            f"No time-window centers fall inside decode_window={decode_window}; "
            f"available centers span [{min(centers):.6f}, {max(centers):.6f}]."
        )
    return selected


def run_bushmeg_loso_decode(
    *,
    data_dir: Path,
    out_path: Path,
    participants: str | Iterable[str | int] | None = None,
    file_template: str = DEFAULT_MAIN_TEMPLATE,
    cache_dir: Path | None = None,
    fieldtrip_root_path: str | Sequence[str | int] | None = None,
    label_column: str = "condition",
    label_base: float | None = 1.0,
    trialinfo_column: int = 0,
    trim_overlong_labels: bool = True,
    tmin: float | None = -0.35,
    tmax: float | None = 0.25,
    window_ms: float = 100.0,
    step_ms: float = 25.0,
    decode_window: Sequence[float] | None = DEFAULT_DECODE_WINDOW,
    decoders: str | Sequence[str] = DEFAULT_DECODERS,
    emission_mode: str = "uncalibrated",
    feature_preprocessor: str = "pca_whiten",
    pca_components: int | float | str | None = 128,
    window_feature_mode: str = DEFAULT_WINDOW_FEATURE_MODE,
    temporal_bins: int = DEFAULT_TEMPORAL_BINS,
    normalization: str = "subject_baseline_whiten",
    baseline_window: Sequence[float] | None = DEFAULT_BASELINE_WINDOW,
    pseudotrials_per_class: int = 4,
    pseudotrial_seed: int = 13,
    ensemble_mode: str = "log",
    max_iter: int = 2000,
    max_folds: int | None = None,
    resume: bool = True,
) -> pd.DataFrame:
    """Run strict source-only BUSH-MEG LOSO decoding and write a CSV table."""

    participant_ids = _participant_tokens(participants)
    if len(participant_ids) < 2:
        raise ValueError("LOSO decoding requires at least two participants.")
    normalized_decoders = tuple(normalize_decoder_name(decoder) for decoder in _split_csv_tokens(decoders))
    emission_mode = normalize_emission_mode(emission_mode)
    feature_preprocessor = normalize_feature_preprocessor(feature_preprocessor)
    window_feature_mode = normalize_window_feature_mode(window_feature_mode)
    temporal_bins_value = _normalize_temporal_bins(temporal_bins)
    normalization = normalize_epoch_normalization(normalization)
    baseline_window_value = _normalize_baseline_window(None if baseline_window is None else tuple(baseline_window))
    decode_window_value = _decode_window_tuple(None if decode_window is None else tuple(decode_window))
    fieldtrip_root_tokens = parse_path_tokens(fieldtrip_root_path, DEFAULT_ROOT_PATH)
    cache_root = cache_dir or out_path.with_suffix(".subject-cache")

    cached_subjects: list[CachedSubject] = []
    for participant in participant_ids:
        cached_subjects.append(
            _prepare_subject_cache(
                participant=participant,
                source_path=data_dir / file_template.format(participant=participant, subject=participant),
                cache_dir=cache_root,
                fieldtrip_root_path=fieldtrip_root_tokens,
                label_column=label_column,
                label_base=label_base,
                trialinfo_column=trialinfo_column,
                tmin=tmin,
                tmax=tmax,
                normalization=normalization,
                baseline_window=baseline_window_value,
                trim_overlong_labels=trim_overlong_labels,
            )
        )

    times = _subject_times(cached_subjects[0])
    for subject in cached_subjects[1:]:
        subject_times = _subject_times(subject)
        if len(subject_times) != len(times) or not np.allclose(subject_times, times, rtol=1e-7, atol=1e-12):
            raise ValueError(f"Participant {subject.participant} has a non-matching time axis after cropping/cache preparation.")

    raw_label_blocks = [_subject_labels(subject) for subject in cached_subjects]
    encoder = LabelEncoder().fit(np.concatenate(raw_label_blocks))
    class_names = encoder.classes_
    classes = np.arange(len(class_names), dtype=int)
    windows = time_windows(times, window_ms=window_ms, step_ms=step_ms)
    selected_windows = _select_centered_windows(windows, decode_window_value)

    rows = _load_existing_rows(out_path, resume=resume)
    if resume:
        rows = _resume_rows_for_window_feature_config(rows, window_feature_mode=window_feature_mode, temporal_bins=temporal_bins_value)
        rows = _drop_incomplete_resume_rows(rows)
    completed = _completed_ensemble_subjects(rows)
    fold_subjects = [subject for subject in cached_subjects if subject.participant not in completed]
    if max_folds is not None:
        fold_subjects = fold_subjects[: int(max_folds)]

    for heldout_subject in fold_subjects:
        source_subjects = [subject for subject in cached_subjects if subject.participant != heldout_subject.participant]
        y_true = encoder.transform(_subject_labels(heldout_subject))
        per_window_probabilities: list[np.ndarray] = []
        last_n_train = 0
        for window in selected_windows:
            probabilities, n_train = _fit_predict_window(
                source_subjects=source_subjects,
                heldout_subject=heldout_subject,
                window=window,
                times=times,
                encoder=encoder,
                classes=classes,
                decoders=normalized_decoders,
                emission_mode=emission_mode,
                feature_preprocessor=feature_preprocessor,
                pca_components=pca_components,
                window_feature_mode=window_feature_mode,
                temporal_bins=temporal_bins_value,
                max_iter=max_iter,
                pseudotrials_per_class=int(pseudotrials_per_class),
                pseudotrial_seed=int(pseudotrial_seed),
                ensemble_mode=ensemble_mode,
            )
            last_n_train = n_train
            per_window_probabilities.append(probabilities)
            rows.append(
                _metric_row(
                    analysis="window",
                    heldout_subject=heldout_subject,
                    source_subjects=source_subjects,
                    probabilities=probabilities,
                    y_true=y_true,
                    classes=classes,
                    class_names=class_names,
                    window=window,
                    selected_windows=selected_windows,
                    times=times,
                    n_train=n_train,
                    decoders=normalized_decoders,
                    emission_mode=emission_mode,
                    feature_preprocessor=feature_preprocessor,
                    pca_components=pca_components,
                    window_feature_mode=window_feature_mode,
                    temporal_bins=temporal_bins_value,
                    normalization=normalization,
                    baseline_window=baseline_window_value,
                    pseudotrials_per_class=int(pseudotrials_per_class),
                    ensemble_mode=ensemble_mode,
                )
            )
        temporal_probabilities = _average_probabilities(per_window_probabilities, mode=ensemble_mode)
        rows.append(
            _metric_row(
                analysis="temporal_ensemble",
                heldout_subject=heldout_subject,
                source_subjects=source_subjects,
                probabilities=temporal_probabilities,
                y_true=y_true,
                classes=classes,
                class_names=class_names,
                window=None,
                selected_windows=selected_windows,
                times=times,
                n_train=last_n_train,
                decoders=normalized_decoders,
                emission_mode=emission_mode,
                feature_preprocessor=feature_preprocessor,
                pca_components=pca_components,
                window_feature_mode=window_feature_mode,
                temporal_bins=temporal_bins_value,
                normalization=normalization,
                baseline_window=baseline_window_value,
                pseudotrials_per_class=int(pseudotrials_per_class),
                ensemble_mode=ensemble_mode,
            )
        )
        _write_results_incremental(out_path, rows)
        print(f"Finished held-out participant {heldout_subject.participant}; wrote {out_path}")

    return _write_results_incremental(out_path, rows)


def _default_data_dir() -> Path:
    configured = os.environ.get("BUSHMEG_DATA_DIR") or os.environ.get("PYMEGDEC_DATA_DIR")
    return Path(configured) if configured else Path(".cache") / "bushmeg"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run memory-bounded source-only LOSO decoding on BUSH-MEG Part*Data.mat files.")
    parser.add_argument("--data-dir", type=Path, default=_default_data_dir())
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--participants", default=DEFAULT_BUSHMEG_PARTICIPANTS)
    parser.add_argument("--file-template", default=DEFAULT_MAIN_TEMPLATE)
    parser.add_argument("--cache-dir", type=Path, help="Subject cache directory. Default: <out>.subject-cache")
    parser.add_argument("--fieldtrip-root-path", default="data,0")
    parser.add_argument("--label-column", default="condition")
    parser.add_argument("--fieldtrip-label-base", type=float, default=1.0)
    parser.add_argument("--trialinfo-column", type=int, default=0)
    parser.add_argument("--fieldtrip-no-trim-overlong-labels", action="store_true")
    parser.add_argument("--tmin", type=float, default=-0.35)
    parser.add_argument("--tmax", type=float, default=0.25)
    parser.add_argument("--window-ms", type=float, default=100.0)
    parser.add_argument("--step-ms", type=float, default=25.0)
    parser.add_argument(
        "--decode-window",
        nargs=2,
        type=float,
        default=DEFAULT_DECODE_WINDOW,
        metavar=("START", "STOP"),
        help="Only fit/evaluate windows whose centers fall in START..STOP seconds. Use a targeted prior window to keep LOSO cheap.",
    )
    parser.add_argument("--decoders", default=",".join(DEFAULT_DECODERS), help="Comma-separated decoder ensemble, e.g. logistic,linear_svm,correlation-prototype.")
    parser.add_argument("--emission-mode", choices=EMISSION_MODE_CHOICES, default="uncalibrated")
    parser.add_argument("--feature-preprocessor", choices=FEATURE_PREPROCESSOR_CHOICES, default="pca_whiten")
    parser.add_argument("--pca-components", default="128")
    parser.add_argument(
        "--window-feature-mode",
        "--feature-mode",
        dest="window_feature_mode",
        default=DEFAULT_WINDOW_FEATURE_MODE,
        help=(
            "Feature representation inside each time window. "
            f"Available: {', '.join(WINDOW_FEATURE_MODE_CHOICES)}. "
            "Use compact modes such as mean_slope, dct, or stats to reduce sample-level noise and memory."
        ),
    )
    parser.add_argument("--temporal-bins", type=int, default=DEFAULT_TEMPORAL_BINS, help="Number of bins/DCT coefficients used by compact window feature modes.")
    parser.add_argument("--normalization", choices=EPOCH_NORMALIZATION_RUN_CHOICES, default="subject_baseline_whiten")
    parser.add_argument("--baseline-window", nargs=2, type=float, default=DEFAULT_BASELINE_WINDOW, metavar=("START", "STOP"))
    parser.add_argument("--pseudotrials-per-class", type=int, default=4)
    parser.add_argument("--pseudotrial-seed", type=int, default=13)
    parser.add_argument("--ensemble-mode", choices=("mean", "log"), default="log")
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--max-folds", type=int, help="Optional smoke-test cap on held-out subjects.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore any existing --out CSV and recompute all folds.")
    parser.add_argument("--selection-metric", choices=RESULT_SELECTION_METRIC_CHOICES, default="balanced_accuracy")
    args = parser.parse_args(argv)

    results = run_bushmeg_loso_decode(
        data_dir=args.data_dir,
        out_path=args.out,
        participants=args.participants,
        file_template=args.file_template,
        cache_dir=args.cache_dir,
        fieldtrip_root_path=args.fieldtrip_root_path,
        label_column=args.label_column,
        label_base=args.fieldtrip_label_base,
        trialinfo_column=args.trialinfo_column,
        trim_overlong_labels=not args.fieldtrip_no_trim_overlong_labels,
        tmin=args.tmin,
        tmax=args.tmax,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        decode_window=tuple(args.decode_window),
        decoders=args.decoders,
        emission_mode=args.emission_mode,
        feature_preprocessor=args.feature_preprocessor,
        pca_components=args.pca_components,
        window_feature_mode=args.window_feature_mode,
        temporal_bins=args.temporal_bins,
        normalization=args.normalization,
        baseline_window=tuple(args.baseline_window),
        pseudotrials_per_class=args.pseudotrials_per_class,
        pseudotrial_seed=args.pseudotrial_seed,
        ensemble_mode=args.ensemble_mode,
        max_iter=args.max_iter,
        max_folds=args.max_folds,
        resume=not args.no_resume,
    )
    if results.empty:
        print(f"No rows written to {args.out}")
        return 0
    for analysis, summary in results.groupby("analysis", sort=True):
        metric = summary[args.selection_metric]
        print(f"{analysis}: mean {args.selection_metric}={metric.mean():.4f} over {len(metric)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

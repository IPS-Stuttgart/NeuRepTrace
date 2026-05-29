from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.decoding import SlidingEstimator
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from neureptrace.decoding import (
    DECODER_CLI_CHOICES,
    EMISSION_MODE_CHOICES,
    FEATURE_PREPROCESSOR_CHOICES,
    TUNING_SCORING_CHOICES,
    make_cross_validator,
    make_decoder,
    make_tuning_cross_validator,
    normalize_anova_select_percentile,
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    normalize_pca_components,
    normalize_pls_components,
    normalize_tuning_scoring,
    parse_c_grid,
    predict_emission_probabilities,
    time_windows,
)
from neureptrace.fieldtrip_mat import INPUT_FORMAT_CHOICES, load_fieldtrip_raw_mat_epochs, parse_path_tokens
from neureptrace.metrics import brier_score_multiclass, expected_calibration_error, reliability_bins
from neureptrace.observations import ProbabilityObservationTable, stable_hash

FIELDTRIP_DEFAULT_ROOT_PATH = ("data", 0)
EMISSION_RUN_CHOICES = (*EMISSION_MODE_CHOICES, "both")
FEATURE_PREPROCESSOR_RUN_CHOICES = (*FEATURE_PREPROCESSOR_CHOICES, "pca-whiten", "anova-select", "select-percentile", "pls-da", "pls")
EPOCH_NORMALIZATION_CHOICES = (
    "none",
    "subject_z",
    "subject_trial_z",
    "subject_baseline_z",
    "subject_baseline_whiten",
)
EPOCH_NORMALIZATION_RUN_CHOICES = (
    *EPOCH_NORMALIZATION_CHOICES,
    "subject-z",
    "subject-trial-z",
    "subject-baseline-z",
    "subject-baseline-whiten",
)
RESULT_SELECTION_METRIC_CHOICES = (
    "accuracy",
    "balanced_accuracy",
    "top2_accuracy",
    "top3_accuracy",
    "log_loss",
    "brier",
    "ece",
)
RESULT_SUMMARY_METRIC_COLUMNS = RESULT_SELECTION_METRIC_CHOICES
RESULT_SELECTION_MINIMIZE_METRICS = {"log_loss", "brier", "ece"}
TIME_DECODE_BACKEND_CHOICES = ("auto", "sklearn", "mne")
CLASS_PRIOR_CORRECTION_CHOICES = ("none", "train_uniform")
CLASS_PRIOR_CORRECTION_RUN_CHOICES = (*CLASS_PRIOR_CORRECTION_CHOICES, "train-uniform")
DEFAULT_BASELINE_WINDOW = (-0.35, -0.05)
BASELINE_WHITENING_SHRINKAGE = 0.1
BASELINE_WHITENING_EIGENVALUE_FLOOR = 1e-6
MNE_SLIDING_MAX_FEATURE_BYTES = 512 * 1024 * 1024
TimeWindow = tuple[int, int, float]
TemporalTrainWindow = tuple[float, float]
DecodeWindow = tuple[float, float]
TEMPORAL_TRAIN_MODE_CHOICES = ("window_ensemble", "pooled")
TEMPORAL_TRAIN_MODE_RUN_CHOICES = (*TEMPORAL_TRAIN_MODE_CHOICES, "window-ensemble")


def _add_subject(row: dict, subject: str | None) -> dict:
    if subject is not None:
        row = {"subject": subject, **row}
    return row


def _group_aliases(value: object) -> set[str]:
    text = str(value).strip()
    if not text:
        return set()

    normalized = text.lower()
    aliases = {text, normalized}
    if normalized.startswith("sub-"):
        suffix = normalized.removeprefix("sub-")
        aliases.add(suffix)
        if suffix.isdigit():
            aliases.add(str(int(suffix)))
    elif normalized.isdigit():
        aliases.add(f"sub-{int(normalized):02d}")
    return {alias for alias in aliases if alias}


def _normalize_outer_test_groups(value: object | Sequence[object] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        values: list[object] = [
            item.strip().strip("\"'")
            for comma_part in text.split(",")
            for item in comma_part.split()
            if item.strip().strip("\"'")
        ]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        values = [value]

    return tuple(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


def _filter_splits_for_outer_test_groups(
    splits: Sequence[tuple[int, tuple[np.ndarray, np.ndarray]]],
    groups: np.ndarray | None,
    outer_test_groups: object | Sequence[object] | None,
) -> list[tuple[int, tuple[np.ndarray, np.ndarray]]]:
    requested_groups = _normalize_outer_test_groups(outer_test_groups)
    if not requested_groups:
        return list(splits)
    if groups is None:
        raise ValueError("outer_test_groups requires group_column so held-out groups can be identified.")

    requested_aliases = set().union(*(_group_aliases(group) for group in requested_groups))
    group_values = np.asarray(groups)
    selected: list[tuple[int, tuple[np.ndarray, np.ndarray]]] = []
    available_aliases: set[str] = set()
    for fold, (train_idx, test_idx) in splits:
        test_group_values = np.unique(group_values[test_idx])
        split_aliases = set().union(*(_group_aliases(group) for group in test_group_values))
        available_aliases.update(split_aliases)
        if requested_aliases & split_aliases:
            selected.append((fold, (train_idx, test_idx)))

    if not selected:
        requested = ", ".join(sorted(requested_aliases))
        available = ", ".join(sorted(available_aliases))
        raise ValueError(f"No outer CV split matched outer_test_groups={requested}. Available groups: {available}.")
    return selected


def normalize_input_format(input_format: str | None) -> str:
    """Normalize supported epoch input formats for the direct decoder."""

    normalized = "mne-epochs" if input_format is None else str(input_format).strip().lower().replace("_", "-")
    aliases = {
        "mne": "mne-epochs",
        "mne-epoch": "mne-epochs",
        "mne-epochs-fif": "mne-epochs",
        "fif": "mne-epochs",
        "epochs": "mne-epochs",
        "fieldtrip": "fieldtrip-mat",
        "fieldtrip-raw": "fieldtrip-mat",
        "fieldtrip-raw-mat": "fieldtrip-mat",
        "mat": "fieldtrip-mat",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in INPUT_FORMAT_CHOICES:
        raise ValueError(
            f"Unknown input format '{input_format}'. Available formats: {', '.join(INPUT_FORMAT_CHOICES)}."
        )
    return normalized


def _parse_path_tokens(raw_path: str | Sequence[str | int] | None) -> tuple[str | int, ...]:
    """Parse a YAML/CLI-style FieldTrip root path such as ``data,0``."""

    return parse_path_tokens(raw_path, FIELDTRIP_DEFAULT_ROOT_PATH)


def _load_epochs_and_metadata(
    epochs_path: Path,
    metadata_csv: Path | None,
    *,
    input_format: str = "mne-epochs",
    fieldtrip_root_path: str | None = None,
    fieldtrip_label_base: float | None = 1.0,
    fieldtrip_ch_type: str = "grad",
    fieldtrip_trim_overlong_labels: bool = True,
    label_column: str = "condition",
) -> tuple[mne.Epochs, pd.DataFrame]:
    input_format = normalize_input_format(input_format)
    if input_format == "mne-epochs":
        epochs = mne.read_epochs(epochs_path, preload=True, verbose="error")
        metadata = epochs.metadata.copy() if epochs.metadata is not None else None
    elif input_format == "fieldtrip-mat":
        epochs, metadata = load_fieldtrip_raw_mat_epochs(
            epochs_path,
            root_path=_parse_path_tokens(fieldtrip_root_path),
            label_column=label_column,
            label_base=fieldtrip_label_base,
            ch_type=fieldtrip_ch_type,
            trim_overlong_labels=fieldtrip_trim_overlong_labels,
        )
    else:
        raise ValueError(f"Unknown input_format '{input_format}'. Available formats: {', '.join(INPUT_FORMAT_CHOICES)}.")
    if metadata_csv is not None:
        metadata = pd.read_csv(metadata_csv)
    if metadata is None:
        raise ValueError("No metadata found. Provide --metadata-csv or use epochs with metadata.")
    if len(metadata) != len(epochs):
        raise ValueError(
            f"Metadata row count ({len(metadata)}) does not match epochs ({len(epochs)})."
        )
    return epochs, metadata.reset_index(drop=True)


def _best_params_json(models) -> str:
    if isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
        best_params = [getattr(model, "best_params_", None) for model in models]
        best_params = [params for params in best_params if params is not None]
    else:
        best_params = getattr(models, "best_params_", None)
    return "" if not best_params else json.dumps(best_params, sort_keys=True, default=str, separators=(",", ":"))


def _best_scores(models) -> list[float]:
    if isinstance(models, Sequence) and not isinstance(models, (str, bytes)):
        return [float(model.best_score_) for model in models if hasattr(model, "best_score_")]
    if hasattr(models, "best_score_"):
        return [float(models.best_score_)]
    return []


def _stable_shuffle_seed(seed: int, context: Sequence[object]) -> int:
    payload = {"seed": int(seed), "context": [str(item) for item in context]}
    return int(stable_hash(payload, length=16), 16)


def _shuffle_training_labels(labels: np.ndarray, *, seed: int, context: Sequence[object]) -> np.ndarray:
    """Return a deterministic count-preserving permutation for train-only null controls."""

    labels = np.asarray(labels, dtype=int).reshape(-1)
    rng = np.random.default_rng(_stable_shuffle_seed(seed, context))
    return rng.permutation(labels)


def _fold_training_labels(
    labels: np.ndarray,
    train_idx: Sequence[int] | np.ndarray,
    *,
    label_shuffle_control: bool,
    label_shuffle_seed: int,
    context: Sequence[object],
) -> np.ndarray:
    train_labels = np.asarray(labels, dtype=int)[np.asarray(train_idx, dtype=int)]
    if not label_shuffle_control:
        return train_labels
    return _shuffle_training_labels(train_labels, seed=int(label_shuffle_seed), context=context)


def _tuning_metadata(
    models,
    *,
    tune_hyperparameters: bool,
    tuning_cv_splits: int,
    tuning_scoring: str,
    tuning_c_grid: Sequence[float],
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "tuned_hyperparameters": bool(tune_hyperparameters),
        "tuning_cv_splits": int(tuning_cv_splits) if tune_hyperparameters else "",
        "tuning_scoring": tuning_scoring if tune_hyperparameters else "",
        "tuning_c_grid": "|".join(str(value) for value in tuning_c_grid) if tune_hyperparameters else "",
        "best_params": "",
    }
    if not tune_hyperparameters:
        return metadata
    metadata["best_params"] = _best_params_json(models)
    scores = _best_scores(models)
    if len(scores) == 1:
        metadata["best_score"] = scores[0]
    elif scores:
        metadata["best_score"] = float(np.mean(scores))
        metadata["best_scores"] = json.dumps(scores, separators=(",", ":"))
    return metadata


def normalize_epoch_normalization(name: str | None) -> str:
    """Normalize subject-level epoch normalization names for the MNE decoder."""

    normalized = "none" if name is None else str(name).strip().lower().replace("-", "_")
    if normalized in {"identity", "raw", "no", "false"}:
        return "none"
    if normalized not in EPOCH_NORMALIZATION_CHOICES:
        raise ValueError(
            f"Unknown normalization '{name}'. Available normalizations: {', '.join(EPOCH_NORMALIZATION_CHOICES)}."
        )
    return normalized


def normalize_time_decode_backend(name: str | None) -> str:
    """Normalize the implementation backend for same-time decoding."""

    normalized = "auto" if name is None else str(name).strip().lower().replace("-", "_")
    if normalized == "mne_decoding":
        return "mne"
    if normalized not in TIME_DECODE_BACKEND_CHOICES:
        raise ValueError(f"Unknown time-decode backend '{name}'. Available backends: {', '.join(TIME_DECODE_BACKEND_CHOICES)}.")
    return normalized


def _normalize_baseline_window(baseline_window: tuple[float, float] | list[float] | None) -> tuple[float, float]:
    if baseline_window is None:
        return DEFAULT_BASELINE_WINDOW
    if len(baseline_window) != 2:
        raise ValueError("baseline_window must contain exactly two times: start and stop.")
    start, stop = map(float, baseline_window)
    if stop < start:
        raise ValueError("baseline_window stop must be greater than or equal to start.")
    return start, stop


def _baseline_time_mask(times: np.ndarray, baseline_window: tuple[float, float]) -> np.ndarray:
    start, stop = baseline_window
    tolerance = 1e-12
    mask = (times >= start - tolerance) & (times <= stop + tolerance)
    if not np.any(mask):
        raise ValueError(f"baseline_window [{start}, {stop}] does not overlap the epochs time axis.")
    return mask


def _nonzero_std(std: np.ndarray) -> np.ndarray:
    return np.where(std < 1e-12, 1.0, std)


def _channel_mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=(0, 2), keepdims=True)
    std = _nonzero_std(values.std(axis=(0, 2), keepdims=True))
    return mean, std


def _covariance_matrix(features: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=float)
    n_features = int(features.shape[1])
    if features.shape[0] < 2:
        return np.eye(n_features, dtype=float)
    covariance = np.cov(features, rowvar=False)
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    return 0.5 * (covariance + covariance.T)


def _shrink_covariance(covariance: np.ndarray, *, shrinkage: float) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    diagonal = np.diag(np.diag(covariance))
    return (1.0 - float(shrinkage)) * covariance + float(shrinkage) * diagonal


def _whitening_matrix(covariance: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigen_floor = max(float(np.max(eigenvalues)) * BASELINE_WHITENING_EIGENVALUE_FLOOR, 1e-12)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, eigen_floor))
    whitening = (eigenvectors * inverse_sqrt) @ eigenvectors.T
    return 0.5 * (whitening + whitening.T)


def _baseline_channel_whitening_matrix(data: np.ndarray, times: np.ndarray, baseline_window: tuple[float, float]) -> np.ndarray:
    mask = _baseline_time_mask(times, baseline_window)
    baseline_trial_means = np.mean(data[:, :, mask], axis=2)
    covariance = _covariance_matrix(baseline_trial_means)
    covariance = _shrink_covariance(covariance, shrinkage=BASELINE_WHITENING_SHRINKAGE)
    return _whitening_matrix(covariance)


def _apply_epoch_normalization(
    data: np.ndarray,
    times: np.ndarray,
    normalization: str,
    *,
    baseline_window: tuple[float, float],
) -> np.ndarray:
    """Apply subject-level normalization before extracting time-window features.

    ``subject_baseline_whiten`` mirrors PyMEGDec's channel-wise baseline
    whitening: subtract the baseline channel mean and apply a shrinkage
    covariance whitening matrix fitted from per-trial baseline channel means.
    """

    data = np.asarray(data, dtype=float)
    normalization = normalize_epoch_normalization(normalization)
    if normalization == "none":
        return data

    normalized = data.copy()
    if normalization == "subject_z":
        mean, std = _channel_mean_std(normalized)
        return (normalized - mean) / std

    if normalization == "subject_trial_z":
        mean = normalized.mean(axis=(1, 2), keepdims=True)
        std = _nonzero_std(normalized.std(axis=(1, 2), keepdims=True))
        return (normalized - mean) / std

    mask = _baseline_time_mask(times, baseline_window)
    baseline = normalized[:, :, mask]
    baseline_mean, baseline_std = _channel_mean_std(baseline)
    if normalization == "subject_baseline_z":
        return (normalized - baseline_mean) / baseline_std

    if normalization == "subject_baseline_whiten":
        whitening = _baseline_channel_whitening_matrix(normalized, times, baseline_window)
        centered = normalized - baseline_mean
        whitened = np.einsum("ntc,dc->ntd", np.transpose(centered, (0, 2, 1)), whitening)
        return np.transpose(whitened, (0, 2, 1))

    raise ValueError(f"Unsupported normalization: {normalization}")


def _normalize_temporal_train_window(
    temporal_train_window: tuple[float, float] | list[float] | None,
) -> TemporalTrainWindow | None:
    if temporal_train_window is None:
        return None
    if len(temporal_train_window) != 2:
        raise ValueError("temporal_train_window must contain exactly two times: start and stop.")
    start, stop = map(float, temporal_train_window)
    if stop < start:
        raise ValueError("temporal_train_window stop must be greater than or equal to start.")
    return start, stop


def _normalize_temporal_train_mode(mode: str | None) -> str:
    """Normalize how a non-empty temporal train window is used.

    ``window_ensemble`` keeps the historical behavior: fit one model per
    selected train-time window and average probabilities. ``pooled`` treats all
    selected train-time windows from the source subjects as fold-local temporal
    augmentation and fits one classifier on the pooled rows. The pooled mode is
    considerably cheaper and can improve small cross-subject M/EEG datasets by
    increasing the effective number of source examples without using held-out
    subject trials.
    """

    normalized = "window_ensemble" if mode is None else str(mode).strip().lower().replace("-", "_")
    aliases = {
        "ensemble": "window_ensemble",
        "train_window_ensemble": "window_ensemble",
        "temporal_ensemble": "window_ensemble",
        "pool": "pooled",
        "pooled_windows": "pooled",
        "temporal_pool": "pooled",
        "train_window_pooled": "pooled",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TEMPORAL_TRAIN_MODE_CHOICES:
        raise ValueError(
            f"Unknown temporal_train_mode '{mode}'. Available modes: "
            f"{', '.join(TEMPORAL_TRAIN_MODE_CHOICES)}."
        )
    return normalized


def _select_temporal_train_windows(
    windows: list[TimeWindow],
    temporal_train_window: tuple[float, float] | list[float] | None,
) -> list[TimeWindow] | None:
    normalized = _normalize_temporal_train_window(temporal_train_window)
    if normalized is None:
        return None
    train_start, train_stop = normalized
    selected = [window for window in windows if train_start <= window[2] <= train_stop]
    if selected:
        return selected

    available_centers = [window[2] for window in windows]
    if not available_centers:
        raise ValueError("No time windows are available for temporal train-window selection.")
    raise ValueError(
        "No time-window centers fall inside temporal_train_window "
        f"[{train_start}, {train_stop}]. Available centers span "
        f"[{min(available_centers)}, {max(available_centers)}]."
    )


def _normalize_decode_window(decode_window: tuple[float, float] | list[float] | None) -> DecodeWindow | None:
    if decode_window is None:
        return None
    if len(decode_window) != 2:
        raise ValueError("decode_window must contain exactly two times: start and stop.")
    start, stop = map(float, decode_window)
    if stop < start:
        raise ValueError("decode_window stop must be greater than or equal to start.")
    return start, stop


def _select_decode_windows(windows: list[TimeWindow], decode_window: DecodeWindow | None) -> list[TimeWindow]:
    if decode_window is None:
        return list(windows)
    decode_start, decode_stop = decode_window
    selected = [window for window in windows if decode_start <= window[2] <= decode_stop]
    if selected:
        return selected

    available_centers = [window[2] for window in windows]
    if not available_centers:
        raise ValueError("No time windows are available for decode-window selection.")
    raise ValueError(
        "No time-window centers fall inside decode_window "
        f"[{decode_start}, {decode_stop}]. Available centers span "
        f"[{min(available_centers)}, {max(available_centers)}]."
    )


def normalize_class_prior_correction(mode: str | None) -> str:
    """Normalize train-fold class-prior correction modes."""

    normalized = "none" if mode is None else str(mode).strip().lower().replace("-", "_")
    if normalized not in CLASS_PRIOR_CORRECTION_CHOICES:
        raise ValueError(
            f"Unknown class_prior_correction '{mode}'. Available modes: "
            f"{', '.join(CLASS_PRIOR_CORRECTION_CHOICES)}."
        )
    return normalized


def _features_for_window(data: np.ndarray, window: TimeWindow) -> np.ndarray:
    start, stop, _center = window
    return data[:, :, start:stop].reshape(data.shape[0], -1)


def _pooled_temporal_training_set(
    feature_cache: dict[TimeWindow, np.ndarray],
    train_windows: list[TimeWindow],
    train_idx: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return fold-local temporal augmentation rows for selected train windows."""

    n_windows = len(train_windows)
    pooled_features = np.concatenate([feature_cache[window][train_idx] for window in train_windows], axis=0)
    pooled_labels = np.tile(labels[train_idx], n_windows)
    pooled_groups = None if groups is None else np.tile(groups[train_idx], n_windows)
    return pooled_features, pooled_labels, pooled_groups


def _estimate_window_feature_bytes(data: np.ndarray, windows: Sequence[TimeWindow]) -> int:
    if not windows:
        return 0
    start, stop, _center = windows[0]
    n_features = int(data.shape[1]) * int(stop - start)
    return int(data.shape[0]) * n_features * len(windows) * np.dtype(data.dtype).itemsize


def _window_feature_batches(
    data: np.ndarray,
    windows: Sequence[TimeWindow],
    *,
    max_bytes: int = MNE_SLIDING_MAX_FEATURE_BYTES,
) -> list[list[TimeWindow]]:
    if not windows:
        return []
    bytes_per_window = max(1, _estimate_window_feature_bytes(data, windows[:1]))
    windows_per_batch = max(1, int(max_bytes // bytes_per_window))
    return [list(windows[start : start + windows_per_batch]) for start in range(0, len(windows), windows_per_batch)]


def _features_for_window_batch(data: np.ndarray, windows: Sequence[TimeWindow]) -> np.ndarray:
    """Return ``(epochs, flattened-window-features, windows)`` for MNE SlidingEstimator."""

    return np.stack([_features_for_window(data, window) for window in windows], axis=-1)


def _iter_mne_sliding_same_time_predictions(
    *,
    data: np.ndarray,
    windows: Sequence[TimeWindow],
    train_labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model,
    emission_mode: str,
    classes: np.ndarray,
) -> list[tuple[TimeWindow, object, np.ndarray]]:
    """Fit one MNE SlidingEstimator batch and return aligned probabilities by window."""

    predictions: list[tuple[TimeWindow, object, np.ndarray]] = []
    for window_batch in _window_feature_batches(data, windows):
        feature_tensor = _features_for_window_batch(data, window_batch)
        sliding = SlidingEstimator(model, scoring="accuracy", verbose=False)
        sliding.fit(feature_tensor[train_idx], train_labels)
        for window_index, time_window in enumerate(window_batch):
            estimator = sliding.estimators_[window_index]
            probabilities = _align_probability_columns(
                predict_emission_probabilities(
                    estimator,
                    feature_tensor[test_idx, :, window_index],
                    emission_mode=emission_mode,
                ),
                model=estimator,
                classes=classes,
            )
            predictions.append((time_window, estimator, probabilities))
    return predictions


def _probability_average(probability_sum: np.ndarray, n_models: int) -> np.ndarray:
    probabilities = probability_sum / float(n_models)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Averaged probabilities must have positive row sums.")
    return probabilities / row_sums


def _apply_class_prior_correction(
    probabilities: np.ndarray,
    train_labels: np.ndarray,
    classes: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Adjust posterior probabilities by train-fold class priors."""

    mode = normalize_class_prior_correction(mode)
    probabilities = np.asarray(probabilities, dtype=float)
    if mode == "none":
        return probabilities

    train_labels = np.asarray(train_labels, dtype=int)
    classes = np.asarray(classes, dtype=int)
    counts = np.asarray([np.count_nonzero(train_labels == class_label) for class_label in classes], dtype=float)
    if counts.sum() <= 0.0:
        raise ValueError("Cannot apply class-prior correction without training labels.")
    priors = counts / counts.sum()
    safe_priors = np.where(priors > 0.0, priors, 1.0)
    corrected = probabilities / safe_priors.reshape(1, -1)
    row_sums = corrected.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Class-prior-corrected probabilities must have positive row sums.")
    return corrected / row_sums


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    """Return top-k accuracy for probability columns aligned to integer labels."""

    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional array.")
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must contain the same number of rows.")
    if probabilities.shape[1] == 0:
        raise ValueError("probabilities must contain at least one class column.")
    if k < 1:
        raise ValueError("k must be at least one.")

    effective_k = min(int(k), probabilities.shape[1])
    top_columns = np.argsort(probabilities, axis=1)[:, ::-1][:, :effective_k]
    return float(np.mean(np.any(top_columns == labels[:, None], axis=1)))


def _model_probability_classes(model) -> np.ndarray | None:
    """Return the class order that corresponds to a model's probability columns."""

    classes = getattr(model, "classes_", None)
    if classes is not None:
        return np.asarray(classes)

    best_estimator = getattr(model, "best_estimator_", None)
    if best_estimator is not None:
        return _model_probability_classes(best_estimator)

    steps = getattr(model, "steps", None)
    if steps:
        return _model_probability_classes(steps[-1][1])

    for attribute in ("estimator", "base_estimator"):
        nested = getattr(model, attribute, None)
        if nested is not None:
            nested_classes = _model_probability_classes(nested)
            if nested_classes is not None:
                return nested_classes

    return None


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Predicted probabilities must be a two-dimensional array.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predicted probabilities must be finite.")
    if np.any(probabilities < 0.0):
        raise ValueError("Predicted probabilities must be non-negative.")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Predicted probabilities must have positive row sums.")
    return probabilities / row_sums


def _align_probability_columns(
    probabilities: np.ndarray,
    *,
    model,
    classes: np.ndarray,
) -> np.ndarray:
    """Align estimator probability columns to the global encoded class order."""

    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes)
    if probabilities.ndim != 2:
        raise ValueError("Predicted probabilities must be a two-dimensional array.")

    model_classes = _model_probability_classes(model)
    if model_classes is None:
        if probabilities.shape[1] != len(classes):
            raise ValueError(
                "Cannot align probability columns because the fitted model does not expose classes_ "
                f"and emitted {probabilities.shape[1]} columns for {len(classes)} global classes."
            )
        return _normalize_probability_rows(probabilities)

    model_classes = np.asarray(model_classes)
    if len(model_classes) != probabilities.shape[1]:
        raise ValueError(
            f"Fitted model reports {len(model_classes)} classes but emitted "
            f"{probabilities.shape[1]} probability columns."
        )
    if len(np.unique(model_classes)) != len(model_classes):
        raise ValueError("Fitted model reports duplicate classes; probability columns are ambiguous.")

    class_to_column = {class_label: class_index for class_index, class_label in enumerate(classes.tolist())}
    aligned = np.zeros((probabilities.shape[0], len(classes)), dtype=float)
    for source_column, class_label in enumerate(model_classes.tolist()):
        try:
            target_column = class_to_column[class_label]
        except KeyError as exc:
            raise ValueError(f"Fitted model emitted unknown class {class_label!r}.") from exc
        aligned[:, target_column] = probabilities[:, source_column]

    return _normalize_probability_rows(aligned)


def _train_window_summary(
    epochs: mne.Epochs,
    train_windows: list[TimeWindow],
) -> tuple[float, float, float]:
    return (
        float(np.mean([window[2] for window in train_windows])),
        float(min(epochs.times[window[0]] for window in train_windows)),
        float(max(epochs.times[window[1] - 1] for window in train_windows)),
    )


def _best_time_by_metric(time_summary: pd.DataFrame, metric: str) -> float:
    if metric not in RESULT_SELECTION_METRIC_CHOICES:
        raise ValueError(f"Unknown selection metric '{metric}'.")
    if metric in RESULT_SELECTION_MINIMIZE_METRICS:
        return float(time_summary[metric].idxmin())
    return float(time_summary[metric].idxmax())


def _model_hash(
    *,
    decoder_name: str,
    emission_mode: str,
    max_iter: int,
    feature_preprocessor: str,
    pca_components: int | float | None,
    normalization: str,
    baseline_window: tuple[float, float],
    temporal_mode: str,
    temporal_train_window: TemporalTrainWindow | None,
    train_window_centers: list[float] | None = None,
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int | None = None,
    tuning_scoring: str | None = None,
    tuning_c_grid: Sequence[float] | None = None,
    tuning_metadata: dict[str, object] | None = None,
    backend: str = "sklearn",
    class_prior_correction: str = "none",
    label_shuffle_control: bool = False,
    label_shuffle_seed: int = 13,
) -> str:
    payload: dict[str, object] = {
        "backend": backend,
        "decoder": decoder_name,
        "emission_mode": emission_mode,
        "max_iter": max_iter,
        "feature_preprocessor": feature_preprocessor,
        "pca_components": pca_components,
        "normalization": normalization,
        "baseline_window": baseline_window,
        "temporal_mode": temporal_mode,
        "temporal_train_window": temporal_train_window,
        "train_window_centers": train_window_centers,
    }
    if class_prior_correction != "none":
        payload["class_prior_correction"] = class_prior_correction
    if tune_hyperparameters:
        payload.update(
            {
                "tune_hyperparameters": True,
                "tuning_cv_splits": tuning_cv_splits,
                "tuning_scoring": tuning_scoring,
                "tuning_c_grid": tuple(tuning_c_grid or ()),
                "best_params": (tuning_metadata or {}).get("best_params", ""),
            }
        )
    if label_shuffle_control:
        payload.update(
            {
                "label_shuffle_control": True,
                "label_shuffle_seed": int(label_shuffle_seed),
            }
        )
    return stable_hash(payload)


def _append_decoded_outputs(
    *,
    rows: list[dict],
    calibration_rows: list[dict],
    observation_rows: list[dict],
    probabilities: np.ndarray,
    test_labels: np.ndarray,
    test_idx: np.ndarray,
    original_indices: np.ndarray,
    session_values: np.ndarray | None,
    groups: np.ndarray | None,
    group_column: str | None,
    classes: np.ndarray,
    class_names: np.ndarray,
    fold: int,
    n_train: int,
    decoder_name: str,
    emission_mode: str,
    feature_preprocessor_name: str,
    pca_components_value: int | float | None,
    normalization_name: str,
    baseline_window: tuple[float, float],
    time_window: TimeWindow,
    epochs: mne.Epochs,
    split_id: str,
    preprocessing_hash: str,
    model_hash: str,
    temporal_mode: str,
    temporal_train_window: TemporalTrainWindow | None,
    train_time: float,
    train_window_start: float,
    train_window_stop: float,
    n_train_windows: int,
    calibration_out_path: Path | None,
    calibration_bins: int,
    observation_out_path: Path | None,
    subject: str | None,
    tuning_metadata: dict[str, object] | None = None,
    backend: str = "sklearn",
    class_prior_correction: str = "none",
    label_shuffle_control: bool = False,
    label_shuffle_seed: int = 13,
    outer_test_groups: Sequence[str] = (),
) -> None:
    tuning_metadata = {} if tuning_metadata is None else tuning_metadata
    start, stop, center = time_window
    predictions = probabilities.argmax(axis=1)
    common = {
        "fold": fold,
        "decoder": decoder_name,
        "emission_mode": emission_mode,
        "feature_preprocessor": feature_preprocessor_name,
        "pca_components": "" if pca_components_value is None else pca_components_value,
        "normalization": normalization_name,
        "baseline_window_start": baseline_window[0],
        "baseline_window_stop": baseline_window[1],
        "temporal_mode": temporal_mode,
        "temporal_train_window_start": "" if temporal_train_window is None else temporal_train_window[0],
        "temporal_train_window_stop": "" if temporal_train_window is None else temporal_train_window[1],
        "train_time": train_time,
        "time": center,
        "test_time": center,
        "train_window_start": train_window_start,
        "train_window_stop": train_window_stop,
        "n_train_windows": n_train_windows,
        "window_start": float(epochs.times[start]),
        "window_stop": float(epochs.times[stop - 1]),
        "label_shuffle_control": bool(label_shuffle_control),
        "label_shuffle_seed": int(label_shuffle_seed),
        "class_prior_correction": class_prior_correction,
        "outer_test_groups": "|".join(outer_test_groups),
    }
    row = {
        **common,
        "accuracy": accuracy_score(test_labels, predictions),
        "balanced_accuracy": balanced_accuracy_score(test_labels, predictions),
        "top2_accuracy": _top_k_accuracy(probabilities, test_labels, k=2),
        "top3_accuracy": _top_k_accuracy(probabilities, test_labels, k=3),
        "log_loss": log_loss(test_labels, probabilities, labels=classes),
        "brier": brier_score_multiclass(probabilities, test_labels),
        "ece": expected_calibration_error(probabilities, test_labels),
        "n_train": n_train,
        "n_test": len(test_idx),
        "n_classes": len(classes),
        "class_names": "|".join(map(str, class_names)),
    }
    row.update(tuning_metadata)
    rows.append(_add_subject(row, subject))

    if calibration_out_path is not None:
        for bin_row in reliability_bins(probabilities, test_labels, n_bins=calibration_bins):
            calibration_row = {**common, **bin_row}
            calibration_row.update(tuning_metadata)
            calibration_rows.append(_add_subject(calibration_row, subject))
    if observation_out_path is not None:
        for local_position, filtered_index in enumerate(test_idx):
            true_label = int(test_labels[local_position])
            predicted_label = int(predictions[local_position])
            observation = {
                **common,
                "split_id": split_id,
                "seed": 13,
                "backend": backend,
                "sample_index": int(original_indices[filtered_index]),
                "sequence_id": int(original_indices[filtered_index]),
                "session": "" if session_values is None else session_values[filtered_index],
                "true_label": true_label,
                "true_class": str(class_names[true_label]),
                "predicted_label": predicted_label,
                "predicted_class": str(class_names[predicted_label]),
                "probability_true_class": float(probabilities[local_position, true_label]),
                "confidence": float(probabilities[local_position].max()),
                "is_correct": bool(predicted_label == true_label),
                "calibration_fold": "",
                "preprocessing_hash": preprocessing_hash,
                "model_hash": model_hash,
            }
            if group_column is not None:
                observation["group"] = groups[filtered_index] if groups is not None else ""
            observation.update(tuning_metadata)
            for class_index, class_name in enumerate(class_names):
                observation[f"class_{class_index}"] = str(class_name)
                observation[f"prob_class_{class_index}"] = float(probabilities[local_position, class_index])
            observation_rows.append(_add_subject(observation, subject))


def run_time_resolved_decode(
    epochs_path: Path,
    label_column: str,
    out_path: Path,
    *,
    metadata_csv: Path | None = None,
    input_format: str = "mne-epochs",
    fieldtrip_root_path: str | None = None,
    fieldtrip_label_base: float | None = 1.0,
    fieldtrip_ch_type: str = "grad",
    fieldtrip_trim_overlong_labels: bool = True,
    group_column: str | None = None,
    outer_test_groups: Sequence[object] | str | None = None,
    picks: str = "data",
    tmin: float | None = None,
    tmax: float | None = None,
    window_ms: float = 20.0,
    step_ms: float = 10.0,
    n_splits: int = 5,
    max_iter: int = 1000,
    decoder: str = "logistic",
    emission_mode: str = "calibrated",
    feature_preprocessor: str = "none",
    pca_components: int | float | str | None = None,
    normalization: str = "none",
    baseline_window: tuple[float, float] | None = DEFAULT_BASELINE_WINDOW,
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int = 3,
    tuning_scoring: str = "accuracy",
    tuning_c_grid: Sequence[float] | str | None = None,
    calibration_out_path: Path | None = None,
    calibration_bins: int = 10,
    observation_out_path: Path | None = None,
    subject: str | None = None,
    decode_window: tuple[float, float] | None = None,
    temporal_train_window: tuple[float, float] | None = None,
    temporal_train_mode: str = "window_ensemble",
    time_decode_backend: str = "auto",
    class_prior_correction: str = "none",
    label_shuffle_control: bool = False,
    label_shuffle_seed: int = 13,
) -> pd.DataFrame:
    """Run time-resolved decoding on an MNE epochs file and save metrics as CSV.

    ``decode_window`` restricts the test-time window centers that are evaluated.
    If ``temporal_train_window`` is set, selected train-time windows are used
    according to ``temporal_train_mode``. ``window_ensemble`` trains one model
    per selected train-time window and averages probabilities. ``pooled`` stacks
    the selected source-subject train windows as fold-local temporal
    augmentation and trains one classifier for all test times. Without a
    temporal train window, the historical diagonal
    train-time == test-time decoding path is used.
    """
    epochs, metadata = _load_epochs_and_metadata(
        epochs_path,
        metadata_csv,
        input_format=input_format,
        label_column=label_column,
        fieldtrip_root_path=fieldtrip_root_path,
        fieldtrip_label_base=fieldtrip_label_base,
        fieldtrip_trim_overlong_labels=fieldtrip_trim_overlong_labels,
        fieldtrip_ch_type=fieldtrip_ch_type,
    )
    decoder_name = normalize_decoder_name(decoder)
    emission_modes = list(EMISSION_MODE_CHOICES) if emission_mode == "both" else [normalize_emission_mode(emission_mode)]
    feature_preprocessor_name = normalize_feature_preprocessor(feature_preprocessor)
    normalization_name = normalize_epoch_normalization(normalization)
    baseline_window_value = _normalize_baseline_window(baseline_window)
    if feature_preprocessor_name == "none" and pca_components is not None:
        raise ValueError(
            "pca_components can only be set when feature_preprocessor is 'pca', 'pca_whiten', 'anova_select', or 'pls_da'."
        )
    if feature_preprocessor_name == "anova_select":
        pca_components_value = normalize_anova_select_percentile(pca_components)
    elif feature_preprocessor_name == "pls_da":
        pca_components_value = normalize_pls_components(pca_components)
    elif feature_preprocessor_name != "none":
        pca_components_value = normalize_pca_components(pca_components)
    else:
        pca_components_value = None
    tuning_scoring = normalize_tuning_scoring(tuning_scoring)
    tuning_c_grid_values = parse_c_grid(tuning_c_grid)
    normalized_decode_window = _normalize_decode_window(decode_window)
    normalized_temporal_train_window = _normalize_temporal_train_window(temporal_train_window)
    temporal_train_mode_name = _normalize_temporal_train_mode(temporal_train_mode)
    class_prior_correction_name = normalize_class_prior_correction(class_prior_correction)
    requested_time_decode_backend = normalize_time_decode_backend(time_decode_backend)
    label_shuffle_control = bool(label_shuffle_control)
    label_shuffle_seed = int(label_shuffle_seed)
    outer_test_groups_value = _normalize_outer_test_groups(outer_test_groups)
    if requested_time_decode_backend == "mne" and normalized_temporal_train_window is not None:
        raise ValueError("The MNE time-decode backend currently supports same-time decoding only.")
    time_decode_backend = (
        "sklearn"
        if requested_time_decode_backend == "auto" and normalized_temporal_train_window is not None
        else "mne"
        if requested_time_decode_backend == "auto"
        else requested_time_decode_backend
    )

    if label_column not in metadata.columns:
        raise ValueError(f"Label column '{label_column}' not found in metadata.")
    if group_column is not None and group_column not in metadata.columns:
        raise ValueError(f"Group column '{group_column}' not found in metadata.")

    epochs = epochs.copy().pick(picks)
    if tmin is not None or tmax is not None:
        epochs.crop(tmin=tmin, tmax=tmax)

    raw_labels = metadata[label_column].to_numpy()
    keep = pd.notna(raw_labels)
    original_indices = np.arange(len(raw_labels))[keep]
    epochs = epochs[keep]
    raw_labels = raw_labels[keep]
    metadata = metadata.loc[keep].reset_index(drop=True)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(raw_labels)
    groups = metadata[group_column].to_numpy() if group_column else None
    session_values = metadata["session"].to_numpy() if "session" in metadata.columns else groups
    splitter_name = "stratified-group-kfold" if groups is not None else "stratified-kfold"
    split_id = f"{splitter_name}-{n_splits}"
    if normalized_temporal_train_window is None:
        temporal_mode = "same_time"
    elif temporal_train_mode_name == "pooled":
        temporal_mode = "train_window_pooled"
    else:
        temporal_mode = "train_window_ensemble"
    preprocessing_hash = stable_hash(
        {
            "picks": picks,
            "tmin": tmin,
            "tmax": tmax,
            "window_ms": window_ms,
            "step_ms": step_ms,
            "feature_preprocessor": feature_preprocessor_name,
            "pca_components": pca_components_value,
            "normalization": normalization_name,
            "baseline_window": baseline_window_value,
            "decode_window": normalized_decode_window,
            "temporal_train_window": normalized_temporal_train_window,
            "temporal_train_mode": None if normalized_temporal_train_window is None else temporal_train_mode_name,
            "class_prior_correction": class_prior_correction_name,
            "outer_test_groups": outer_test_groups_value,
        }
    )
    default_model_hash = _model_hash(
        decoder_name=decoder_name,
        emission_mode=emission_mode,
        max_iter=max_iter,
        feature_preprocessor=feature_preprocessor_name,
        pca_components=pca_components_value,
        normalization=normalization_name,
        baseline_window=baseline_window_value,
        temporal_mode=temporal_mode,
        temporal_train_window=normalized_temporal_train_window,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv_splits=tuning_cv_splits,
        tuning_scoring=tuning_scoring,
        tuning_c_grid=tuning_c_grid_values,
        backend=time_decode_backend,
        class_prior_correction=class_prior_correction_name,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )

    raw_data = epochs.get_data(copy=False)
    data = _apply_epoch_normalization(
        raw_data,
        epochs.times,
        normalization_name,
        baseline_window=baseline_window_value,
    )
    classes = np.arange(len(encoder.classes_))
    rows: list[dict] = []
    calibration_rows: list[dict] = []
    observation_rows: list[dict] = []
    all_windows = time_windows(epochs.times, window_ms=window_ms, step_ms=step_ms)
    windows = _select_decode_windows(all_windows, normalized_decode_window)
    selected_train_windows = _select_temporal_train_windows(all_windows, normalized_temporal_train_window)
    splits = _filter_splits_for_outer_test_groups(
        list(enumerate(make_cross_validator(labels, groups, n_splits))),
        groups,
        outer_test_groups_value,
    )

    if selected_train_windows is None and time_decode_backend == "mne":
        for fold, (train_idx, test_idx) in splits:
            test_labels = labels[test_idx]
            train_labels = _fold_training_labels(
                labels,
                train_idx,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
                context=(split_id, fold, "mne"),
            )
            for current_emission_mode in emission_modes:
                tuning_cv = (
                    make_tuning_cross_validator(train_labels, None if groups is None else groups[train_idx], tuning_cv_splits)
                    if tune_hyperparameters
                    else 3
                )
                model = make_decoder(
                    decoder_name,
                    max_iter=max_iter,
                    emission_mode=current_emission_mode,
                    feature_preprocessor=feature_preprocessor_name,
                    pca_components=pca_components_value,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv=tuning_cv,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                )
                for time_window, fitted_model, probabilities in _iter_mne_sliding_same_time_predictions(
                    data=data,
                    windows=windows,
                    train_labels=train_labels,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    model=model,
                    emission_mode=current_emission_mode,
                    classes=classes,
                ):
                    probabilities = _apply_class_prior_correction(
                        probabilities,
                        train_labels,
                        classes,
                        class_prior_correction_name,
                    )
                    start, stop, center = time_window
                    tuning_metadata = _tuning_metadata(
                        fitted_model,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                    )
                    current_model_hash = _model_hash(
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        max_iter=max_iter,
                        feature_preprocessor=feature_preprocessor_name,
                        pca_components=pca_components_value,
                        normalization=normalization_name,
                        baseline_window=baseline_window_value,
                        temporal_mode=temporal_mode,
                        temporal_train_window=None,
                        train_window_centers=[center],
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                        tuning_metadata=tuning_metadata,
                        backend=time_decode_backend,
                        class_prior_correction=class_prior_correction_name,
                        label_shuffle_control=label_shuffle_control,
                        label_shuffle_seed=label_shuffle_seed,
                    )
                    _append_decoded_outputs(
                        rows=rows,
                        calibration_rows=calibration_rows,
                        observation_rows=observation_rows,
                        probabilities=probabilities,
                        test_labels=test_labels,
                        test_idx=test_idx,
                        original_indices=original_indices,
                        session_values=session_values,
                        groups=groups,
                        group_column=group_column,
                        classes=classes,
                        class_names=encoder.classes_,
                        fold=fold,
                        n_train=len(train_idx),
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        feature_preprocessor_name=feature_preprocessor_name,
                        pca_components_value=pca_components_value,
                        normalization_name=normalization_name,
                        baseline_window=baseline_window_value,
                        time_window=time_window,
                        epochs=epochs,
                        split_id=split_id,
                        preprocessing_hash=preprocessing_hash,
                        model_hash=current_model_hash,
                        temporal_mode=temporal_mode,
                        temporal_train_window=normalized_temporal_train_window,
                        train_time=center,
                        train_window_start=float(epochs.times[start]),
                        train_window_stop=float(epochs.times[stop - 1]),
                        n_train_windows=1,
                        calibration_out_path=calibration_out_path,
                        calibration_bins=calibration_bins,
                        observation_out_path=observation_out_path,
                        subject=subject,
                        tuning_metadata=tuning_metadata,
                        backend=time_decode_backend,
                        class_prior_correction=class_prior_correction_name,
                        label_shuffle_control=label_shuffle_control,
                        label_shuffle_seed=label_shuffle_seed,
                        outer_test_groups=outer_test_groups_value,
                    )
    elif selected_train_windows is None:
        for time_window in windows:
            features = _features_for_window(data, time_window)
            start, stop, center = time_window
            for fold, (train_idx, test_idx) in splits:
                test_labels = labels[test_idx]
                train_labels = _fold_training_labels(
                    labels,
                    train_idx,
                    label_shuffle_control=label_shuffle_control,
                    label_shuffle_seed=label_shuffle_seed,
                    context=(split_id, fold, "same_time"),
                )
                for current_emission_mode in emission_modes:
                    tuning_cv = (
                        make_tuning_cross_validator(train_labels, None if groups is None else groups[train_idx], tuning_cv_splits)
                        if tune_hyperparameters
                        else 3
                    )
                    model = make_decoder(
                        decoder_name,
                        max_iter=max_iter,
                        emission_mode=current_emission_mode,
                        feature_preprocessor=feature_preprocessor_name,
                        pca_components=pca_components_value,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv=tuning_cv,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                    )
                    model.fit(features[train_idx], train_labels)

                    probabilities = _align_probability_columns(
                        predict_emission_probabilities(
                            model,
                            features[test_idx],
                            emission_mode=current_emission_mode,
                        ),
                        model=model,
                        classes=classes,
                    )
                    probabilities = _apply_class_prior_correction(
                        probabilities,
                        train_labels,
                        classes,
                        class_prior_correction_name,
                    )
                    tuning_metadata = _tuning_metadata(
                        model,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                    )
                    current_model_hash = _model_hash(
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        max_iter=max_iter,
                        feature_preprocessor=feature_preprocessor_name,
                        pca_components=pca_components_value,
                        normalization=normalization_name,
                        baseline_window=baseline_window_value,
                        temporal_mode=temporal_mode,
                        temporal_train_window=None,
                        train_window_centers=[center],
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                        tuning_metadata=tuning_metadata,
                        backend=time_decode_backend,
                        class_prior_correction=class_prior_correction_name,
                        label_shuffle_control=label_shuffle_control,
                        label_shuffle_seed=label_shuffle_seed,
                    )
                    _append_decoded_outputs(
                        rows=rows,
                        calibration_rows=calibration_rows,
                        observation_rows=observation_rows,
                        probabilities=probabilities,
                        test_labels=test_labels,
                        test_idx=test_idx,
                        original_indices=original_indices,
                        session_values=session_values,
                        groups=groups,
                        group_column=group_column,
                        classes=classes,
                        class_names=encoder.classes_,
                        fold=fold,
                        n_train=len(train_idx),
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        feature_preprocessor_name=feature_preprocessor_name,
                        pca_components_value=pca_components_value,
                        normalization_name=normalization_name,
                        baseline_window=baseline_window_value,
                        time_window=time_window,
                        epochs=epochs,
                        split_id=split_id,
                        preprocessing_hash=preprocessing_hash,
                        model_hash=current_model_hash,
                        temporal_mode=temporal_mode,
                        temporal_train_window=normalized_temporal_train_window,
                        train_time=center,
                        train_window_start=float(epochs.times[start]),
                        train_window_stop=float(epochs.times[stop - 1]),
                        n_train_windows=1,
                        calibration_out_path=calibration_out_path,
                        calibration_bins=calibration_bins,
                        observation_out_path=observation_out_path,
                        subject=subject,
                        tuning_metadata=tuning_metadata,
                        backend=time_decode_backend,
                        class_prior_correction=class_prior_correction_name,
                        label_shuffle_control=label_shuffle_control,
                        label_shuffle_seed=label_shuffle_seed,
                        outer_test_groups=outer_test_groups_value,
                    )
    elif temporal_train_mode_name == "pooled":
        train_time, train_window_start, train_window_stop = _train_window_summary(epochs, selected_train_windows)
        train_window_centers = [window[2] for window in selected_train_windows]
        model_windows = list(dict.fromkeys([*windows, *selected_train_windows]))
        feature_cache = {time_window: _features_for_window(data, time_window) for time_window in model_windows}
        for fold, (train_idx, test_idx) in splits:
            test_labels = labels[test_idx]
            train_labels = _fold_training_labels(
                labels,
                train_idx,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
                context=(split_id, fold, "pooled", *train_window_centers),
            )
            fold_labels = labels.copy()
            fold_labels[train_idx] = train_labels
            pooled_train_features, pooled_train_labels, pooled_train_groups = _pooled_temporal_training_set(
                feature_cache,
                selected_train_windows,
                train_idx,
                fold_labels,
                groups,
            )
            for current_emission_mode in emission_modes:
                tuning_cv = (
                    make_tuning_cross_validator(pooled_train_labels, pooled_train_groups, tuning_cv_splits)
                    if tune_hyperparameters
                    else 3
                )
                model = make_decoder(
                    decoder_name,
                    max_iter=max_iter,
                    emission_mode=current_emission_mode,
                    feature_preprocessor=feature_preprocessor_name,
                    pca_components=pca_components_value,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv=tuning_cv,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                )
                model.fit(pooled_train_features, pooled_train_labels)
                tuning_metadata = _tuning_metadata(
                    model,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                )
                current_model_hash = _model_hash(
                    decoder_name=decoder_name,
                    emission_mode=current_emission_mode,
                    max_iter=max_iter,
                    feature_preprocessor=feature_preprocessor_name,
                    pca_components=pca_components_value,
                    normalization=normalization_name,
                    baseline_window=baseline_window_value,
                    temporal_mode=temporal_mode,
                    temporal_train_window=normalized_temporal_train_window,
                    train_window_centers=train_window_centers,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                    tuning_metadata=tuning_metadata,
                    label_shuffle_control=label_shuffle_control,
                    label_shuffle_seed=label_shuffle_seed,
                    class_prior_correction=class_prior_correction_name,
                )
                for test_window in windows:
                    probabilities = _align_probability_columns(
                        predict_emission_probabilities(
                            model,
                            feature_cache[test_window][test_idx],
                            emission_mode=current_emission_mode,
                        ),
                        model=model,
                        classes=classes,
                    )
                    probabilities = _apply_class_prior_correction(
                        probabilities,
                        train_labels,
                        classes,
                        class_prior_correction_name,
                    )
                    _append_decoded_outputs(
                        rows=rows,
                        calibration_rows=calibration_rows,
                        observation_rows=observation_rows,
                        probabilities=probabilities,
                        test_labels=test_labels,
                        test_idx=test_idx,
                        original_indices=original_indices,
                        session_values=session_values,
                        groups=groups,
                        group_column=group_column,
                        classes=classes,
                        class_names=encoder.classes_,
                        fold=fold,
                        n_train=len(pooled_train_labels),
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        feature_preprocessor_name=feature_preprocessor_name,
                        pca_components_value=pca_components_value,
                        normalization_name=normalization_name,
                        baseline_window=baseline_window_value,
                        time_window=test_window,
                        epochs=epochs,
                        split_id=split_id,
                        preprocessing_hash=preprocessing_hash,
                        model_hash=current_model_hash,
                        temporal_mode=temporal_mode,
                        temporal_train_window=normalized_temporal_train_window,
                        train_time=train_time,
                        train_window_start=train_window_start,
                        train_window_stop=train_window_stop,
                        n_train_windows=len(selected_train_windows),
                        calibration_out_path=calibration_out_path,
                        calibration_bins=calibration_bins,
                        observation_out_path=observation_out_path,
                        subject=subject,
                        tuning_metadata=tuning_metadata,
                        class_prior_correction=class_prior_correction_name,
                        label_shuffle_control=label_shuffle_control,
                        label_shuffle_seed=label_shuffle_seed,
                        outer_test_groups=outer_test_groups_value,
                    )
    else:
        train_time, train_window_start, train_window_stop = _train_window_summary(epochs, selected_train_windows)
        train_window_centers = [window[2] for window in selected_train_windows]
        model_windows = list(dict.fromkeys([*windows, *selected_train_windows]))
        feature_cache = {time_window: _features_for_window(data, time_window) for time_window in model_windows}
        for fold, (train_idx, test_idx) in splits:
            test_labels = labels[test_idx]
            train_labels = _fold_training_labels(
                labels,
                train_idx,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
                context=(split_id, fold, "train_window_ensemble", *train_window_centers),
            )
            for current_emission_mode in emission_modes:
                tuning_cv = (
                    make_tuning_cross_validator(train_labels, None if groups is None else groups[train_idx], tuning_cv_splits)
                    if tune_hyperparameters
                    else 3
                )
                fitted_models = []
                probability_sums = {
                    time_window: np.zeros((len(test_idx), len(classes)), dtype=float)
                    for time_window in windows
                }
                for train_window in selected_train_windows:
                    train_features = feature_cache[train_window]
                    model = make_decoder(
                        decoder_name,
                        max_iter=max_iter,
                        emission_mode=current_emission_mode,
                        feature_preprocessor=feature_preprocessor_name,
                        pca_components=pca_components_value,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv=tuning_cv,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                    )
                    model.fit(train_features[train_idx], train_labels)
                    fitted_models.append(model)
                    for test_window in windows:
                        probability_sums[test_window] += _align_probability_columns(
                            predict_emission_probabilities(
                                model,
                                feature_cache[test_window][test_idx],
                                emission_mode=current_emission_mode,
                            ),
                            model=model,
                            classes=classes,
                        )

                tuning_metadata = _tuning_metadata(
                    fitted_models,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                )
                current_model_hash = _model_hash(
                    decoder_name=decoder_name,
                    emission_mode=current_emission_mode,
                    max_iter=max_iter,
                    feature_preprocessor=feature_preprocessor_name,
                    pca_components=pca_components_value,
                    normalization=normalization_name,
                    baseline_window=baseline_window_value,
                    temporal_mode=temporal_mode,
                    temporal_train_window=normalized_temporal_train_window,
                    train_window_centers=train_window_centers,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                    tuning_metadata=tuning_metadata,
                    backend=time_decode_backend,
                    class_prior_correction=class_prior_correction_name,
                    label_shuffle_control=label_shuffle_control,
                    label_shuffle_seed=label_shuffle_seed,
                )
                for test_window in windows:
                    probabilities = _probability_average(probability_sums[test_window], len(selected_train_windows))
                    probabilities = _apply_class_prior_correction(
                        probabilities,
                        train_labels,
                        classes,
                        class_prior_correction_name,
                    )
                    _append_decoded_outputs(
                        rows=rows,
                        calibration_rows=calibration_rows,
                        observation_rows=observation_rows,
                        probabilities=probabilities,
                        test_labels=test_labels,
                        test_idx=test_idx,
                        original_indices=original_indices,
                        session_values=session_values,
                        groups=groups,
                        group_column=group_column,
                        classes=classes,
                        class_names=encoder.classes_,
                        fold=fold,
                        n_train=len(train_idx),
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        feature_preprocessor_name=feature_preprocessor_name,
                        pca_components_value=pca_components_value,
                        normalization_name=normalization_name,
                        baseline_window=baseline_window_value,
                        time_window=test_window,
                        epochs=epochs,
                        split_id=split_id,
                        preprocessing_hash=preprocessing_hash,
                        model_hash=current_model_hash,
                        temporal_mode=temporal_mode,
                        temporal_train_window=normalized_temporal_train_window,
                        train_time=train_time,
                        train_window_start=train_window_start,
                        train_window_stop=train_window_stop,
                        n_train_windows=len(selected_train_windows),
                        calibration_out_path=calibration_out_path,
                        calibration_bins=calibration_bins,
                        observation_out_path=observation_out_path,
                        subject=subject,
                        tuning_metadata=tuning_metadata,
                        backend=time_decode_backend,
                        class_prior_correction=class_prior_correction_name,
                        label_shuffle_control=label_shuffle_control,
                        label_shuffle_seed=label_shuffle_seed,
                        outer_test_groups=outer_test_groups_value,
                    )

    results = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    if calibration_out_path is not None:
        calibration_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(calibration_rows).to_csv(calibration_out_path, index=False)
    if observation_out_path is not None:
        ProbabilityObservationTable(pd.DataFrame(observation_rows)).standardized(
            defaults={
                "backend": time_decode_backend,
                "split_id": split_id,
                "seed": 13,
                "calibration_fold": "",
                "preprocessing_hash": preprocessing_hash,
                "model_hash": default_model_hash,
            }
        ).to_csv(observation_out_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run calibrated time-resolved decoding on MNE Epochs FIF or FieldTrip raw MATLAB input."
    )
    parser.add_argument("--epochs", type=Path, required=True, help="Input MNE Epochs FIF file or FieldTrip raw MATLAB .mat file.")
    parser.add_argument(
        "--input-format",
        choices=INPUT_FORMAT_CHOICES,
        default="mne-epochs",
        help="Input container/structure. Use fieldtrip-mat for PyMEGDec/Bush FieldTrip-style .mat files.",
    )
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument(
        "--fieldtrip-root-path",
        default=",".join(map(str, FIELDTRIP_DEFAULT_ROOT_PATH)),
        help="Comma-separated path to the FieldTrip raw struct inside a MATLAB file. Default: data,0.",
    )
    parser.add_argument("--fieldtrip-label-base", type=int, default=1, help="Label base used by trialinfo labels in FieldTrip input.")
    parser.add_argument(
        "--fieldtrip-no-trim-overlong-labels",
        action="store_true",
        help="Fail instead of trimming overlong FieldTrip channel metadata to the trial channel count.",
    )
    parser.add_argument("--fieldtrip-ch-type", default="grad", help="MNE channel type used for FieldTrip trial rows.")
    parser.add_argument("--group-column")
    parser.add_argument(
        "--outer-test-group",
        action="append",
        dest="outer_test_groups",
        help="Restrict decoding to outer folds whose held-out group matches this value. Repeat for multiple groups.",
    )
    parser.add_argument("--picks", default="data")
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--step-ms", type=float, default=10.0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--decoder", choices=DECODER_CLI_CHOICES, default="logistic")
    parser.add_argument("--emission-mode", choices=EMISSION_RUN_CHOICES, default="calibrated")
    parser.add_argument("--feature-preprocessor", choices=FEATURE_PREPROCESSOR_RUN_CHOICES, default="none")
    parser.add_argument(
        "--pca-components",
        help=(
            "PCA component count or explained-variance fraction. With "
            "--feature-preprocessor anova-select, this is the selected feature percentile."
        ),
    )
    parser.add_argument(
        "--normalization",
        choices=EPOCH_NORMALIZATION_RUN_CHOICES,
        default="none",
        help="Subject-level epoch normalization applied before time-window feature extraction.",
    )
    parser.add_argument(
        "--baseline-window",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        default=DEFAULT_BASELINE_WINDOW,
        help="Baseline time window in seconds for subject_baseline_z and subject_baseline_whiten.",
    )
    parser.add_argument("--tune-hyperparameters", action="store_true", help="Use nested inner-CV hyperparameter selection inside each outer train fold.")
    parser.add_argument("--tuning-cv-splits", type=int, default=3, help="Maximum number of inner CV folds for --tune-hyperparameters.")
    parser.add_argument("--tuning-scoring", choices=TUNING_SCORING_CHOICES, default="accuracy", help="Inner-CV objective for --tune-hyperparameters.")
    parser.add_argument("--selection-metric", choices=RESULT_SELECTION_METRIC_CHOICES, default="accuracy", help="Metric used only for the console 'best time' summary.")
    parser.add_argument(
        "--tuning-c-grid",
        default=",".join(str(value) for value in parse_c_grid(None)),
        help="Comma-separated positive C values for tuned logistic regression and linear SVM.",
    )
    parser.add_argument("--calibration-out", type=Path)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--observations-out", type=Path, help="Optional held-out trial/time probability observation CSV.")
    parser.add_argument("--subject", help="Optional subject identifier to include in output CSVs.")
    parser.add_argument(
        "--label-shuffle-control",
        action="store_true",
        help="Shuffle training labels inside each outer fold as a deterministic null control. Test labels and splits stay unchanged.",
    )
    parser.add_argument("--label-shuffle-seed", type=int, default=13, help="Seed for --label-shuffle-control.")
    parser.add_argument(
        "--time-decode-backend",
        choices=TIME_DECODE_BACKEND_CHOICES,
        default="auto",
        help="Implementation backend. auto uses mne.decoding.SlidingEstimator for same-time decoding and sklearn for temporal train-window decoding.",
    )
    parser.add_argument(
        "--class-prior-correction",
        choices=CLASS_PRIOR_CORRECTION_RUN_CHOICES,
        default="none",
        help="Optional train-fold prior correction. train_uniform divides posterior probabilities by train-fold class priors before scoring.",
    )
    parser.add_argument(
        "--decode-window",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        help="Evaluate only time-window centers in START..STOP seconds.",
    )
    parser.add_argument(
        "--temporal-train-window",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        help=(
            "Use time-window centers in START..STOP seconds for non-diagonal "
            "temporal training. The behavior is controlled by --temporal-train-mode."
        ),
    )
    parser.add_argument(
        "--temporal-train-mode",
        choices=TEMPORAL_TRAIN_MODE_RUN_CHOICES,
        default="window_ensemble",
        help=(
            "How --temporal-train-window is used: window_ensemble fits one model "
            "per selected train window; pooled stacks selected train windows as "
            "fold-local temporal augmentation and fits one model."
        ),
    )
    parser.add_argument(
        "--fieldtrip-root-path",
        default="data,0",
        help="Comma-separated path to the FieldTrip raw struct inside a .mat file. Default: data,0.",
    )
    parser.add_argument("--fieldtrip-label-base", type=float, default=1.0, help="Subtract this value from numeric trialinfo labels for FieldTrip MAT input.")
    parser.add_argument("--fieldtrip-ch-type", default="grad", help="MNE channel type assigned to FieldTrip trial rows.")
    parser.add_argument(
        "--fieldtrip-no-trim-overlong-labels",
        action="store_true",
        help="Fail instead of trimming overlong FieldTrip channel-level metadata to the trial channel count.",
    )
    args = parser.parse_args()

    results = run_time_resolved_decode(
        epochs_path=args.epochs,
        metadata_csv=args.metadata_csv,
        input_format=args.input_format,
        fieldtrip_root_path=args.fieldtrip_root_path,
        fieldtrip_label_base=args.fieldtrip_label_base,
        fieldtrip_ch_type=args.fieldtrip_ch_type,
        fieldtrip_trim_overlong_labels=not args.fieldtrip_no_trim_overlong_labels,
        label_column=args.label_column,
        group_column=args.group_column,
        outer_test_groups=tuple(args.outer_test_groups) if args.outer_test_groups is not None else None,
        out_path=args.out,
        picks=args.picks,
        tmin=args.tmin,
        tmax=args.tmax,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
        n_splits=args.n_splits,
        max_iter=args.max_iter,
        decoder=args.decoder,
        emission_mode=args.emission_mode,
        feature_preprocessor=args.feature_preprocessor,
        pca_components=args.pca_components,
        normalization=args.normalization,
        baseline_window=tuple(args.baseline_window),
        tune_hyperparameters=args.tune_hyperparameters,
        tuning_cv_splits=args.tuning_cv_splits,
        tuning_scoring=args.tuning_scoring,
        tuning_c_grid=args.tuning_c_grid,
        calibration_out_path=args.calibration_out,
        calibration_bins=args.calibration_bins,
        observation_out_path=args.observations_out,
        subject=args.subject,
        decode_window=tuple(args.decode_window) if args.decode_window is not None else None,
        temporal_train_window=tuple(args.temporal_train_window) if args.temporal_train_window is not None else None,
        temporal_train_mode=args.temporal_train_mode,
        time_decode_backend=args.time_decode_backend,
        class_prior_correction=args.class_prior_correction,
        label_shuffle_control=args.label_shuffle_control,
        label_shuffle_seed=args.label_shuffle_seed,
    )
    print(f"Wrote {args.out}")
    if args.observations_out is not None:
        print(f"Wrote probability observations: {args.observations_out}")
    for emission_mode_name, summary in results.groupby("emission_mode", sort=True):
        time_summary = summary.groupby("time")[list(RESULT_SUMMARY_METRIC_COLUMNS)].mean()
        best_time = _best_time_by_metric(time_summary, args.selection_metric)
        best_value = time_summary.loc[best_time, args.selection_metric]
        direction = "lowest" if args.selection_metric in RESULT_SELECTION_MINIMIZE_METRICS else "highest"
        print(
            f"Best {emission_mode_name} mean {args.selection_metric} "
            f"({direction}): {best_value:.3f} at {best_time:.3f}s"
        )


if __name__ == "__main__":
    main()

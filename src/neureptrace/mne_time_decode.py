from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
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
from neureptrace.decoding.source_alignment import (
    ALIGNMENT_DIAGNOSTIC_COLUMNS,
    SOURCE_ALIGNMENT_ANCHOR_MODES,
    SOURCE_ALIGNMENT_METHODS,
    SOURCE_ALIGNMENT_TARGET_PROJECTIONS,
    SourceAlignmentResult,
    align_train_test_features,
    source_alignment_config,
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
SOURCE_ALIGNMENT_RUN_METHODS = (*SOURCE_ALIGNMENT_METHODS, "off", "raw")
SOURCE_ALIGNMENT_RUN_ANCHOR_MODES = (
    *SOURCE_ALIGNMENT_ANCHOR_MODES,
    *(mode.replace("_", "-") for mode in SOURCE_ALIGNMENT_ANCHOR_MODES),
)
SOURCE_ALIGNMENT_RUN_TARGET_PROJECTIONS = (
    *SOURCE_ALIGNMENT_TARGET_PROJECTIONS,
    "oracle",
    "oracle_target",
    "target_calibrated",
    "oracle_target_calibrated",
)
CLASS_PRIOR_CORRECTION_CHOICES = ("none", "train_uniform")
CLASS_PRIOR_CORRECTION_RUN_CHOICES = (*CLASS_PRIOR_CORRECTION_CHOICES, "train-uniform")
SOURCE_CALIBRATION_CHOICES = (
    "none",
    "temperature",
    "class_bias",
    "temperature_plus_class_bias",
    "confusion_correction_l2",
)
SOURCE_CALIBRATION_RUN_CHOICES = (
    *SOURCE_CALIBRATION_CHOICES,
    "class-bias",
    "temperature-plus-class-bias",
    "confusion-correction-l2",
)
SOURCE_CALIBRATION_TEMPERATURE_GRID = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
SOURCE_CALIBRATION_BIAS_SCALE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SOURCE_CALIBRATION_L2_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
DEFAULT_BASELINE_WINDOW = (-0.35, -0.05)
BASELINE_WHITENING_SHRINKAGE = 0.1
BASELINE_WHITENING_EIGENVALUE_FLOOR = 1e-6
MNE_SLIDING_MAX_FEATURE_BYTES = 512 * 1024 * 1024
TimeWindow = tuple[int, int, float]
TemporalTrainWindow = tuple[float, float]
DecodeWindow = tuple[float, float]
TEMPORAL_TRAIN_MODE_CHOICES = ("window_ensemble", "pooled")
TEMPORAL_TRAIN_MODE_RUN_CHOICES = (*TEMPORAL_TRAIN_MODE_CHOICES, "window-ensemble")
SOURCE_TIME_SELECTION_CHOICES = (
    "none",
    "source_oof_best_time",
    "source_oof_time_weighted_logits",
    "source_oof_classwise_time_weighted_logits",
    "source_oof_logit_stacker",
)
SOURCE_TIME_SELECTION_RUN_CHOICES = (
    *SOURCE_TIME_SELECTION_CHOICES,
    "source-oof-best-time",
    "source-oof-time-weighted-logits",
    "source-oof-classwise-time-weighted-logits",
    "source-oof-logit-stacker",
)
DEFAULT_SOURCE_TIME_SELECTION_TIMES = (0.088, 0.136, 0.184, 0.232, 0.280)
SOURCE_TIME_SELECTION_OUTPUT_DECODER_SUFFIX = {
    "source_oof_best_time": "source_oof_best_time",
    "source_oof_time_weighted_logits": "source_oof_time_weighted_logits",
    "source_oof_classwise_time_weighted_logits": "source_oof_classwise_time_weighted_logits",
    "source_oof_logit_stacker": "source_oof_logit_stacker",
}
SOURCE_TIME_SELECTION_WEIGHT_GRID_STEP = 0.1
SOURCE_TIME_SELECTION_CLASSWISE_L2 = 0.5
SOURCE_TIME_SELECTION_CLASSWISE_MAX_ITER = 3
SOURCE_TIME_STACKER_TYPE = "shared_time_weights_plus_class_bias"
SOURCE_TIME_STACKER_REGULARIZATION = "strong"
SOURCE_TIME_STACKER_TIME_L2 = 1.0
SOURCE_TIME_STACKER_BIAS_L2 = 0.25
SOURCE_TIME_STACKER_BIAS_SCALE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
PROBABILITY_EPSILON = 1e-12


@dataclass(frozen=True)
class SourceProbabilityCalibrator:
    """Source-only probability transform learned from inner validation folds."""

    mode: str
    temperature: float = 1.0
    bias: tuple[float, ...] = ()
    matrix: tuple[tuple[float, ...], ...] = ()
    score: float = float("nan")
    parameter: str = ""


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


def normalize_source_time_selection(mode: str | None) -> str:
    normalized = "none" if mode is None else str(mode).strip().lower().replace("-", "_")
    if normalized not in SOURCE_TIME_SELECTION_CHOICES:
        raise ValueError(
            f"Unknown source_time_selection '{mode}'. Available modes: {', '.join(SOURCE_TIME_SELECTION_CHOICES)}."
        )
    return normalized


def _parse_float_sequence(value: object | Sequence[object] | None, *, default: Sequence[float]) -> tuple[float, ...]:
    if value is None or value == "":
        return tuple(float(item) for item in default)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        values = [part.strip() for chunk in text.split(",") for part in chunk.split() if part.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, bytes):
        values = list(value)
    else:
        values = [value]
    parsed = tuple(float(item) for item in values)
    if not parsed:
        raise ValueError("Expected at least one time value.")
    return parsed


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


@dataclass(frozen=True, slots=True)
class AlignmentAnchorValues:
    values: np.ndarray | None
    column: str
    source: str


_STIMULUS_ID_ANCHOR_COLUMNS = (
    "stimulus_id",
    "stim_file",
    "stimulus_file",
    "stimulus",
    "stimulus_name",
    "image_id",
    "image",
    "word",
)
_EVENT_CODE_ANCHOR_COLUMNS = (
    "event_code",
    "trigger",
    "event_id",
    "event",
    "value",
    "trial_type",
)
_RUN_ANCHOR_COLUMNS = ("run", "run_id", "run_label")


def _alignment_anchor_values(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    label_column: str,
    anchor_mode: str,
    anchor_column: str | None = None,
) -> AlignmentAnchorValues:
    """Resolve optional metadata-derived alignment anchors for one Epochs table."""

    mode = str(anchor_mode).strip().lower().replace("-", "_")
    requested_column = "" if anchor_column is None else str(anchor_column).strip()
    if mode in {"class_mean", "class_repetition"}:
        return AlignmentAnchorValues(values=None, column=requested_column, source="decoder_labels")
    if mode in {"stimulus_id_mean", "stimulus_id_repetition"}:
        column = _select_alignment_anchor_column(
            metadata,
            requested_column=requested_column,
            candidates=_STIMULUS_ID_ANCHOR_COLUMNS,
            mode=mode,
        )
        return AlignmentAnchorValues(values=_metadata_anchor_vector(metadata[column], name=column), column=column, source="metadata")
    if mode == "event_code_mean":
        column = _select_alignment_anchor_column(
            metadata,
            requested_column=requested_column,
            candidates=_EVENT_CODE_ANCHOR_COLUMNS,
            mode=mode,
        )
        return AlignmentAnchorValues(values=_metadata_anchor_vector(metadata[column], name=column), column=column, source="metadata")
    if mode == "run_event_index_within_stimulus":
        values, column = _run_event_index_within_stimulus_anchors(
            metadata,
            labels,
            label_column=label_column,
            requested_column=requested_column,
        )
        return AlignmentAnchorValues(values=values, column=column, source="derived_metadata")
    raise ValueError(f"Unsupported alignment anchor mode: {anchor_mode!r}.")


def _select_alignment_anchor_column(
    metadata: pd.DataFrame,
    *,
    requested_column: str,
    candidates: Sequence[str],
    mode: str,
) -> str:
    if requested_column:
        if requested_column not in metadata.columns:
            raise ValueError(f"alignment_anchor_column '{requested_column}' not found for {mode}.")
        return requested_column
    for column in candidates:
        if column in metadata.columns:
            return column
    raise ValueError(
        f"{mode} requires an alignment anchor column. Tried: {', '.join(candidates)}. "
        "Set decoding.alignment_anchor_column or --alignment-anchor-column explicitly."
    )


def _metadata_anchor_vector(values: Sequence[object] | pd.Series, *, name: str) -> np.ndarray:
    series = pd.Series(values, copy=False)
    if series.isna().any():
        raise ValueError(f"alignment anchor column '{name}' contains missing values.")
    return series.astype(str).to_numpy(dtype=object)


def _run_event_index_within_stimulus_anchors(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    *,
    label_column: str,
    requested_column: str,
) -> tuple[np.ndarray, str]:
    stimulus_column = _select_alignment_anchor_column(
        metadata,
        requested_column=requested_column,
        candidates=_STIMULUS_ID_ANCHOR_COLUMNS,
        mode="run_event_index_within_stimulus",
    )
    run_column = next((column for column in _RUN_ANCHOR_COLUMNS if column in metadata.columns), "")
    stimulus_values = _metadata_anchor_vector(metadata[stimulus_column], name=stimulus_column)
    if run_column:
        run_values = _metadata_anchor_vector(metadata[run_column], name=run_column)
    else:
        run_values = np.full(len(metadata), "run-unknown", dtype=object)
    label_values = (
        _metadata_anchor_vector(metadata[label_column], name=label_column)
        if label_column in metadata.columns
        else np.asarray(labels, dtype=object).astype(str)
    )
    counts: dict[tuple[str, str, str], int] = {}
    anchors: list[str] = []
    for run, stimulus, label in zip(run_values, stimulus_values, label_values, strict=False):
        key = (str(run), str(stimulus), str(label))
        counts[key] = counts.get(key, 0) + 1
        anchors.append(f"run={key[0]}|stimulus={key[1]}|label={key[2]}|index={counts[key]}")
    column = f"{run_column or 'run'}+{stimulus_column}+event_index"
    return np.asarray(anchors, dtype=object), column


def _test_subject_label(groups: np.ndarray | None, test_idx: np.ndarray, *, fallback: object = "") -> str:
    if groups is None:
        return str(fallback)
    values = pd.Series(groups[test_idx]).dropna().astype(str).drop_duplicates().tolist()
    return "|".join(values)


def _alignment_diagnostic_row(
    result: SourceAlignmentResult,
    *,
    dataset_name: str,
    test_subject: str,
    alignment_window_center: float,
    alignment_window_size: float,
    decode_window_center: float,
    decode_window_size: float,
) -> dict[str, object]:
    row = {column: "" for column in ALIGNMENT_DIAGNOSTIC_COLUMNS}
    row.update(result.diagnostics)
    row.update(
        {
            "dataset": dataset_name,
            "test_subject": test_subject,
            "alignment_window_center": float(alignment_window_center),
            "alignment_window_size": float(alignment_window_size),
            "decode_window_center": float(decode_window_center),
            "decode_window_size": float(decode_window_size),
        }
    )
    return row


def _write_alignment_diagnostics(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows), columns=ALIGNMENT_DIAGNOSTIC_COLUMNS).to_csv(path, index=False)


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


def _nearest_candidate_windows(windows: Sequence[TimeWindow], requested_times: Sequence[float]) -> list[TimeWindow]:
    if not windows:
        raise ValueError("No decode windows are available for source-time selection.")
    centers = np.asarray([window[2] for window in windows], dtype=float)
    selected: list[TimeWindow] = []
    seen: set[TimeWindow] = set()
    for requested_time in requested_times:
        window = windows[int(np.argmin(np.abs(centers - float(requested_time))))]
        if window not in seen:
            selected.append(window)
            seen.add(window)
    if not selected:
        raise ValueError("No source-time-selection candidate windows were selected.")
    return selected


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _combine_probability_logits(probability_cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probability_cube, PROBABILITY_EPSILON, 1.0))
    return _softmax_rows(np.tensordot(logits, weights, axes=([1], [0])))


def _combine_probability_logits_classwise(probability_cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probability_cube, PROBABILITY_EPSILON, 1.0))
    weights = np.asarray(weights, dtype=float)
    expected_shape = (logits.shape[2], logits.shape[1])
    if weights.shape != expected_shape:
        raise ValueError(f"Classwise source-time weights must have shape {expected_shape}, got {weights.shape}.")
    row_sums = weights.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Each class-specific source-time weight row must have positive mass.")
    normalized = weights / row_sums
    return _softmax_rows(np.einsum("ntc,ct->nc", logits, normalized))


def _nonnegative_weight_candidates(n_times: int, step: float = SOURCE_TIME_SELECTION_WEIGHT_GRID_STEP) -> np.ndarray:
    if n_times <= 0:
        raise ValueError("Need at least one source-time-selection candidate.")
    if n_times == 1:
        return np.ones((1, 1), dtype=float)
    levels = np.arange(0.0, 1.0 + step / 2.0, step)
    candidates: list[np.ndarray] = []

    def visit(prefix: list[float], remaining: int) -> None:
        if remaining == 1:
            value = 1.0 - sum(prefix)
            if value >= -step / 2.0:
                candidate = np.asarray([*prefix, max(0.0, value)], dtype=float)
                total = float(candidate.sum())
                if total > 0 and abs(total - 1.0) <= step / 2.0:
                    candidates.append(candidate / total)
            return
        for value in levels:
            if sum(prefix) + value <= 1.0 + step / 2.0:
                visit([*prefix, float(value)], remaining - 1)

    visit([], n_times)
    if not candidates:
        raise ValueError("No nonnegative source-time-selection weight candidates were generated.")
    return np.unique(np.vstack(candidates).round(12), axis=0)


def _nonnegative_weight_candidates_with_uniform(n_times: int) -> np.ndarray:
    candidates = _nonnegative_weight_candidates(n_times)
    uniform = np.full((1, n_times), 1.0 / n_times, dtype=float)
    return np.unique(np.vstack([candidates, uniform]).round(12), axis=0)


def _balanced_cross_entropy_for_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = _normalize_probability_rows(probabilities)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    counts = np.bincount(labels, minlength=probabilities.shape[1]).astype(float)
    if np.any(counts <= 0.0):
        raise ValueError("Classwise source-time weighting requires every class in source validation labels.")
    sample_weights = 1.0 / counts[labels]
    sample_weights = sample_weights / sample_weights.mean()
    losses = -np.log(np.clip(probabilities[np.arange(len(labels)), labels], PROBABILITY_EPSILON, 1.0))
    return float(np.average(losses, weights=sample_weights))


def _classwise_weight_objective(probability_cube: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    probabilities = _combine_probability_logits_classwise(probability_cube, weights)
    uniform = np.full(weights.shape[1], 1.0 / weights.shape[1], dtype=float)
    l2_to_uniform = float(np.mean(np.sum((weights - uniform.reshape(1, -1)) ** 2, axis=1)))
    return _balanced_cross_entropy_for_probabilities(probabilities, labels) + (
        SOURCE_TIME_SELECTION_CLASSWISE_L2 * l2_to_uniform
    )


def _fit_classwise_nonnegative_time_weights(
    probability_cube: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, float]:
    n_times = probability_cube.shape[1]
    n_classes = probability_cube.shape[2]
    weights = np.full((n_classes, n_times), 1.0 / n_times, dtype=float)
    candidates = _nonnegative_weight_candidates_with_uniform(n_times)
    best_objective = _classwise_weight_objective(probability_cube, labels, weights)

    for _iteration in range(SOURCE_TIME_SELECTION_CLASSWISE_MAX_ITER):
        improved = False
        for class_index in range(n_classes):
            best_row = weights[class_index].copy()
            for candidate in candidates:
                trial = weights.copy()
                trial[class_index] = candidate
                objective = _classwise_weight_objective(probability_cube, labels, trial)
                if objective < best_objective - 1e-12:
                    best_objective = objective
                    best_row = candidate.copy()
                    improved = True
            weights[class_index] = best_row
        if not improved:
            break

    probabilities = _combine_probability_logits_classwise(probability_cube, weights)
    score = float(balanced_accuracy_score(labels, probabilities.argmax(axis=1)))
    return weights, score


def _fit_logit_stacker_time_weights_and_bias(
    probability_cube: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    n_times = probability_cube.shape[1]
    uniform = np.full(n_times, 1.0 / n_times, dtype=float)
    best_weights = uniform
    best_bias = np.zeros(probability_cube.shape[2], dtype=float)
    best_objective = np.inf
    best_score = -np.inf

    logits = np.log(np.clip(probability_cube, PROBABILITY_EPSILON, 1.0))
    for weights in _nonnegative_weight_candidates_with_uniform(n_times):
        base_logits = np.tensordot(logits, weights, axes=([1], [0]))
        bias_direction = _marginal_class_bias(_softmax_rows(base_logits))
        for bias_scale in SOURCE_TIME_STACKER_BIAS_SCALE_GRID:
            bias = bias_direction * float(bias_scale)
            probabilities = _softmax_rows(base_logits + bias.reshape(1, -1))
            time_l2 = float(np.sum((weights - uniform) ** 2))
            bias_l2 = float(np.mean(bias**2))
            objective = _balanced_cross_entropy_for_probabilities(probabilities, labels) + (
                SOURCE_TIME_STACKER_TIME_L2 * time_l2
            ) + (SOURCE_TIME_STACKER_BIAS_L2 * bias_l2)
            score = float(balanced_accuracy_score(labels, probabilities.argmax(axis=1)))
            if objective < best_objective - 1e-12 or (
                abs(objective - best_objective) <= 1e-12 and score > best_score
            ):
                best_objective = objective
                best_score = score
                best_weights = weights.copy()
                best_bias = bias.copy()

    return best_weights, best_bias, best_score, float(best_objective)


def _format_source_time_weights(weights: np.ndarray) -> str:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        return "|".join(f"{float(weight):.12g}" for weight in weights)
    if weights.ndim == 2:
        return "/".join("|".join(f"{float(weight):.12g}" for weight in row) for row in weights)
    raise ValueError("Source-time weights must be a vector or class-by-time matrix.")


def _format_source_time_bias(bias: np.ndarray | None) -> str:
    if bias is None:
        return ""
    return "|".join(f"{float(value):.12g}" for value in np.asarray(bias, dtype=float))


def _source_time_active_indices(weights: np.ndarray) -> list[int]:
    weights = np.asarray(weights, dtype=float)
    time_mass = weights if weights.ndim == 1 else weights.sum(axis=0)
    return [index for index, weight in enumerate(time_mass) if weight > 0.0]


def _combine_source_time_probabilities(
    probability_cube: np.ndarray,
    weights: np.ndarray,
    selected_indices: Sequence[int],
    bias: np.ndarray | None = None,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        active_weights = weights[list(selected_indices)]
        logits = np.log(np.clip(probability_cube, PROBABILITY_EPSILON, 1.0))
        combined_logits = np.tensordot(logits, active_weights / active_weights.sum(), axes=([1], [0]))
        if bias is not None:
            combined_logits = combined_logits + np.asarray(bias, dtype=float).reshape(1, -1)
        return _softmax_rows(combined_logits)
    active_weights = weights[:, list(selected_indices)]
    return _combine_probability_logits_classwise(probability_cube, active_weights)


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


def normalize_source_calibration(mode: str | None) -> str:
    """Normalize source-only post-hoc calibration modes."""

    normalized = "none" if mode is None else str(mode).strip().lower().replace("-", "_")
    if normalized not in SOURCE_CALIBRATION_CHOICES:
        raise ValueError(
            f"Unknown source_calibration '{mode}'. Available modes: "
            f"{', '.join(SOURCE_CALIBRATION_CHOICES)}."
        )
    return normalized


def _probabilities_to_logits(probabilities: np.ndarray) -> np.ndarray:
    return np.log(np.clip(_normalize_probability_rows(probabilities), 1e-12, 1.0))


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=float)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _balanced_score_for_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.asarray(probabilities).argmax(axis=1)
    return float(balanced_accuracy_score(np.asarray(labels, dtype=int), predictions))


def _marginal_class_bias(probabilities: np.ndarray) -> np.ndarray:
    marginal = _normalize_probability_rows(probabilities).mean(axis=0)
    target = np.full(marginal.shape, 1.0 / len(marginal), dtype=float)
    bias = np.log(np.clip(target, 1e-12, 1.0)) - np.log(np.clip(marginal, 1e-12, 1.0))
    return bias - float(np.mean(bias))


def _calibrator_with_best_score(candidates: Sequence[SourceProbabilityCalibrator], oof_probabilities: np.ndarray, labels: np.ndarray) -> SourceProbabilityCalibrator:
    best: SourceProbabilityCalibrator | None = None
    best_score = -np.inf
    for candidate in candidates:
        score = _balanced_score_for_probabilities(
            apply_source_probability_calibration(oof_probabilities, candidate),
            labels,
        )
        if score > best_score:
            best = candidate
            best_score = score
    if best is None:
        return SourceProbabilityCalibrator(mode="none")
    return SourceProbabilityCalibrator(
        mode=best.mode,
        temperature=best.temperature,
        bias=best.bias,
        matrix=best.matrix,
        score=best_score,
        parameter=best.parameter,
    )


def fit_source_probability_calibrator(
    probabilities: np.ndarray,
    labels: np.ndarray,
    mode: str,
) -> SourceProbabilityCalibrator:
    """Fit a probability re-ranking transform from source-only validation predictions.

    ``probabilities`` should be out-of-fold predictions from inner splits of the
    outer training subjects. The returned transform is then applied to the
    held-out target subject predictions from the final model.
    """

    mode = normalize_source_calibration(mode)
    probabilities = _normalize_probability_rows(probabilities)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Source calibration probabilities and labels must have the same row count.")
    n_classes = probabilities.shape[1]
    if mode == "none":
        return SourceProbabilityCalibrator(mode="none")

    bias = _marginal_class_bias(probabilities)
    candidates: list[SourceProbabilityCalibrator] = []
    if mode == "temperature":
        candidates = [
            SourceProbabilityCalibrator(mode=mode, temperature=temperature, parameter=f"temperature={temperature:g}")
            for temperature in SOURCE_CALIBRATION_TEMPERATURE_GRID
        ]
    elif mode == "class_bias":
        candidates = [
            SourceProbabilityCalibrator(
                mode=mode,
                bias=tuple((bias * scale).tolist()),
                parameter=f"bias_scale={scale:g}",
            )
            for scale in SOURCE_CALIBRATION_BIAS_SCALE_GRID
        ]
    elif mode == "temperature_plus_class_bias":
        candidates = [
            SourceProbabilityCalibrator(
                mode=mode,
                temperature=temperature,
                bias=tuple((bias * scale).tolist()),
                parameter=f"temperature={temperature:g};bias_scale={scale:g}",
            )
            for temperature in SOURCE_CALIBRATION_TEMPERATURE_GRID
            for scale in SOURCE_CALIBRATION_BIAS_SCALE_GRID
        ]
    elif mode == "confusion_correction_l2":
        targets = np.eye(n_classes, dtype=float)[labels]
        design = probabilities
        identity = np.eye(n_classes, dtype=float)
        for penalty in SOURCE_CALIBRATION_L2_GRID:
            matrix = np.linalg.solve(
                design.T @ design + float(penalty) * identity,
                design.T @ targets + float(penalty) * identity,
            )
            candidates.append(
                SourceProbabilityCalibrator(
                    mode=mode,
                    matrix=tuple(tuple(float(value) for value in row) for row in matrix),
                    parameter=f"l2={penalty:g}",
                )
            )
    else:
        raise ValueError(f"Unsupported source calibration mode: {mode}")
    return _calibrator_with_best_score(candidates, probabilities, labels)


def apply_source_probability_calibration(
    probabilities: np.ndarray,
    calibrator: SourceProbabilityCalibrator,
) -> np.ndarray:
    probabilities = _normalize_probability_rows(probabilities)
    mode = normalize_source_calibration(calibrator.mode)
    if mode == "none":
        return probabilities
    if mode in {"temperature", "class_bias", "temperature_plus_class_bias"}:
        logits = _probabilities_to_logits(probabilities) / max(float(calibrator.temperature), 1e-12)
        if calibrator.bias:
            logits = logits + np.asarray(calibrator.bias, dtype=float).reshape(1, -1)
        return _softmax(logits)
    if mode == "confusion_correction_l2":
        if not calibrator.matrix:
            return probabilities
        corrected = probabilities @ np.asarray(calibrator.matrix, dtype=float)
        corrected = np.clip(corrected, 1e-12, None)
        return corrected / corrected.sum(axis=1, keepdims=True)
    raise ValueError(f"Unsupported source calibration mode: {mode}")


def source_calibration_metadata(calibrator: SourceProbabilityCalibrator) -> dict[str, object]:
    """Return compact provenance for source-only calibration outputs."""

    return {
        "source_calibration": normalize_source_calibration(calibrator.mode),
        "source_calibration_parameter": calibrator.parameter,
        "source_calibration_inner_score": "" if not np.isfinite(calibrator.score) else float(calibrator.score),
    }


def fit_inner_source_probability_calibrator(
    *,
    features: np.ndarray,
    train_idx: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray | None,
    decoder_name: str,
    emission_mode: str,
    max_iter: int,
    feature_preprocessor: str,
    pca_components: int | float | None,
    tune_hyperparameters: bool,
    tuning_cv_splits: int,
    tuning_scoring: str,
    tuning_c_grid: Sequence[float],
    classes: np.ndarray,
    source_calibration: str,
) -> SourceProbabilityCalibrator:
    """Fit source calibration with inner CV over the current outer train fold."""

    source_calibration = normalize_source_calibration(source_calibration)
    if source_calibration == "none":
        return SourceProbabilityCalibrator(mode="none")
    train_idx = np.asarray(train_idx, dtype=int)
    train_labels = np.asarray(train_labels, dtype=int)
    train_groups = None if train_groups is None else np.asarray(train_groups)
    try:
        inner_splits = make_tuning_cross_validator(train_labels, train_groups, tuning_cv_splits)
    except ValueError:
        return SourceProbabilityCalibrator(mode="none")

    oof_probabilities = np.zeros((len(train_idx), len(classes)), dtype=float)
    oof_labels = np.asarray(train_labels, dtype=int)
    filled = np.zeros(len(train_idx), dtype=bool)
    for inner_train_local, inner_valid_local in inner_splits:
        inner_tuning_cv = (
            make_tuning_cross_validator(
                train_labels[inner_train_local],
                None if train_groups is None else train_groups[inner_train_local],
                tuning_cv_splits,
            )
            if tune_hyperparameters
            else 3
        )
        model = make_decoder(
            decoder_name,
            max_iter=max_iter,
            emission_mode=emission_mode,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
            tune_hyperparameters=tune_hyperparameters,
            tuning_cv=inner_tuning_cv,
            tuning_scoring=tuning_scoring,
            tuning_c_grid=tuning_c_grid,
        )
        model.fit(features[train_idx[inner_train_local]], train_labels[inner_train_local])
        oof_probabilities[inner_valid_local] = _align_probability_columns(
            predict_emission_probabilities(
                model,
                features[train_idx[inner_valid_local]],
                emission_mode=emission_mode,
            ),
            model=model,
            classes=classes,
        )
        filled[inner_valid_local] = True
    if not np.all(filled):
        return SourceProbabilityCalibrator(mode="none")
    return fit_source_probability_calibrator(oof_probabilities, oof_labels, source_calibration)


def _fit_source_time_selector(
    *,
    feature_cache: dict[TimeWindow, np.ndarray],
    candidate_windows: Sequence[TimeWindow],
    train_idx: np.ndarray,
    train_labels: np.ndarray,
    train_groups: np.ndarray | None,
    decoder_name: str,
    emission_mode: str,
    max_iter: int,
    feature_preprocessor: str,
    pca_components: int | float | str | None,
    tune_hyperparameters: bool,
    tuning_cv_splits: int,
    tuning_scoring: str,
    tuning_c_grid: Sequence[float] | str | None,
    classes: np.ndarray,
    class_prior_correction: str,
    mode: str,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    inner_splits = list(make_tuning_cross_validator(train_labels, train_groups, tuning_cv_splits))
    if not inner_splits:
        raise ValueError("Source-time selection needs at least one inner validation split.")

    validation_probabilities: list[list[np.ndarray]] = [[] for _ in candidate_windows]
    validation_labels: list[np.ndarray] = []
    for inner_train_local, inner_valid_local in inner_splits:
        validation_labels.append(train_labels[inner_valid_local])
        for window_index, window in enumerate(candidate_windows):
            features = feature_cache[window]
            inner_tuning_cv = (
                make_tuning_cross_validator(
                    train_labels[inner_train_local],
                    None if train_groups is None else train_groups[inner_train_local],
                    tuning_cv_splits,
                )
                if tune_hyperparameters
                else 3
            )
            model = make_decoder(
                decoder_name,
                max_iter=max_iter,
                emission_mode=emission_mode,
                feature_preprocessor=feature_preprocessor,
                pca_components=pca_components,
                tune_hyperparameters=tune_hyperparameters,
                tuning_cv=inner_tuning_cv,
                tuning_scoring=tuning_scoring,
                tuning_c_grid=tuning_c_grid,
            )
            model.fit(features[train_idx[inner_train_local]], train_labels[inner_train_local])
            probabilities = _align_probability_columns(
                predict_emission_probabilities(
                    model,
                    features[train_idx[inner_valid_local]],
                    emission_mode=emission_mode,
                ),
                model=model,
                classes=classes,
            )
            probabilities = _apply_class_prior_correction(
                probabilities,
                train_labels[inner_train_local],
                classes,
                class_prior_correction,
            )
            validation_probabilities[window_index].append(probabilities)

    labels = np.concatenate(validation_labels)
    probability_cube = np.stack([np.vstack(parts) for parts in validation_probabilities], axis=1)
    time_scores = [
        float(balanced_accuracy_score(labels, probability_cube[:, window_index, :].argmax(axis=1)))
        for window_index in range(len(candidate_windows))
    ]

    if mode == "source_oof_best_time":
        best_index = int(np.argmax(time_scores))
        weights = np.zeros(len(candidate_windows), dtype=float)
        weights[best_index] = 1.0
        source_score = time_scores[best_index]
        bias = None
        objective: float | None = None
    elif mode == "source_oof_time_weighted_logits":
        best_weights = None
        source_score = -np.inf
        for weights_candidate in _nonnegative_weight_candidates_with_uniform(len(candidate_windows)):
            probabilities = _combine_probability_logits(probability_cube, weights_candidate)
            score = float(balanced_accuracy_score(labels, probabilities.argmax(axis=1)))
            if score > source_score:
                best_weights = weights_candidate
                source_score = score
        if best_weights is None:
            raise ValueError("No source-time-selection weights were selected.")
        weights = best_weights
        bias = None
        objective = None
    elif mode == "source_oof_classwise_time_weighted_logits":
        weights, source_score = _fit_classwise_nonnegative_time_weights(probability_cube, labels)
        bias = None
        objective = None
    elif mode == "source_oof_logit_stacker":
        weights, bias, source_score, objective = _fit_logit_stacker_time_weights_and_bias(probability_cube, labels)
    else:
        raise ValueError(f"Unknown source-time-selection mode '{mode}'.")

    time_mass = weights if weights.ndim == 1 else weights.sum(axis=0)
    metadata = {
        "source_time_selection": mode,
        "source_time_selection_candidate_times": "|".join(f"{window[2]:.12g}" for window in candidate_windows),
        "source_time_selection_time_scores": "|".join(f"{score:.12g}" for score in time_scores),
        "source_time_selection_weights": _format_source_time_weights(weights),
        "source_time_selection_selected_time": float(candidate_windows[int(np.argmax(time_mass))][2]),
        "source_time_selection_inner_score": float(source_score),
        "source_time_selection_weight_type": "classwise" if weights.ndim == 2 else ("stacker" if bias is not None else "global"),
    }
    if bias is not None:
        metadata.update(
            {
                "source_time_selection_stacker_type": SOURCE_TIME_STACKER_TYPE,
                "source_time_selection_stacker_regularization": SOURCE_TIME_STACKER_REGULARIZATION,
                "source_time_selection_class_bias": _format_source_time_bias(bias),
                "source_time_selection_inner_objective": float(objective) if objective is not None else "",
            }
        )
    return weights, bias, metadata


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
    source_calibration: str = "none",
    source_time_selection: str = "none",
    source_time_selection_times: Sequence[float] | None = None,
    alignment_metadata: Mapping[str, object] | None = None,
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
    if source_calibration != "none":
        payload["source_calibration"] = source_calibration
    if source_time_selection != "none":
        payload["source_time_selection"] = source_time_selection
        payload["source_time_selection_times"] = tuple(source_time_selection_times or ())
    if alignment_metadata:
        payload["source_alignment"] = dict(alignment_metadata)
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
    source_calibration_metadata: dict[str, object] | None = None,
    alignment_metadata: Mapping[str, object] | None = None,
    label_shuffle_control: bool = False,
    label_shuffle_seed: int = 13,
    outer_test_groups: Sequence[str] = (),
) -> None:
    tuning_metadata = {} if tuning_metadata is None else tuning_metadata
    source_calibration_metadata = (
        source_calibration_metadata
        if source_calibration_metadata is not None
        else {"source_calibration": "none", "source_calibration_parameter": "", "source_calibration_inner_score": ""}
    )
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
        **source_calibration_metadata,
        "outer_test_groups": "|".join(outer_test_groups),
    }
    if alignment_metadata:
        common.update(dict(alignment_metadata))
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
    dataset_name: str | None = None,
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
    source_calibration: str = "none",
    source_time_selection: str = "none",
    source_time_selection_times: Sequence[float] | str | None = None,
    source_time_selection_output_time: float = 0.184,
    alignment_method: str = "none",
    alignment_anchor_mode: str = "class_mean",
    alignment_anchor_column: str | None = None,
    alignment_repetition_cap: int | str | None = 16,
    alignment_components: int | float | str | None = 64,
    alignment_times: Sequence[float] | str | None = None,
    alignment_target_projection: str = "group_projection",
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
    source_calibration_name = normalize_source_calibration(source_calibration)
    source_time_selection_name = normalize_source_time_selection(source_time_selection)
    source_time_selection_time_values = _parse_float_sequence(
        source_time_selection_times,
        default=DEFAULT_SOURCE_TIME_SELECTION_TIMES,
    )
    source_time_selection_output_time = float(source_time_selection_output_time)
    alignment_config = source_alignment_config(
        method=alignment_method,
        anchor_mode=alignment_anchor_mode,
        anchor_column=alignment_anchor_column,
        repetition_cap=alignment_repetition_cap,
        components=alignment_components,
        times=alignment_times,
        target_projection=alignment_target_projection,
    )
    requested_time_decode_backend = normalize_time_decode_backend(time_decode_backend)
    label_shuffle_control = bool(label_shuffle_control)
    label_shuffle_seed = int(label_shuffle_seed)
    dataset_name_value = "" if dataset_name is None else str(dataset_name)
    outer_test_groups_value = _normalize_outer_test_groups(outer_test_groups)
    if requested_time_decode_backend == "mne" and normalized_temporal_train_window is not None:
        raise ValueError("The MNE time-decode backend currently supports same-time decoding only.")
    if alignment_config.enabled:
        if requested_time_decode_backend == "mne":
            raise ValueError("source alignment requires the sklearn time-decode backend.")
        if group_column is None:
            raise ValueError("source alignment requires group_column so source subjects can be identified.")
        if normalized_temporal_train_window is not None:
            raise ValueError("source alignment currently supports same-time decoding only.")
        if source_time_selection_name != "none":
            raise ValueError("source alignment should be combined with posthoc response-window aggregation, not source_time_selection.")
        if source_calibration_name != "none":
            raise ValueError("source alignment currently requires source_calibration='none'.")
    if source_time_selection_name != "none" and normalized_temporal_train_window is not None:
        raise ValueError("source_time_selection currently supports same-time decoding only.")
    time_decode_backend = (
        "sklearn"
        if requested_time_decode_backend == "auto"
        and (normalized_temporal_train_window is not None or source_time_selection_name != "none" or alignment_config.enabled)
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
    alignment_anchor_info = (
        _alignment_anchor_values(
            metadata,
            labels,
            label_column=label_column,
            anchor_mode=alignment_config.anchor_mode,
            anchor_column=alignment_config.anchor_column,
        )
        if alignment_config.enabled
        else AlignmentAnchorValues(values=None, column="", source="")
    )
    if alignment_config.enabled:
        alignment_config = replace(alignment_config, anchor_column=alignment_anchor_info.column)
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
            "source_calibration": source_calibration_name,
            "source_time_selection": source_time_selection_name,
            "source_time_selection_times": source_time_selection_time_values,
            "source_alignment": alignment_config.static_metadata() if alignment_config.enabled else {},
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
        source_calibration=source_calibration_name,
        source_time_selection=source_time_selection_name,
        source_time_selection_times=source_time_selection_time_values,
        alignment_metadata=alignment_config.static_metadata() if alignment_config.enabled else None,
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
    alignment_diagnostic_rows: list[dict[str, object]] = []
    all_windows = time_windows(epochs.times, window_ms=window_ms, step_ms=step_ms)
    windows = _select_decode_windows(all_windows, normalized_decode_window)
    selected_train_windows = _select_temporal_train_windows(all_windows, normalized_temporal_train_window)
    splits = _filter_splits_for_outer_test_groups(
        list(enumerate(make_cross_validator(labels, groups, n_splits))),
        groups,
        outer_test_groups_value,
    )

    if source_time_selection_name != "none":
        if time_decode_backend == "mne":
            raise ValueError("source_time_selection currently requires the sklearn time-decode backend.")
        if source_calibration_name != "none":
            raise ValueError("source_time_selection should be run with source_calibration='none'.")
        candidate_windows = _nearest_candidate_windows(windows, source_time_selection_time_values)
        feature_cache = {time_window: _features_for_window(data, time_window) for time_window in candidate_windows}
        selection_decoder_name = f"{decoder_name}_{SOURCE_TIME_SELECTION_OUTPUT_DECODER_SUFFIX[source_time_selection_name]}"
        candidate_centers = [window[2] for window in candidate_windows]
        for fold, (train_idx, test_idx) in splits:
            test_labels = labels[test_idx]
            train_labels = _fold_training_labels(
                labels,
                train_idx,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
                context=(split_id, fold, source_time_selection_name, *candidate_centers),
            )
            train_groups = None if groups is None else groups[train_idx]
            for current_emission_mode in emission_modes:
                weights, bias, selection_metadata = _fit_source_time_selector(
                    feature_cache=feature_cache,
                    candidate_windows=candidate_windows,
                    train_idx=train_idx,
                    train_labels=train_labels,
                    train_groups=train_groups,
                    decoder_name=decoder_name,
                    emission_mode=current_emission_mode,
                    max_iter=max_iter,
                    feature_preprocessor=feature_preprocessor_name,
                    pca_components=pca_components_value,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                    classes=classes,
                    class_prior_correction=class_prior_correction_name,
                    mode=source_time_selection_name,
                )
                test_probability_parts: list[np.ndarray] = []
                final_tuning_metadata: dict[str, object] = {}
                selected_indices = _source_time_active_indices(weights)
                for window_index in selected_indices:
                    window = candidate_windows[window_index]
                    features = feature_cache[window]
                    tuning_cv = (
                        make_tuning_cross_validator(train_labels, train_groups, tuning_cv_splits)
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
                    test_probability_parts.append(probabilities)
                    if not final_tuning_metadata:
                        final_tuning_metadata = _tuning_metadata(
                            model,
                            tune_hyperparameters=tune_hyperparameters,
                            tuning_cv_splits=tuning_cv_splits,
                            tuning_scoring=tuning_scoring,
                            tuning_c_grid=tuning_c_grid_values,
                        )
                if not test_probability_parts:
                    raise ValueError("Source-time selection produced no final test probabilities.")
                probability_cube = np.stack(test_probability_parts, axis=1)
                probabilities = _combine_source_time_probabilities(probability_cube, weights, selected_indices, bias=bias)
                tuning_metadata = {**final_tuning_metadata, **selection_metadata}
                selected_windows = [candidate_windows[index] for index in selected_indices]
                synthetic_window = (
                    min(window[0] for window in selected_windows),
                    max(window[1] for window in selected_windows),
                    source_time_selection_output_time,
                )
                current_model_hash = _model_hash(
                    decoder_name=selection_decoder_name,
                    emission_mode=current_emission_mode,
                    max_iter=max_iter,
                    feature_preprocessor=feature_preprocessor_name,
                    pca_components=pca_components_value,
                    normalization=normalization_name,
                    baseline_window=baseline_window_value,
                    temporal_mode=source_time_selection_name,
                    temporal_train_window=None,
                    train_window_centers=candidate_centers,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                    tuning_metadata=tuning_metadata,
                    backend=time_decode_backend,
                    class_prior_correction=class_prior_correction_name,
                    source_calibration=source_calibration_name,
                    source_time_selection=source_time_selection_name,
                    source_time_selection_times=source_time_selection_time_values,
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
                    decoder_name=selection_decoder_name,
                    emission_mode=current_emission_mode,
                    feature_preprocessor_name=feature_preprocessor_name,
                    pca_components_value=pca_components_value,
                    normalization_name=normalization_name,
                    baseline_window=baseline_window_value,
                    time_window=synthetic_window,
                    epochs=epochs,
                    split_id=split_id,
                    preprocessing_hash=preprocessing_hash,
                    model_hash=current_model_hash,
                    temporal_mode=source_time_selection_name,
                    temporal_train_window=None,
                    train_time=source_time_selection_output_time,
                    train_window_start=float(min(epochs.times[window[0]] for window in selected_windows)),
                    train_window_stop=float(max(epochs.times[window[1] - 1] for window in selected_windows)),
                    n_train_windows=len(selected_windows),
                    calibration_out_path=calibration_out_path,
                    calibration_bins=calibration_bins,
                    observation_out_path=observation_out_path,
                    subject=subject,
                    tuning_metadata=tuning_metadata,
                    backend=time_decode_backend,
                    class_prior_correction=class_prior_correction_name,
                    source_calibration_metadata=source_calibration_metadata(SourceProbabilityCalibrator(mode="none")),
                    label_shuffle_control=label_shuffle_control,
                    label_shuffle_seed=label_shuffle_seed,
                    outer_test_groups=outer_test_groups_value,
                )
    elif selected_train_windows is None and time_decode_backend == "mne":
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
                    calibrator = SourceProbabilityCalibrator(mode="none")
                    source_metadata = source_calibration_metadata(calibrator)
                    if source_calibration_name != "none":
                        raise ValueError("source_calibration currently requires the sklearn time-decode backend.")
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
                        source_calibration=source_calibration_name,
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
                        source_calibration_metadata=source_metadata,
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
                train_feature_matrix = features[train_idx]
                test_feature_matrix = features[test_idx]
                alignment_metadata = None
                if alignment_config.enabled:
                    alignment_result = align_train_test_features(
                        train_features=train_feature_matrix,
                        train_labels=train_labels,
                        train_subject_ids=groups[train_idx],
                        test_features=test_feature_matrix,
                        target_labels=(
                            test_labels
                            if alignment_config.oracle_target_calibrated and alignment_anchor_info.values is None
                            else None
                        ),
                        train_anchor_values=(
                            None if alignment_anchor_info.values is None else alignment_anchor_info.values[train_idx]
                        ),
                        target_anchor_values=(
                            None
                            if alignment_anchor_info.values is None or not alignment_config.oracle_target_calibrated
                            else alignment_anchor_info.values[test_idx]
                        ),
                        config=alignment_config,
                    )
                    train_feature_matrix = alignment_result.train_features
                    test_feature_matrix = alignment_result.test_features
                    alignment_metadata = alignment_result.metadata
                    alignment_diagnostic_rows.append(
                        _alignment_diagnostic_row(
                            alignment_result,
                            dataset_name=dataset_name_value,
                            test_subject=_test_subject_label(groups, test_idx, fallback=fold),
                            alignment_window_center=center,
                            alignment_window_size=float(window_ms) / 1000.0,
                            decode_window_center=center,
                            decode_window_size=float(window_ms) / 1000.0,
                        )
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
                    model.fit(train_feature_matrix, train_labels)

                    probabilities = _align_probability_columns(
                        predict_emission_probabilities(
                            model,
                            test_feature_matrix,
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
                    calibrator = fit_inner_source_probability_calibrator(
                        features=features,
                        train_idx=train_idx,
                        train_labels=train_labels,
                        train_groups=None if groups is None else groups[train_idx],
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        max_iter=max_iter,
                        feature_preprocessor=feature_preprocessor_name,
                        pca_components=pca_components_value,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                        classes=classes,
                        source_calibration=source_calibration_name,
                    )
                    probabilities = apply_source_probability_calibration(probabilities, calibrator)
                    source_metadata = source_calibration_metadata(calibrator)
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
                        source_calibration=source_calibration_name,
                        alignment_metadata=alignment_metadata,
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
                        source_calibration_metadata=source_metadata,
                        alignment_metadata=alignment_metadata,
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
                    if source_calibration_name != "none":
                        raise ValueError("source_calibration currently supports same-time decoding only.")
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
                        source_calibration_metadata=source_calibration_metadata(SourceProbabilityCalibrator(mode="none")),
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
                    if source_calibration_name != "none":
                        raise ValueError("source_calibration currently supports same-time decoding only.")
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
                        source_calibration_metadata=source_calibration_metadata(SourceProbabilityCalibrator(mode="none")),
                        label_shuffle_control=label_shuffle_control,
                        label_shuffle_seed=label_shuffle_seed,
                        outer_test_groups=outer_test_groups_value,
                    )

    results = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    if alignment_config.enabled:
        _write_alignment_diagnostics(out_path.parent / "alignment_diagnostics.csv", alignment_diagnostic_rows)
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
    parser.add_argument("--dataset-name", help="Dataset identifier written to alignment_diagnostics.csv.")
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
        "--source-calibration",
        choices=SOURCE_CALIBRATION_RUN_CHOICES,
        default="none",
        help="Nested source-only probability re-ranking learned from inner folds of each outer training set.",
    )
    parser.add_argument(
        "--source-time-selection",
        choices=SOURCE_TIME_SELECTION_RUN_CHOICES,
        default="none",
        help="Nested source-only time selection learned from inner folds of each outer training set.",
    )
    parser.add_argument(
        "--source-time-selection-times",
        default=",".join(str(time) for time in DEFAULT_SOURCE_TIME_SELECTION_TIMES),
        help="Comma-separated candidate time centers for --source-time-selection.",
    )
    parser.add_argument(
        "--source-time-selection-output-time",
        type=float,
        default=0.184,
        help="Reported output time for source-time-selected predictions.",
    )
    parser.add_argument(
        "--alignment-method",
        choices=SOURCE_ALIGNMENT_RUN_METHODS,
        default="none",
        help="Strict source-only feature alignment method fitted inside each outer fold.",
    )
    parser.add_argument(
        "--alignment-anchor-mode",
        choices=SOURCE_ALIGNMENT_RUN_ANCHOR_MODES,
        default="class_mean",
        help="Source anchor rows used to fit strict alignment.",
    )
    parser.add_argument(
        "--alignment-anchor-column",
        help=(
            "Metadata column for stimulus/event alignment anchors. If omitted, "
            "stimulus_id modes auto-select columns such as stim_file, and event_code_mean auto-selects trigger."
        ),
    )
    parser.add_argument(
        "--alignment-repetition-cap",
        default="16",
        help="Maximum repetitions per class for class_repetition alignment anchors; use all/none for all common repetitions.",
    )
    parser.add_argument(
        "--alignment-components",
        default="64",
        help="Common-space component count for strict source alignment.",
    )
    parser.add_argument(
        "--alignment-times",
        default="0.088,0.136,0.184,0.232,0.280",
        help="Response-window time centers recorded for strict alignment screens.",
    )
    parser.add_argument(
        "--alignment-target-projection",
        choices=SOURCE_ALIGNMENT_RUN_TARGET_PROJECTIONS,
        default="group_projection",
        help=(
            "Held-out target projection mode. group_projection is the benchmark-valid strict source-only mode; "
            "oracle_target_calibrated_alignment uses held-out labels and is a debug upper bound only."
        ),
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
        dataset_name=args.dataset_name,
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
        source_calibration=args.source_calibration,
        source_time_selection=args.source_time_selection,
        source_time_selection_times=args.source_time_selection_times,
        source_time_selection_output_time=args.source_time_selection_output_time,
        alignment_method=args.alignment_method,
        alignment_anchor_mode=args.alignment_anchor_mode,
        alignment_anchor_column=args.alignment_anchor_column,
        alignment_repetition_cap=args.alignment_repetition_cap,
        alignment_components=args.alignment_components,
        alignment_times=args.alignment_times,
        alignment_target_projection=args.alignment_target_projection,
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

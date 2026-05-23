"""Strict source-only LOSO decoding workflow for the BUSH-MEG main task.

The workflow is cue-free by default: it loads only the configured main-task
FieldTrip MATLAB files, performs leave-one-subject-out evaluation, and selects
window/model hyperparameters by an inner LOSO loop over source subjects only.
Optionally, cue files can be used for fold-local source-subject weighting; cue
epochs are never added as classifier training or test trials.

The implementation differs from the generic time-resolved decoder in two ways
that matter for BUSH-MEG:

* temporal features are compact per-channel bin means rather than very large
  sensor-by-sample windows;
* optional source-class template-similarity features turn each trial into a
  low-dimensional vector of leave-source-subject-out class-prototype scores;
* a candidate may average probabilities from several nearby post-stimulus
  windows, giving a strict source-only temporal bagging baseline.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from neureptrace import mne_time_decode as _base
from neureptrace.dataset_config import (
    _fieldtrip_file_specs,
    _validation_section,
    apply_overrides,
    expand_path,
    load_config,
    validate_dataset_config,
)
from neureptrace.decoding import (
    make_decoder,
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    normalize_pca_components,
    parse_c_grid,
    predict_emission_probabilities,
)
from neureptrace.io.fieldtrip_mat import load_fieldtrip_mat_epochs
from neureptrace.bushmeg_cue_source_weights import (
    CueSourceWeights,
    resolve_cue_source_weights,
    write_cue_source_weight_csv,
)

DEFAULT_SELECTION_METRIC = "balanced_accuracy"
DEFAULT_RANDOM_SEED = 13
SUPPORTED_SELECTION_METRICS = {"balanced_accuracy", "accuracy", "log_loss"}
SOURCE_FEATURE_FAMILIES = (
    "bin_means",
    "template_similarity",
    "template_similarity_plus_bin_means",
)
MINIMIZE_SELECTION_METRICS = {"log_loss"}
SAMPLE_WEIGHTING_MODES = {"none", "class_balanced", "subject_balanced", "subject_class_balanced"}
CLASS_BIAS_MODES = {"none", "log_prior", "balanced_accuracy"}
DEFAULT_CLASS_BIAS_DELTAS = (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0)
DEFAULT_CLASS_BIAS_ROUNDS = 2
FEATURE_KIND_CHOICES = (
    "evoked",
    "logvar",
    "covariance",
    "evoked_logvar",
    "evoked_covariance",
    "prototype",
    "evoked_prototype",
    "logvar_prototype",
    "evoked_logvar_prototype",
    "mnn_evoked",
    "mnn_logvar",
    "mnn_covariance",
    "mnn_evoked_logvar",
    "mnn_evoked_covariance",
    "mnn_prototype",
    "mnn_evoked_prototype",
    "mnn_logvar_prototype",
    "mnn_evoked_logvar_prototype",
    "xdawn",
    "xdawn_prototype",
)
DEFAULT_XDAWN_COMPONENTS = 8
DEFAULT_COVARIANCE_MAX_CHANNELS = 64
DEFAULT_MNN_BASELINE_WINDOW = (-0.35, -0.05)
PROTOTYPE_FEATURE_KINDS = frozenset(
    {
        "prototype",
        "evoked_prototype",
        "logvar_prototype",
        "evoked_logvar_prototype",
        "xdawn_prototype",
        "mnn_prototype",
        "mnn_evoked_prototype",
        "mnn_logvar_prototype",
        "mnn_evoked_logvar_prototype",
    }
)
SUPERVISED_FEATURE_KINDS = PROTOTYPE_FEATURE_KINDS | {"xdawn"}
PROTOTYPE_BASE_FEATURE_KINDS = {
    "prototype": "evoked",
    "evoked_prototype": "evoked",
    "logvar_prototype": "logvar",
    "evoked_logvar_prototype": "evoked_logvar",
    "mnn_prototype": "mnn_evoked",
    "mnn_evoked_prototype": "mnn_evoked",
    "mnn_logvar_prototype": "mnn_logvar",
    "mnn_evoked_logvar_prototype": "mnn_evoked_logvar",
}


@dataclass(slots=True)
class SubjectEpochs:
    """Cropped and optionally subject-normalized epochs for one participant."""

    subject: str
    data: np.ndarray
    times: np.ndarray
    metadata: pd.DataFrame
    labels: np.ndarray


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """One temporal window, represented by center time and width in seconds."""

    center: float
    width: float

    @property
    def start(self) -> float:
        return self.center - self.width / 2.0

    @property
    def stop(self) -> float:
        return self.center + self.width / 2.0


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    """One source-only decoder candidate evaluated by inner LOSO."""

    name: str
    decoder: str
    emission_mode: str
    feature_preprocessor: str
    pca_components: int | float | None
    classifier_param: float | None
    temporal_bins: int
    windows: tuple[WindowSpec, ...]
    feature_kind: str = "evoked"
    xdawn_components: int | None = None
    covariance_max_channels: int = DEFAULT_COVARIANCE_MAX_CHANNELS
    feature_family: str = "bin_means"
    sample_weighting: str = "none"
    class_bias: str = "none"

    @property
    def window_centers(self) -> tuple[float, ...]:
        return tuple(window.center for window in self.windows)

    @property
    def window_widths(self) -> tuple[float, ...]:
        return tuple(window.width for window in self.windows)


class FeatureCache:
    """Small per-subject cache for compact temporal-bin features."""

    def __init__(self, subjects: Mapping[str, SubjectEpochs]):
        self._subjects = dict(subjects)
        self._cache: dict[tuple[str, WindowSpec, int, str, int], np.ndarray] = {}
        self._template_cache: dict[tuple[tuple[str, ...], WindowSpec, int, str, int, int], np.ndarray] = {}

    def get(
        self,
        subject: str,
        window: WindowSpec,
        temporal_bins: int,
        *,
        feature_kind: str = "evoked",
        covariance_max_channels: int = DEFAULT_COVARIANCE_MAX_CHANNELS,
    ) -> np.ndarray:
        normalized_feature_kind = normalize_source_feature_kind(feature_kind)
        key = (subject, window, int(temporal_bins), normalized_feature_kind, int(covariance_max_channels))
        if key not in self._cache:
            subject_epochs = self._subjects[subject]
            self._cache[key] = _window_features(
                subject_epochs.data,
                subject_epochs.times,
                window,
                temporal_bins=int(temporal_bins),
                feature_kind=normalized_feature_kind,
                covariance_max_channels=int(covariance_max_channels),
            )
        return self._cache[key]

    def class_templates(
        self,
        reference_subjects: Sequence[str],
        window: WindowSpec,
        temporal_bins: int,
        n_classes: int,
        *,
        feature_kind: str = "evoked",
        covariance_max_channels: int = DEFAULT_COVARIANCE_MAX_CHANNELS,
    ) -> np.ndarray:
        """Return subject-balanced class templates for a source-subject set."""

        reference_key = tuple(sorted(str(subject) for subject in reference_subjects))
        if not reference_key:
            raise ValueError("At least one reference subject is required for template-similarity features.")
        normalized_feature_kind = normalize_source_feature_kind(feature_kind)
        key = (
            reference_key,
            window,
            int(temporal_bins),
            normalized_feature_kind,
            int(covariance_max_channels),
            int(n_classes),
        )
        if key not in self._template_cache:
            self._template_cache[key] = _class_template_features(
                self,
                self._subjects,
                reference_key,
                window,
                int(temporal_bins),
                int(n_classes),
                feature_kind=normalized_feature_kind,
                covariance_max_channels=int(covariance_max_channels),
            )
        return self._template_cache[key]


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {}) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping.")
    return dict(value)


def _list_value(value: Any, default: Sequence[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def normalize_source_feature_family(value: Any) -> str:
    """Normalize source-only BUSH-MEG feature-family names."""

    normalized = "bin_means" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "sensor_bin_means": "bin_means",
        "bin_mean": "bin_means",
        "binmean": "bin_means",
        "templates": "template_similarity",
        "template_corr": "template_similarity",
        "template_correlation": "template_similarity",
        "prototype_similarity": "template_similarity",
        "prototypes": "template_similarity",
        "source_templates": "template_similarity",
        "template_similarity_augmented": "template_similarity_plus_bin_means",
        "template_corr_plus_bin_means": "template_similarity_plus_bin_means",
        "templates_plus_bin_means": "template_similarity_plus_bin_means",
        "hybrid_template_similarity": "template_similarity_plus_bin_means",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SOURCE_FEATURE_FAMILIES:
        raise ValueError(f"Unknown source feature family '{value}'. Available families: {', '.join(SOURCE_FEATURE_FAMILIES)}.")
    return normalized


def _normalize_sample_weighting(value: str | None) -> str:
    normalized = "none" if value is None else str(value).strip().lower().replace("-", "_")
    if normalized in {"", "false", "off", "no", "unweighted"}:
        return "none"
    if normalized in {"class", "classes", "class_weight", "class_weights", "balanced"}:
        return "class_balanced"
    if normalized in {"subject", "subjects", "subject_weight", "subject_weights"}:
        return "subject_balanced"
    if normalized in {"subject_class", "class_subject", "subject_class_weight", "subject_class_weights"}:
        return "subject_class_balanced"
    if normalized not in SAMPLE_WEIGHTING_MODES:
        raise ValueError(f"Unknown sample weighting mode '{value}'. Available modes: {sorted(SAMPLE_WEIGHTING_MODES)}.")
    return normalized


def _normalize_class_bias(value: str | None) -> str:
    normalized = "none" if value is None else str(value).strip().lower().replace("-", "_")
    if normalized in {"", "false", "off", "no", "unbiased"}:
        return "none"
    if normalized in {"prior", "class_prior", "logprior", "inverse_prior", "inverse_class_prior"}:
        return "log_prior"
    if normalized in {"balanced_acc", "train_balanced_accuracy", "source_balanced_accuracy", "source_loso_balanced_accuracy"}:
        return "balanced_accuracy"
    if normalized not in CLASS_BIAS_MODES:
        raise ValueError(f"Unknown class-bias mode '{value}'. Available modes: {sorted(CLASS_BIAS_MODES)}.")
    return normalized


def normalize_source_feature_kind(feature_kind: str) -> str:
    normalized = str(feature_kind).strip().lower().replace("-", "_")
    if normalized not in FEATURE_KIND_CHOICES:
        raise ValueError(
            f"Unknown source_loso feature kind '{feature_kind}'. "
            f"Available values: {', '.join(FEATURE_KIND_CHOICES)}."
        )
    return normalized


def _window_size_seconds(preprocessing: Mapping[str, Any], default: float = 0.100) -> float:
    if "window_size" in preprocessing:
        return float(preprocessing["window_size"])
    if "window_ms" in preprocessing:
        return float(preprocessing["window_ms"]) / 1000.0
    return float(default)


def _float_list_value(value: Any, default: Sequence[float]) -> list[float]:
    """Return a config scalar/list as finite floats."""

    values = [float(item) for item in _list_value(value, default)]
    if not values:
        raise ValueError("Expected at least one floating-point value.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Window-grid values must be finite.")
    return values


def _inclusive_float_range(start: float, stop: float, step: float) -> list[float]:
    """Return an inclusive float grid, robust to binary rounding noise."""

    start = float(start)
    stop = float(stop)
    step = float(step)
    if not np.all(np.isfinite([start, stop, step])):
        raise ValueError("Window range start/stop/step must be finite.")
    if step <= 0.0:
        raise ValueError("Window range step must be positive.")
    if stop < start:
        raise ValueError("Window range stop must be greater than or equal to start.")
    values: list[float] = []
    current = start
    tolerance = abs(step) * 1e-9 + 1e-12
    while current <= stop + tolerance:
        values.append(round(float(current), 12))
        current += step
    return values


def _config_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean.")


def _window_sets_from_config_item(
    item: Mapping[str, Any],
    *,
    default_window_width: float,
    preprocessing: Mapping[str, Any],
) -> list[tuple[str, tuple[WindowSpec, ...]]]:
    """Expand one ``source_loso.candidate_grid.window_sets`` entry.

    Supported forms:

    * legacy explicit centers: ``centers: [0.150, 0.175]``;
    * generated late grids: ``start: 0.300, stop: 0.550, step: 0.050``;
    * compact full-epoch windows: ``full_epoch: true, start: 0.0, stop: 0.65``.

    The full-epoch form still uses compact temporal-bin means, so it does not
    explode into sensor-by-sample flattened features.
    """

    name = str(item.get("name", "windows"))
    if _config_bool(item.get("full_epoch"), default=False):
        start = float(item.get("start", item.get("tmin", preprocessing.get("tmin", 0.0))))
        stop = float(item.get("stop", item.get("tmax", preprocessing.get("tmax", start + default_window_width))))
        if not np.all(np.isfinite([start, stop])) or stop <= start:
            raise ValueError(f"full_epoch window set '{name}' must have finite stop > start.")
        return [(name, (WindowSpec(center=(start + stop) / 2.0, width=stop - start),))]

    raw_widths = item.get("window_sizes", item.get("widths", item.get("window_size", item.get("width", default_window_width))))
    widths = _float_list_value(raw_widths, [default_window_width])
    if any(width <= 0.0 for width in widths):
        raise ValueError(f"window set '{name}' contains a non-positive width.")

    if "centers" in item:
        centers = _float_list_value(item.get("centers"), [])
    elif {"start", "stop", "step"}.issubset(item):
        centers = _inclusive_float_range(float(item["start"]), float(item["stop"]), float(item["step"]))
    else:
        raise ValueError(
            f"window set '{name}' must define either centers, start/stop/step, or full_epoch: true."
        )

    expanded: list[tuple[str, tuple[WindowSpec, ...]]] = []
    for width in widths:
        width_name = name if len(widths) == 1 else f"{name}_w{int(round(width * 1000.0))}ms"
        expanded.append((width_name, tuple(WindowSpec(center=center, width=width) for center in centers)))
    return expanded


def _candidate_window_sets(grid: Mapping[str, Any], preprocessing: Mapping[str, Any]) -> list[tuple[str, tuple[WindowSpec, ...]]]:
    default_window_width = _window_size_seconds(preprocessing)
    raw_window_sets = grid.get("window_sets") or [
        {"name": "single_150ms", "centers": [0.150], "window_size": default_window_width},
        {"name": "single_175ms", "centers": [0.175], "window_size": default_window_width},
        {"name": "single_200ms", "centers": [0.200], "window_size": default_window_width},
        {"name": "triplet_150_200ms", "centers": [0.150, 0.175, 0.200], "window_size": default_window_width},
    ]
    window_sets: list[tuple[str, tuple[WindowSpec, ...]]] = []
    for item in raw_window_sets:
        if not isinstance(item, Mapping):
            raise ValueError("window_sets entries must be mappings.")
        window_sets.extend(
            _window_sets_from_config_item(
                item,
                default_window_width=default_window_width,
                preprocessing=preprocessing,
            )
        )
    return window_sets


def _preprocessing_normalization_name(preprocessing: Mapping[str, Any]) -> str:
    """Return the configured subject-level normalization, accepting legacy aliases."""

    if "normalization" in preprocessing:
        return str(preprocessing["normalization"])
    return str(preprocessing.get("epoch_normalization", "none"))


def _resolve_output(config: Mapping[str, Any], *, config_dir: Path, key: str, default: str) -> Path:
    outputs = _section(config, "outputs")
    value = outputs.get(key, default)
    dataset_name = str(_section(config, "dataset").get("name", "dataset"))
    formatted = str(value).format(dataset=dataset_name)
    path = Path(formatted)
    if path.is_absolute():
        return path
    base = outputs.get("base_dir", "results/{dataset}")
    base_path = expand_path(str(base).format(dataset=dataset_name), base_dir=Path.cwd())
    return base_path / path


def _crop_data(data: np.ndarray, times: np.ndarray, *, tmin: float | None, tmax: float | None) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(times), dtype=bool)
    tolerance = 1e-12
    if tmin is not None:
        mask &= times >= float(tmin) - tolerance
    if tmax is not None:
        mask &= times <= float(tmax) + tolerance
    if not np.any(mask):
        raise ValueError(f"Crop window [{tmin}, {tmax}] does not overlap the epoch time axis.")
    return data[:, :, mask], times[mask]


def _apply_subject_epoch_normalization(
    data: np.ndarray,
    times: np.ndarray,
    normalization: str,
    *,
    baseline_window: tuple[float, float],
) -> np.ndarray:
    """Apply subject-local normalization using this subject's own epochs only."""

    normalization = _base.normalize_epoch_normalization(normalization)
    data = np.asarray(data, dtype=np.float64)
    if normalization == "none":
        return data.astype(np.float32, copy=False)
    if normalization == "subject_trial_z":
        mean = data.mean(axis=(1, 2), keepdims=True)
        std = _base._nonzero_std(data.std(axis=(1, 2), keepdims=True))
        return ((data - mean) / std).astype(np.float32, copy=False)
    if normalization == "subject_z":
        mean, std = _base._channel_mean_std(data)
        return ((data - mean) / std).astype(np.float32, copy=False)

    mask = _base._baseline_time_mask(times, baseline_window)
    baseline = data[:, :, mask]
    baseline_mean, baseline_std = _base._channel_mean_std(baseline)
    if normalization == "subject_baseline_z":
        return ((data - baseline_mean) / baseline_std).astype(np.float32, copy=False)
    if normalization == "subject_baseline_whiten":
        whitening = _base._baseline_channel_whitening_matrix(data, times, baseline_window)
        centered = data - baseline_mean
        whitened = np.einsum("ntc,dc->ntd", np.transpose(centered, (0, 2, 1)), whitening)
        return np.transpose(whitened, (0, 2, 1)).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported normalization: {normalization}")


def _load_subjects_from_config(
    config: Mapping[str, Any], *, config_dir: Path) -> tuple[dict[str, SubjectEpochs], LabelEncoder]:
    validate_dataset_config(config, base_dir=config_dir, check_files=True)
    dataset = _section(config, "dataset")
    if dataset.get("type") != "fieldtrip_mat":
        raise ValueError("bushmeg-source-loso currently expects dataset.type='fieldtrip_mat'.")

    metadata_config = _section(config, "metadata")
    matlab_config = _section(config, "matlab")
    preprocessing = _section(config, "preprocessing")
    decoding = _section(config, "decoding") or _section(config, "workflow")
    source_loso = _section(config, "source_loso")
    label_column = str(decoding.get("label_column", "stimulus_class"))
    group_column = str(source_loso.get("group_column", decoding.get("group_column", "participant")))
    baseline_window = tuple(preprocessing.get("baseline_window", _base.DEFAULT_BASELINE_WINDOW))
    normalization = _preprocessing_normalization_name(preprocessing)
    tmin = preprocessing.get("tmin")
    tmax = preprocessing.get("tmax")

    loader_config: dict[str, Any] = {**matlab_config, **dataset, "metadata": metadata_config}
    if "validation" in config:
        loader_config["validation"] = _validation_section(config)

    loaded: dict[str, SubjectEpochs] = {}
    all_label_values: list[Any] = []
    staged: list[tuple[str, np.ndarray, np.ndarray, pd.DataFrame]] = []
    for path, extra_metadata in _fieldtrip_file_specs(config, base_dir=config_dir):
        dataset_epochs = load_fieldtrip_mat_epochs(path, loader_config, extra_metadata=extra_metadata)
        metadata = dataset_epochs.metadata.reset_index(drop=True).copy()
        if group_column not in metadata.columns:
            participant = str(extra_metadata.get("participant", path.stem))
            metadata[group_column] = participant
        subject_values = pd.unique(metadata[group_column].astype(str))
        if len(subject_values) != 1:
            raise ValueError(f"Expected one {group_column} value per file; found {subject_values.tolist()} in {path}.")
        subject = str(subject_values[0])
        if label_column not in metadata.columns:
            raise ValueError(f"Label column '{label_column}' not found in {path} metadata.")
        keep = pd.notna(metadata[label_column]).to_numpy()
        if not np.any(keep):
            raise ValueError(f"No labeled trials remain for subject {subject}.")
        data = dataset_epochs.data[keep]
        metadata = metadata.loc[keep].reset_index(drop=True)
        data, times = _crop_data(data, dataset_epochs.times, tmin=tmin, tmax=tmax)
        normalized = _apply_subject_epoch_normalization(
            data,
            times,
            normalization,
            baseline_window=(float(baseline_window[0]), float(baseline_window[1])),
        )
        staged.append((subject, normalized, times.astype(float, copy=True), metadata))
        all_label_values.extend(metadata[label_column].tolist())

    encoder = LabelEncoder().fit(np.asarray(all_label_values, dtype=object))
    for subject, data, times, metadata in staged:
        labels = encoder.transform(metadata[label_column].to_numpy())
        loaded[subject] = SubjectEpochs(
            subject=subject,
            data=data,
            times=times,
            metadata=metadata,
            labels=labels.astype(int, copy=False),
        )
    if len(loaded) < 3:
        raise ValueError("Need at least three subjects for outer and inner LOSO decoding.")
    return loaded, encoder


def _sample_indices_for_window(times: np.ndarray, window: WindowSpec) -> np.ndarray:
    tolerance = 1e-12
    indices = np.flatnonzero((times >= window.start - tolerance) & (times <= window.stop + tolerance))
    if indices.size == 0:
        raise ValueError(
            f"Window centered at {window.center:.6g}s with width {window.width:.6g}s "
            f"does not overlap available times [{times[0]:.6g}, {times[-1]:.6g}]."
        )
    return indices


def _window_bin_mean_features(
    data: np.ndarray,
    times: np.ndarray,
    window: WindowSpec,
    *,
    temporal_bins: int,
) -> np.ndarray:
    """Return trial × (channel×bin) features using per-bin temporal means."""

    if temporal_bins < 1:
        raise ValueError("temporal_bins must be at least one.")
    indices = _sample_indices_for_window(times, window)
    bins = np.array_split(indices, int(temporal_bins))
    if any(len(bin_indices) == 0 for bin_indices in bins):
        raise ValueError(
            f"Window {window.center:.6g}s/{window.width:.6g}s has only {len(indices)} samples, "
            f"not enough for {temporal_bins} temporal bins."
        )
    features = [data[:, :, bin_indices].mean(axis=2) for bin_indices in bins]
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _row_center_unit_scale(features: np.ndarray) -> np.ndarray:
    """Center each feature vector and scale it to unit Euclidean norm."""

    features = np.asarray(features, dtype=np.float64)
    centered = features - features.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    return centered / np.where(norms < 1e-12, 1.0, norms)


def _template_similarity_features(features: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Return centered-cosine similarities from trials to class templates."""

    similarities = _row_center_unit_scale(features) @ _row_center_unit_scale(templates).T
    return np.clip(similarities, -1.0, 1.0).astype(np.float32, copy=False)


def _train_standardized_features(
    train_features: np.ndarray, test_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize train/test features with source-train statistics only."""

    train = np.asarray(train_features, dtype=np.float64)
    test = np.asarray(test_features, dtype=np.float64)
    mean = train.mean(axis=0, keepdims=True)
    scale = _base._nonzero_std(train.std(axis=0, keepdims=True))
    return (train - mean) / scale, (test - mean) / scale


def _class_means(
    features: np.ndarray, labels: np.ndarray, *, n_classes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Prototype features and labels have incompatible lengths.")
    class_sums = np.zeros((int(n_classes), features.shape[1]), dtype=np.float64)
    np.add.at(class_sums, labels, features)
    counts = np.bincount(labels, minlength=int(n_classes)).astype(
        np.float64, copy=False
    )
    means = np.zeros_like(class_sums)
    valid = counts > 0.0
    means[valid] = class_sums[valid] / counts[valid, None]
    return means, class_sums, counts


def _prototype_score_matrix(
    features: np.ndarray, prototypes: np.ndarray, *, epsilon: float = 1e-12
) -> np.ndarray:
    """Return cosine and negative-squared-distance scores to each class prototype."""

    features = np.asarray(features, dtype=np.float64)
    prototypes = np.asarray(prototypes, dtype=np.float64)
    dot = features @ prototypes.T
    feature_norm = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), epsilon)
    prototype_norm = np.maximum(
        np.linalg.norm(prototypes, axis=1, keepdims=True).T, epsilon
    )
    cosine = dot / (feature_norm * prototype_norm)
    feature_sq = np.sum(features * features, axis=1, keepdims=True)
    prototype_sq = np.sum(prototypes * prototypes, axis=1, keepdims=True).T
    squared_distance = np.maximum(feature_sq + prototype_sq - 2.0 * dot, 0.0)
    distance_score = -squared_distance / float(max(features.shape[1], 1))
    return np.concatenate([cosine, distance_score], axis=1)


def _paired_prototype_scores(
    features: np.ndarray, prototypes: np.ndarray, *, epsilon: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired cosine and distance scores for row-wise leave-one-out prototypes."""

    dot = np.sum(features * prototypes, axis=1)
    feature_norm = np.maximum(np.linalg.norm(features, axis=1), epsilon)
    prototype_norm = np.maximum(np.linalg.norm(prototypes, axis=1), epsilon)
    cosine = dot / (feature_norm * prototype_norm)
    distance_score = -np.sum((features - prototypes) ** 2, axis=1) / float(
        max(features.shape[1], 1)
    )
    return cosine, distance_score


def _class_prototype_similarity_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    *,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build train-source-only class-prototype features for a candidate window."""

    train_z, test_z = _train_standardized_features(train_features, test_features)
    train_labels = np.asarray(train_labels, dtype=int).reshape(-1)
    prototypes, class_sums, counts = _class_means(
        train_z, train_labels, n_classes=int(n_classes)
    )
    train_scores = _prototype_score_matrix(train_z, prototypes)
    test_scores = _prototype_score_matrix(test_z, prototypes)
    for class_idx in np.unique(train_labels):
        class_idx = int(class_idx)
        row_indices = np.flatnonzero(train_labels == class_idx)
        if counts[class_idx] > 1.0:
            loo_prototypes = (
                class_sums[class_idx][None, :] - train_z[row_indices]
            ) / (counts[class_idx] - 1.0)
            loo_cosine, loo_distance = _paired_prototype_scores(
                train_z[row_indices], loo_prototypes
            )
        else:
            loo_cosine = np.zeros(row_indices.shape[0], dtype=np.float64)
            loo_distance = -np.sum(
                train_z[row_indices] * train_z[row_indices], axis=1
            ) / float(max(train_z.shape[1], 1))
        train_scores[row_indices, class_idx] = loo_cosine
        train_scores[row_indices, int(n_classes) + class_idx] = loo_distance
    return train_scores.astype(np.float32, copy=False), test_scores.astype(
        np.float32, copy=False
    )


def _class_template_features(
    cache: FeatureCache,
    subjects: Mapping[str, SubjectEpochs],
    reference_subjects: Sequence[str],
    window: WindowSpec,
    temporal_bins: int,
    n_classes: int,
    *,
    feature_kind: str = "evoked",
    covariance_max_channels: int = DEFAULT_COVARIANCE_MAX_CHANNELS,
) -> np.ndarray:
    """Build subject-balanced class prototypes from source subjects only."""

    reference_subjects = tuple(reference_subjects)
    if not reference_subjects:
        raise ValueError("At least one reference subject is required for template-similarity features.")
    features_by_subject = {
        subject_id: cache.get(
            subject_id,
            window,
            temporal_bins,
            feature_kind=feature_kind,
            covariance_max_channels=covariance_max_channels,
        )
        for subject_id in reference_subjects
    }
    labels_by_subject = {subject_id: subjects[subject_id].labels for subject_id in reference_subjects}
    fallback_template = np.mean(
        np.vstack([features.mean(axis=0, dtype=np.float64) for features in features_by_subject.values()]),
        axis=0,
    )

    templates: list[np.ndarray] = []
    for class_idx in range(int(n_classes)):
        subject_means = []
        for subject_id in reference_subjects:
            labels = labels_by_subject[subject_id]
            mask = labels == class_idx
            if np.any(mask):
                subject_means.append(features_by_subject[subject_id][mask].mean(axis=0, dtype=np.float64))
        templates.append(np.mean(np.vstack(subject_means), axis=0) if subject_means else fallback_template)
    return np.vstack(templates).astype(np.float32, copy=False)


def _prepare_window_train_test_features(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: FeatureCache,
    candidate: CandidateSpec,
    train_subjects: Sequence[str],
    test_subject: str,
    window: WindowSpec,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fold-local features for one candidate/window."""

    feature_family = normalize_source_feature_family(candidate.feature_family)
    feature_kind = normalize_source_feature_kind(candidate.feature_kind)
    if feature_kind in PROTOTYPE_FEATURE_KINDS:
        base_feature_kind = PROTOTYPE_BASE_FEATURE_KINDS.get(feature_kind, "evoked")
        train_features, test_features = _prepare_window_train_test_features(
            subjects=subjects,
            cache=cache,
            candidate=replace(candidate, feature_kind=base_feature_kind),
            train_subjects=train_subjects,
            test_subject=test_subject,
            window=window,
            n_classes=n_classes,
        )
        prototype_train_features, prototype_test_features = (
            _class_prototype_similarity_features(
                train_features,
                test_features,
                _stack_subject_labels(subjects, train_subjects),
                n_classes=n_classes,
            )
        )
        if feature_kind == "prototype":
            return prototype_train_features, prototype_test_features
        return (
            np.concatenate([train_features, prototype_train_features], axis=1).astype(
                np.float32, copy=False
            ),
            np.concatenate([test_features, prototype_test_features], axis=1).astype(
                np.float32, copy=False
            ),
        )
    if feature_family == "bin_means":
        return (
            _stack_subject_features(
                cache,
                subjects,
                train_subjects,
                window,
                candidate.temporal_bins,
                feature_kind=feature_kind,
                covariance_max_channels=candidate.covariance_max_channels,
            ),
            cache.get(
                test_subject,
                window,
                candidate.temporal_bins,
                feature_kind=feature_kind,
                covariance_max_channels=candidate.covariance_max_channels,
            ),
        )

    train_features_by_subject: list[np.ndarray] = []
    for subject_id in train_subjects:
        subject_features = cache.get(
            subject_id,
            window,
            candidate.temporal_bins,
            feature_kind=feature_kind,
            covariance_max_channels=candidate.covariance_max_channels,
        )
        reference_subjects = [candidate_subject for candidate_subject in train_subjects if candidate_subject != subject_id]
        if not reference_subjects:
            reference_subjects = list(train_subjects)
        similarities = _template_similarity_features(
            subject_features,
            cache.class_templates(
                reference_subjects,
                window,
                candidate.temporal_bins,
                n_classes,
                feature_kind=feature_kind,
                covariance_max_channels=candidate.covariance_max_channels,
            ),
        )
        if feature_family == "template_similarity_plus_bin_means":
            similarities = np.concatenate([subject_features, similarities], axis=1).astype(np.float32, copy=False)
        train_features_by_subject.append(similarities)

    test_features = cache.get(
        test_subject,
        window,
        candidate.temporal_bins,
        feature_kind=feature_kind,
        covariance_max_channels=candidate.covariance_max_channels,
    )
    test_similarities = _template_similarity_features(
        test_features,
        cache.class_templates(
            train_subjects,
            window,
            candidate.temporal_bins,
            n_classes,
            feature_kind=feature_kind,
            covariance_max_channels=candidate.covariance_max_channels,
        ),
    )
    if feature_family == "template_similarity_plus_bin_means":
        test_similarities = np.concatenate([test_features, test_similarities], axis=1).astype(np.float32, copy=False)
    return np.concatenate(train_features_by_subject, axis=0), test_similarities


def _window_log_variance_features(
    data: np.ndarray,
    times: np.ndarray,
    window: WindowSpec,
    *,
    temporal_bins: int,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return trial x (channel x bin) log-variance features."""

    if temporal_bins < 1:
        raise ValueError("temporal_bins must be at least one.")
    indices = _sample_indices_for_window(times, window)
    bins = np.array_split(indices, int(temporal_bins))
    if any(len(bin_indices) == 0 for bin_indices in bins):
        raise ValueError(
            f"Window {window.center:.6g}s/{window.width:.6g}s has only {len(indices)} samples, "
            f"not enough for {temporal_bins} temporal bins."
        )
    features = []
    for bin_indices in bins:
        ddof = 1 if len(bin_indices) > 1 else 0
        variance = np.var(data[:, :, bin_indices], axis=2, ddof=ddof)
        features.append(np.log(np.maximum(variance, epsilon)))
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _channel_subset_indices(n_channels: int, max_channels: int) -> np.ndarray:
    max_channels = max(1, int(max_channels))
    if n_channels <= max_channels:
        return np.arange(n_channels, dtype=int)
    return np.unique(np.linspace(0, n_channels - 1, max_channels, dtype=int))


def _window_covariance_features(
    data: np.ndarray,
    times: np.ndarray,
    window: WindowSpec,
    *,
    covariance_max_channels: int = DEFAULT_COVARIANCE_MAX_CHANNELS,
    shrinkage: float = 0.10,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Return compact per-trial shrinkage covariance features."""

    indices = _sample_indices_for_window(times, window)
    channel_indices = _channel_subset_indices(data.shape[1], covariance_max_channels)
    window_data = np.asarray(data[:, channel_indices][:, :, indices], dtype=np.float64)
    n_trials, n_channels, n_times = window_data.shape
    tri = np.triu_indices(n_channels)
    features = np.empty((n_trials, len(tri[0])), dtype=np.float32)
    identity = np.eye(n_channels, dtype=np.float64)
    for trial_index in range(n_trials):
        trial = window_data[trial_index] - window_data[trial_index].mean(axis=1, keepdims=True)
        denom = max(n_times - 1, 1)
        covariance = (trial @ trial.T) / float(denom)
        mean_variance = float(np.trace(covariance) / max(n_channels, 1))
        covariance = (
            (1.0 - float(shrinkage)) * covariance
            + float(shrinkage) * mean_variance * identity
        )
        covariance /= max(float(np.trace(covariance)), epsilon)
        features[trial_index] = covariance[tri]
    return features


def _mnn_base_feature_kind(feature_kind: str) -> str | None:
    """Return the underlying feature kind for an MNN-prefixed feature kind."""

    if not feature_kind.startswith("mnn_"):
        return None
    return feature_kind[len("mnn_") :]


def _mnn_baseline_indices(
    times: np.ndarray,
    *,
    baseline_window: tuple[float, float] = DEFAULT_MNN_BASELINE_WINDOW,
) -> np.ndarray:
    """Return MNN baseline indices, falling back to all negative samples."""

    times = np.asarray(times, dtype=float)
    tolerance = 1e-12
    start, stop = baseline_window
    indices = np.flatnonzero((times >= float(start) - tolerance) & (times <= float(stop) + tolerance))
    if indices.size == 0:
        indices = np.flatnonzero(times < -tolerance)
    if indices.size == 0:
        raise ValueError(
            "MNN feature extraction requires pre-stimulus baseline samples; "
            f"no samples were found in {baseline_window} or at times < 0."
        )
    return indices


def _baseline_channel_whitener(
    data: np.ndarray,
    times: np.ndarray,
    *,
    baseline_window: tuple[float, float] = DEFAULT_MNN_BASELINE_WINDOW,
    shrinkage: float = 0.10,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Estimate a subject-local baseline covariance inverse square root."""

    baseline_indices = _mnn_baseline_indices(times, baseline_window=baseline_window)
    baseline = np.asarray(data[:, :, baseline_indices], dtype=np.float64)
    n_channels = baseline.shape[1]
    flattened = np.transpose(baseline, (1, 0, 2)).reshape(n_channels, -1)
    flattened -= flattened.mean(axis=1, keepdims=True)
    covariance = (flattened @ flattened.T) / float(max(flattened.shape[1] - 1, 1))
    mean_variance = float(np.trace(covariance) / max(n_channels, 1))
    covariance = (1.0 - float(shrinkage)) * covariance + float(shrinkage) * mean_variance * np.eye(n_channels)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    inverse_sqrt = 1.0 / np.sqrt(np.maximum(eigenvalues, float(epsilon)))
    return (eigenvectors * inverse_sqrt[None, :]) @ eigenvectors.T


def _apply_channel_whitener(data: np.ndarray, whitener: np.ndarray) -> np.ndarray:
    """Apply a channel-space whitening/projection matrix to trial epochs."""

    return np.einsum("dc,nct->ndt", np.asarray(whitener, dtype=np.float64), np.asarray(data, dtype=np.float64))


def _window_features(
    data: np.ndarray,
    times: np.ndarray,
    window: WindowSpec,
    *,
    temporal_bins: int,
    feature_kind: str = "evoked",
    covariance_max_channels: int = DEFAULT_COVARIANCE_MAX_CHANNELS,
) -> np.ndarray:
    """Return source-LOSO features for one subject/window."""

    feature_kind = normalize_source_feature_kind(feature_kind)
    mnn_base_kind = _mnn_base_feature_kind(feature_kind)
    if mnn_base_kind is not None:
        if mnn_base_kind in SUPERVISED_FEATURE_KINDS:
            raise ValueError(f"{feature_kind} features are supervised and must be fitted inside _predict_candidate.")
        whitened = _apply_channel_whitener(
            data,
            _baseline_channel_whitener(data, times),
        )
        return _window_features(
            whitened,
            times,
            window,
            temporal_bins=temporal_bins,
            feature_kind=mnn_base_kind,
            covariance_max_channels=covariance_max_channels,
        )
    if feature_kind in SUPERVISED_FEATURE_KINDS:
        raise ValueError(f"{feature_kind} features are supervised and must be fitted inside _predict_candidate.")
    evoked = None
    if feature_kind in {"evoked", "evoked_logvar", "evoked_covariance"}:
        evoked = _window_bin_mean_features(data, times, window, temporal_bins=temporal_bins)
    if feature_kind == "evoked":
        assert evoked is not None
        return evoked
    if feature_kind == "logvar":
        return _window_log_variance_features(data, times, window, temporal_bins=temporal_bins)
    if feature_kind == "covariance":
        return _window_covariance_features(
            data,
            times,
            window,
            covariance_max_channels=covariance_max_channels,
        )
    if feature_kind == "evoked_logvar":
        assert evoked is not None
        logvar = _window_log_variance_features(data, times, window, temporal_bins=temporal_bins)
        return np.concatenate([evoked, logvar], axis=1).astype(np.float32, copy=False)
    if feature_kind == "evoked_covariance":
        assert evoked is not None
        covariance = _window_covariance_features(
            data,
            times,
            window,
            covariance_max_channels=covariance_max_channels,
        )
        return np.concatenate([evoked, covariance], axis=1).astype(np.float32, copy=False)
    raise AssertionError(f"Unhandled feature kind: {feature_kind}")


def _stack_subject_data(subjects: Mapping[str, SubjectEpochs], subject_ids: Sequence[str]) -> np.ndarray:
    return np.concatenate([subjects[subject_id].data for subject_id in subject_ids], axis=0)


def _stack_subject_features(
    cache: FeatureCache,
    subjects: Mapping[str, SubjectEpochs],
    subject_ids: Sequence[str],
    window: WindowSpec,
    temporal_bins: int,
    *,
    feature_kind: str = "evoked",
    covariance_max_channels: int = DEFAULT_COVARIANCE_MAX_CHANNELS,
) -> np.ndarray:
    return np.concatenate(
        [
            cache.get(
                subject_id,
                window,
                temporal_bins,
                feature_kind=feature_kind,
                covariance_max_channels=covariance_max_channels,
            )
            for subject_id in subject_ids
        ],
        axis=0,
    )


def _stack_subject_labels(subjects: Mapping[str, SubjectEpochs], subject_ids: Sequence[str]) -> np.ndarray:
    return np.concatenate([subjects[subject_id].labels for subject_id in subject_ids], axis=0)


def _stack_subject_ids(subjects: Mapping[str, SubjectEpochs], subject_ids: Sequence[str]) -> np.ndarray:
    return np.concatenate([np.full(len(subjects[subject_id].labels), subject_id, dtype=object) for subject_id in subject_ids], axis=0)


def _sample_weights_for_training(
    subjects: Mapping[str, SubjectEpochs],
    subject_ids: Sequence[str],
    labels: np.ndarray,
    mode: str,
    subject_weight_multipliers: Mapping[str, float] | None = None,
) -> np.ndarray | None:
    """Return mean-one training weights for class/subject/cue-balanced fitting."""

    mode = _normalize_sample_weighting(mode)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    sample_subjects = _stack_subject_ids(subjects, subject_ids)
    if mode == "none":
        weights = np.ones(labels.shape[0], dtype=np.float64)
    else:
        if mode == "class_balanced":
            keys = [(int(label),) for label in labels]
        elif mode == "subject_balanced":
            keys = [(str(subject),) for subject in sample_subjects]
        elif mode == "subject_class_balanced":
            keys = [(str(subject), int(label)) for subject, label in zip(sample_subjects, labels, strict=True)]
        else:  # pragma: no cover - guarded by _normalize_sample_weighting
            raise ValueError(f"Unsupported sample weighting mode: {mode}")

        counts: dict[tuple[Any, ...], int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        weights = np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)

    if subject_weight_multipliers:
        lookup = {str(subject): float(weight) for subject, weight in subject_weight_multipliers.items()}
        multipliers = np.asarray([lookup.get(str(subject), 1.0) for subject in sample_subjects], dtype=np.float64)
        if np.any(~np.isfinite(multipliers)) or np.any(multipliers < 0.0):
            raise ValueError("Cue/source subject weight multipliers must be finite and non-negative.")
        weights *= multipliers
    elif mode == "none":
        return None

    mean_weight = float(weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        raise ValueError("Training sample weights must have positive finite mean.")
    return weights / mean_weight


def _fit_candidate_model(
    model: Any,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    sample_weight: np.ndarray | None,
):
    """Fit a sklearn decoder, routing sample weights through pipelines."""

    if sample_weight is None:
        model.fit(features, labels)
        return model
    sample_weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    if sample_weight.shape[0] != labels.shape[0]:
        raise ValueError("sample_weight must contain one weight per training row.")
    try:
        model.fit(features, labels, sample_weight=sample_weight)
        return model
    except (TypeError, ValueError) as exc:
        if not hasattr(model, "steps") or not getattr(model, "steps"):
            warnings.warn(f"{model.__class__.__name__} does not accept sample_weight; fitting without weights.", RuntimeWarning, stacklevel=2)
            model.fit(features, labels)
            return model
        final_step_name = model.steps[-1][0]
        try:
            model.fit(features, labels, **{f"{final_step_name}__sample_weight": sample_weight})
            return model
        except (TypeError, ValueError) as routed_exc:
            warnings.warn(
                f"{model.steps[-1][1].__class__.__name__} does not accept sample_weight; fitting without weights. "
                f"First error: {exc}; routed error: {routed_exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            model.fit(features, labels)
            return model
    return model


def _apply_class_bias(probabilities: np.ndarray, bias: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    bias = np.asarray(bias, dtype=float).reshape(-1)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix.")
    if bias.shape[0] != probabilities.shape[1]:
        raise ValueError("class-bias vector length must match the number of probability columns.")
    logits = np.log(np.clip(probabilities, 1e-12, 1.0)) + bias[None, :]
    logits -= np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits, -50.0, 50.0))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _fit_class_bias(probabilities: np.ndarray, labels: np.ndarray, *, n_classes: int, mode: str) -> np.ndarray:
    """Fit a target-free per-class logit bias from source training predictions."""

    mode = _normalize_class_bias(mode)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if mode == "none":
        return np.zeros(n_classes, dtype=float)
    if mode == "log_prior":
        counts = np.bincount(labels, minlength=n_classes).astype(float)
        priors = (counts + 1.0) / (float(counts.sum()) + float(n_classes))
        bias = -np.log(priors)
        return bias - float(bias.mean())
    if mode == "balanced_accuracy":
        return _fit_balanced_accuracy_class_bias(probabilities, labels, n_classes=n_classes)
    raise ValueError(f"Unsupported class-bias mode: {mode}")  # pragma: no cover


def _fit_balanced_accuracy_class_bias(probabilities: np.ndarray, labels: np.ndarray, *, n_classes: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if probabilities.shape != (labels.shape[0], n_classes):
        raise ValueError("probabilities must have shape (n_trials, n_classes).")
    log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0))
    bias = np.zeros(n_classes, dtype=float)
    best_score = float(balanced_accuracy_score(labels, np.argmax(log_probabilities, axis=1)))
    for _ in range(DEFAULT_CLASS_BIAS_ROUNDS):
        improved = False
        for class_idx in range(n_classes):
            best_candidate = bias.copy()
            best_candidate_score = best_score
            for delta in DEFAULT_CLASS_BIAS_DELTAS:
                candidate = bias.copy()
                candidate[class_idx] += float(delta)
                candidate -= float(candidate.mean())
                score = float(balanced_accuracy_score(labels, np.argmax(log_probabilities + candidate[None, :], axis=1)))
                if score > best_candidate_score + 1e-12:
                    best_candidate = candidate
                    best_candidate_score = score
            if best_candidate_score > best_score + 1e-12:
                bias = best_candidate
                best_score = best_candidate_score
                improved = True
        if not improved:
            break
    return bias


def _fit_xdawn_filters(
    data: np.ndarray,
    labels: np.ndarray,
    times: np.ndarray,
    window: WindowSpec,
    *,
    n_components: int,
) -> np.ndarray:
    """Fit supervised ERP-denoising spatial filters on source-train epochs."""

    indices = _sample_indices_for_window(times, window)
    x = np.asarray(data[:, :, indices], dtype=np.float64)
    labels = np.asarray(labels)
    n_channels = x.shape[1]
    n_components = min(max(1, int(n_components)), n_channels)
    flattened = np.transpose(x, (1, 0, 2)).reshape(n_channels, -1)
    flattened -= flattened.mean(axis=1, keepdims=True)
    data_cov = (flattened @ flattened.T) / float(max(flattened.shape[1] - 1, 1))
    signal_cov = np.zeros_like(data_cov)
    for class_label in np.unique(labels):
        class_epochs = x[labels == class_label]
        if class_epochs.size == 0:
            continue
        evoked = class_epochs.mean(axis=0)
        evoked -= evoked.mean(axis=1, keepdims=True)
        signal_cov += (
            (class_epochs.shape[0] / x.shape[0])
            * (evoked @ evoked.T)
            / float(max(evoked.shape[1] - 1, 1))
        )
    ridge = max(float(np.trace(data_cov)) / max(n_channels, 1), 1.0) * 1e-6
    eigenvalues, eigenvectors = eigh(
        signal_cov,
        data_cov + ridge * np.eye(n_channels),
        check_finite=False,
    )
    order = np.argsort(eigenvalues)[::-1][:n_components]
    filters = np.asarray(eigenvectors[:, order], dtype=np.float64)
    norms = np.linalg.norm(filters, axis=0, keepdims=True)
    return filters / np.maximum(norms, 1e-12)


def _xdawn_bin_mean_features(
    data: np.ndarray,
    times: np.ndarray,
    window: WindowSpec,
    *,
    filters: np.ndarray,
    temporal_bins: int,
) -> np.ndarray:
    indices = _sample_indices_for_window(times, window)
    projected = np.einsum("ck,nct->nkt", filters, np.asarray(data[:, :, indices], dtype=np.float64))
    relative_bins = np.array_split(np.arange(projected.shape[2]), int(temporal_bins))
    if any(len(bin_indices) == 0 for bin_indices in relative_bins):
        raise ValueError(
            f"Window {window.center:.6g}s/{window.width:.6g}s has only {len(indices)} samples, "
            f"not enough for {temporal_bins} temporal bins."
        )
    features = [projected[:, :, bin_indices].mean(axis=2) for bin_indices in relative_bins]
    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _xdawn_train_test_features(
    *,
    subjects: Mapping[str, SubjectEpochs],
    train_subjects: Sequence[str],
    test_subject: str,
    train_labels: np.ndarray,
    window: WindowSpec,
    temporal_bins: int,
    n_components: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    train_data = _stack_subject_data(subjects, train_subjects)
    train_times = subjects[train_subjects[0]].times
    filters = _fit_xdawn_filters(
        train_data,
        train_labels,
        train_times,
        window,
        n_components=DEFAULT_XDAWN_COMPONENTS if n_components is None else int(n_components),
    )
    train_features = _xdawn_bin_mean_features(
        train_data,
        train_times,
        window,
        filters=filters,
        temporal_bins=temporal_bins,
    )
    test_epochs = subjects[test_subject]
    test_features = _xdawn_bin_mean_features(
        test_epochs.data,
        test_epochs.times,
        window,
        filters=filters,
        temporal_bins=temporal_bins,
    )
    return train_features, test_features


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    effective_k = min(int(k), probabilities.shape[1])
    kth_scores = np.partition(probabilities, -effective_k, axis=1)[:, -effective_k]
    label_scores = probabilities[np.arange(labels.size), labels]
    return float(np.mean(label_scores >= kth_scores))


def _effective_pca_components(candidate: CandidateSpec, n_features: int | None, n_samples: int | None) -> int | float | None:
    pca_components = candidate.pca_components
    if n_features is None or pca_components is None:
        return pca_components
    if normalize_feature_preprocessor(candidate.feature_preprocessor) not in {"pca", "pca_whiten"}:
        return pca_components
    if isinstance(pca_components, (int, np.integer)):
        max_components = int(n_features) if n_samples is None else min(int(n_features), int(n_samples))
        return min(int(pca_components), max(1, max_components))
    return pca_components


def _candidate_model(candidate: CandidateSpec, *, max_iter: int, n_features: int | None = None, n_samples: int | None = None):
    pca_components = _effective_pca_components(candidate, n_features, n_samples)
    return make_decoder(
        candidate.decoder,
        max_iter=max_iter,
        emission_mode=candidate.emission_mode,
        feature_preprocessor=candidate.feature_preprocessor,
        pca_components=pca_components,
        classifier_param=candidate.classifier_param,
        random_state=DEFAULT_RANDOM_SEED,
    )


def _predict_candidate(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: FeatureCache,
    candidate: CandidateSpec,
    train_subjects: Sequence[str],
    test_subject: str,
    n_classes: int,
    max_iter: int,
    subject_weight_multipliers: Mapping[str, float] | None = None,
) -> np.ndarray:
    train_labels = _stack_subject_labels(subjects, train_subjects)
    sample_weight = _sample_weights_for_training(
        subjects,
        train_subjects,
        train_labels,
        candidate.sample_weighting,
        subject_weight_multipliers=subject_weight_multipliers,
    )
    test_n = len(subjects[test_subject].labels)
    probabilities_sum = np.zeros((test_n, n_classes), dtype=float)
    class_bias_mode = _normalize_class_bias(candidate.class_bias)
    train_probabilities_sum = np.zeros((len(train_labels), n_classes), dtype=float) if class_bias_mode != "none" else None
    classes = np.arange(n_classes)
    feature_kind = normalize_source_feature_kind(candidate.feature_kind)
    for window in candidate.windows:
        if feature_kind in {"xdawn", "xdawn_prototype"}:
            train_features, test_features = _xdawn_train_test_features(
                subjects=subjects,
                train_subjects=train_subjects,
                test_subject=test_subject,
                train_labels=train_labels,
                window=window,
                temporal_bins=candidate.temporal_bins,
                n_components=candidate.xdawn_components,
            )
            if feature_kind == "xdawn_prototype":
                prototype_train_features, prototype_test_features = (
                    _class_prototype_similarity_features(
                        train_features,
                        test_features,
                        train_labels,
                        n_classes=n_classes,
                    )
                )
                train_features = np.concatenate(
                    [train_features, prototype_train_features], axis=1
                ).astype(np.float32, copy=False)
                test_features = np.concatenate(
                    [test_features, prototype_test_features], axis=1
                ).astype(np.float32, copy=False)
        else:
            train_features, test_features = _prepare_window_train_test_features(
                subjects=subjects,
                cache=cache,
                candidate=candidate,
                train_subjects=train_subjects,
                test_subject=test_subject,
                window=window,
                n_classes=n_classes,
            )
        model = _candidate_model(candidate, max_iter=max_iter, n_features=train_features.shape[1], n_samples=train_features.shape[0])
        _fit_candidate_model(model, train_features, train_labels, sample_weight=sample_weight)
        probabilities = predict_emission_probabilities(
            model,
            test_features,
            emission_mode=candidate.emission_mode,
        )
        probabilities_sum += _base._align_probability_columns(
            probabilities,
            model=model,
            classes=classes,
        )
        if train_probabilities_sum is not None:
            train_probabilities = predict_emission_probabilities(
                model,
                train_features,
                emission_mode=candidate.emission_mode,
            )
            train_probabilities_sum += _base._align_probability_columns(
                train_probabilities,
                model=model,
                classes=classes,
            )
    averaged = _base._probability_average(probabilities_sum, len(candidate.windows))
    if train_probabilities_sum is None:
        return averaged
    train_averaged = _base._probability_average(train_probabilities_sum, len(candidate.windows))
    bias = _fit_class_bias(train_averaged, train_labels, n_classes=n_classes, mode=class_bias_mode)
    return _apply_class_bias(averaged, bias)


def _candidate_metrics(probabilities: np.ndarray, labels: np.ndarray, *, n_classes: int) -> dict[str, float]:
    predictions = probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "top2_accuracy": _top_k_accuracy(probabilities, labels, k=2),
        "top3_accuracy": _top_k_accuracy(probabilities, labels, k=3),
    }
    metrics["log_loss"] = float(log_loss(labels, probabilities, labels=np.arange(n_classes)))
    return metrics


def _score_is_better(candidate_score: float, incumbent_score: float | None, *, metric: str) -> bool:
    if incumbent_score is None:
        return True
    if metric in MINIMIZE_SELECTION_METRICS:
        return candidate_score < incumbent_score
    return candidate_score > incumbent_score


def _candidate_rowspec(candidate: CandidateSpec) -> dict[str, Any]:
    return {
        "candidate": candidate.name,
        "feature_family": normalize_source_feature_family(candidate.feature_family),
        "decoder": normalize_decoder_name(candidate.decoder),
        "emission_mode": normalize_emission_mode(candidate.emission_mode),
        "feature_preprocessor": normalize_feature_preprocessor(candidate.feature_preprocessor),
        "pca_components": "" if candidate.pca_components is None else candidate.pca_components,
        "classifier_param": "" if candidate.classifier_param is None else candidate.classifier_param,
        "sample_weighting": _normalize_sample_weighting(candidate.sample_weighting),
        "class_bias": _normalize_class_bias(candidate.class_bias),
        "temporal_bins": candidate.temporal_bins,
        "feature_kind": normalize_source_feature_kind(candidate.feature_kind),
        "xdawn_components": "" if candidate.xdawn_components is None else candidate.xdawn_components,
        "covariance_max_channels": candidate.covariance_max_channels,
        "n_windows": len(candidate.windows),
        "window_centers": "|".join(f"{center:.6g}" for center in candidate.window_centers),
        "window_widths": "|".join(f"{width:.6g}" for width in candidate.window_widths),
    }


def _inner_loso_scores(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: FeatureCache,
    candidate: CandidateSpec,
    outer_test_subject: str,
    n_classes: int,
    max_iter: int,
    cue_source_weights: CueSourceWeights | None = None,
) -> list[dict[str, Any]]:
    source_subjects = [subject for subject in sorted(subjects) if subject != outer_test_subject]
    rows: list[dict[str, Any]] = []
    for inner_test_subject in source_subjects:
        train_subjects = [subject for subject in source_subjects if subject != inner_test_subject]
        probabilities = _predict_candidate(
            subjects=subjects,
            cache=cache,
            candidate=candidate,
            train_subjects=train_subjects,
            test_subject=inner_test_subject,
            n_classes=n_classes,
            max_iter=max_iter,
            subject_weight_multipliers=None if cue_source_weights is None else cue_source_weights.for_fold(inner_test_subject, train_subjects),
        )
        labels = subjects[inner_test_subject].labels
        rows.append(
            {
                "outer_test_subject": outer_test_subject,
                "inner_test_subject": inner_test_subject,
                **_candidate_rowspec(candidate),
                **_candidate_metrics(probabilities, labels, n_classes=n_classes),
                "n_train_subjects": len(train_subjects),
                "n_test_trials": len(labels),
            }
        )
    return rows


def _select_candidate(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: FeatureCache,
    candidates: Sequence[CandidateSpec],
    outer_test_subject: str,
    n_classes: int,
    max_iter: int,
    selection_metric: str,
    cue_source_weights: CueSourceWeights | None = None,
) -> tuple[CandidateSpec, list[dict[str, Any]], dict[str, Any]]:
    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unknown selection metric '{selection_metric}'. Available metrics: {sorted(SUPPORTED_SELECTION_METRICS)}.")
    all_rows: list[dict[str, Any]] = []
    selected: CandidateSpec | None = None
    selected_score: float | None = None
    selected_summary: dict[str, Any] = {}
    for candidate in candidates:
        rows = _inner_loso_scores(
            subjects=subjects,
            cache=cache,
            candidate=candidate,
            outer_test_subject=outer_test_subject,
            n_classes=n_classes,
            max_iter=max_iter,
            cue_source_weights=cue_source_weights,
        )
        all_rows.extend(rows)
        frame = pd.DataFrame(rows)
        mean_score = float(frame[selection_metric].mean())
        std_score = float(frame[selection_metric].std(ddof=0))
        if _score_is_better(mean_score, selected_score, metric=selection_metric):
            selected = candidate
            selected_score = mean_score
            selected_summary = {
                "inner_selection_metric": selection_metric,
                "inner_mean_score": mean_score,
                "inner_std_score": std_score,
                "inner_n_folds": len(frame),
            }
    if selected is None:
        raise ValueError("No candidates were available for source LOSO selection.")
    return selected, all_rows, selected_summary


def _candidate_grid(config: Mapping[str, Any]) -> list[CandidateSpec]:
    preprocessing = _section(config, "preprocessing")
    decoding = _section(config, "decoding") or _section(config, "workflow")
    source_loso = _section(config, "source_loso")
    grid = source_loso.get("candidate_grid", {}) or {}
    if not isinstance(grid, Mapping):
        raise ValueError("source_loso.candidate_grid must be a mapping.")

    window_sets = _candidate_window_sets(grid, preprocessing)

    decoders = [str(value) for value in _list_value(grid.get("decoders"), [decoding.get("decoder", decoding.get("classifier", "multinomial-logistic"))])]
    emission_modes = [str(value) for value in _list_value(grid.get("emission_modes"), [decoding.get("emission_mode", "uncalibrated")])]
    feature_preprocessors = [
        str(value)
        for value in _list_value(
            grid.get("feature_preprocessors"),
            [decoding.get("feature_preprocessor", preprocessing.get("feature_preprocessor", "pca"))],
        )
    ]
    pca_values = _list_value(grid.get("pca_components"), [decoding.get("pca_components", preprocessing.get("pca_components", 96))])
    feature_family_values = grid.get("feature_families", grid.get("feature_family"))
    feature_families = [normalize_source_feature_family(value) for value in _list_value(feature_family_values, ["bin_means"])]
    normalized_pca_values = [None if value in {None, "", "none", "None"} else normalize_pca_components(value) for value in pca_values]
    temporal_bins_values = [int(value) for value in _list_value(grid.get("temporal_bins"), [4])]
    c_grid = [float(value) for value in parse_c_grid(grid.get("c_grid", decoding.get("tuning_c_grid", "0.1,1.0,10.0")))]
    feature_kinds = [
        normalize_source_feature_kind(value)
        for value in _list_value(grid.get("feature_kinds"), [source_loso.get("feature_kind", "evoked")])
    ]
    xdawn_components_values = [
        int(value)
        for value in _list_value(grid.get("xdawn_components"), [source_loso.get("xdawn_components", DEFAULT_XDAWN_COMPONENTS)])
    ]
    covariance_max_channels_values = [int(value) for value in _list_value(grid.get("covariance_max_channels"), [DEFAULT_COVARIANCE_MAX_CHANNELS])]
    deep_weight_decay_grid = [float(value) for value in _list_value(grid.get("deep_weight_decay_grid"), [1e-4])]
    sample_weighting_values = [
        _normalize_sample_weighting(value)
        for value in _list_value(
            grid.get("sample_weightings", grid.get("sample_weighting", source_loso.get("sample_weighting", "none"))),
            ["none"],
        )
    ]
    class_bias_values = [
        _normalize_class_bias(value)
        for value in _list_value(
            grid.get("class_bias_modes", grid.get("class_bias", source_loso.get("class_bias", "none"))),
            ["none"],
        )
    ]

    candidates: list[CandidateSpec] = []
    for window_name, windows in window_sets:
        for feature_family in feature_families:
            for decoder in decoders:
                for emission_mode in emission_modes:
                    for feature_preprocessor in feature_preprocessors:
                        for pca_components in normalized_pca_values:
                            for temporal_bins in temporal_bins_values:
                                for feature_kind in feature_kinds:
                                    xdawn_component_grid = (
                                        xdawn_components_values
                                        if feature_kind in {"xdawn", "xdawn_prototype"}
                                        else [None]
                                    )
                                    covariance_feature_kind = (
                                        PROTOTYPE_BASE_FEATURE_KINDS.get(
                                            feature_kind, feature_kind
                                        )
                                    )
                                    covariance_channel_grid = (
                                        covariance_max_channels_values
                                        if covariance_feature_kind
                                        in {"covariance", "evoked_covariance"}
                                        else [DEFAULT_COVARIANCE_MAX_CHANNELS]
                                    )
                                    for xdawn_components in xdawn_component_grid:
                                        for covariance_max_channels in covariance_channel_grid:
                                            normalized_decoder = normalize_decoder_name(decoder)
                                            classifier_grid = (
                                                deep_weight_decay_grid
                                                if normalized_decoder == "torch_mlp"
                                                else c_grid
                                            )
                                            for sample_weighting in sample_weighting_values:
                                                for class_bias in class_bias_values:
                                                    for classifier_value in classifier_grid:
                                                        parameter_token = (
                                                            f"wd{classifier_value:g}"
                                                            if normalized_decoder == "torch_mlp"
                                                            else f"c{classifier_value:g}"
                                                        )
                                                        name = "__".join(
                                                            [
                                                                window_name,
                                                                feature_family,
                                                                normalized_decoder,
                                                                normalize_emission_mode(emission_mode),
                                                                normalize_feature_preprocessor(feature_preprocessor),
                                                                "pca"
                                                                + (
                                                                    "none"
                                                                    if pca_components is None
                                                                    else str(pca_components)
                                                                ),
                                                                f"bins{temporal_bins}",
                                                                f"feat{feature_kind}",
                                                                f"xdawn{'' if xdawn_components is None else xdawn_components}",
                                                                f"covch{covariance_max_channels}",
                                                                _normalize_sample_weighting(sample_weighting),
                                                                _normalize_class_bias(class_bias),
                                                                parameter_token,
                                                            ]
                                                        )
                                                        candidates.append(
                                                            CandidateSpec(
                                                                name=name,
                                                                decoder=decoder,
                                                                emission_mode=emission_mode,
                                                                feature_preprocessor=feature_preprocessor,
                                                                pca_components=pca_components,
                                                                classifier_param=classifier_value,
                                                                temporal_bins=temporal_bins,
                                                                windows=windows,
                                                                feature_kind=feature_kind,
                                                                xdawn_components=xdawn_components,
                                                                covariance_max_channels=covariance_max_channels,
                                                                feature_family=feature_family,
                                                                sample_weighting=sample_weighting,
                                                                class_bias=class_bias,
                                                            )
                                                        )
    return candidates


def _write_json_sidecar(path: Path, payload: Mapping[str, Any]) -> None:
    sidecar = Path(str(path) + ".provenance.json")
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_bushmeg_source_loso(
    config_path: str | Path,
    *,
    overrides: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    inner_cv_out_path: str | Path | None = None,
    predictions_out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run strict source-only nested LOSO decoding from a BUSH-MEG config."""

    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    source_loso = _section(config, "source_loso")
    selection_metric = str(source_loso.get("selection_metric", DEFAULT_SELECTION_METRIC))
    max_iter = int((_section(config, "decoding") or {}).get("max_iter", 1000))

    subjects, encoder = _load_subjects_from_config(config, config_dir=config_path.parent)
    candidates = _candidate_grid(config)
    cache = FeatureCache(subjects)
    cue_source_weights = resolve_cue_source_weights(config, config_dir=config_path.parent, known_subjects=subjects)
    n_classes = len(encoder.classes_)

    out = Path(out_path) if out_path is not None else _resolve_output(
        config,
        config_dir=config_path.parent,
        key="source_loso_summary_csv",
        default="source_loso_summary.csv",
    )
    inner_out = Path(inner_cv_out_path) if inner_cv_out_path is not None else _resolve_output(
        config,
        config_dir=config_path.parent,
        key="source_loso_inner_cv_csv",
        default="source_loso_inner_cv.csv",
    )
    predictions_out = Path(predictions_out_path) if predictions_out_path is not None else _resolve_output(
        config,
        config_dir=config_path.parent,
        key="source_loso_predictions_csv",
        default="source_loso_predictions.csv",
    )
    cue_weights_out = _resolve_output(
        config,
        config_dir=config_path.parent,
        key="source_loso_cue_weights_csv",
        default="source_loso_cue_source_weights.csv",
    )

    summary_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for outer_test_subject in sorted(subjects):
        selected, candidate_inner_rows, selected_summary = _select_candidate(
            subjects=subjects,
            cache=cache,
            candidates=candidates,
            outer_test_subject=outer_test_subject,
            n_classes=n_classes,
            max_iter=max_iter,
            selection_metric=selection_metric,
            cue_source_weights=cue_source_weights,
        )
        inner_rows.extend(candidate_inner_rows)
        train_subjects = [subject for subject in sorted(subjects) if subject != outer_test_subject]
        fold_subject_weights = None if cue_source_weights is None else cue_source_weights.for_fold(outer_test_subject, train_subjects)
        probabilities = _predict_candidate(
            subjects=subjects,
            cache=cache,
            candidate=selected,
            train_subjects=train_subjects,
            test_subject=outer_test_subject,
            n_classes=n_classes,
            max_iter=max_iter,
            subject_weight_multipliers=fold_subject_weights,
        )
        labels = subjects[outer_test_subject].labels
        predictions = probabilities.argmax(axis=1)
        summary_rows.append(
            {
                "outer_test_subject": outer_test_subject,
                **_candidate_rowspec(selected),
                **selected_summary,
                **_candidate_metrics(probabilities, labels, n_classes=n_classes),
                "n_train_subjects": len(train_subjects),
                "cue_source_weighting": "" if cue_source_weights is None else cue_source_weights.mode,
                "cue_source_weighting_blend": "" if cue_source_weights is None else cue_source_weights.blend,
                "cue_source_weights": "" if not fold_subject_weights else "|".join(f"{subject}:{weight:.8g}" for subject, weight in sorted(fold_subject_weights.items())),
                "n_test_trials": len(labels),
                "n_classes": n_classes,
                "class_names": "|".join(map(str, encoder.classes_)),
            }
        )
        metadata = subjects[outer_test_subject].metadata.reset_index(drop=True)
        for row_idx, (true_label, predicted_label) in enumerate(zip(labels, predictions, strict=True)):
            row: dict[str, Any] = {
                "outer_test_subject": outer_test_subject,
                "trial_index": int(row_idx),
                "candidate": selected.name,
                "true_label": int(true_label),
                "true_class": str(encoder.classes_[true_label]),
                "predicted_label": int(predicted_label),
                "predicted_class": str(encoder.classes_[predicted_label]),
                "probability_true_class": float(probabilities[row_idx, true_label]),
                "confidence": float(np.max(probabilities[row_idx])),
                "is_correct": bool(predicted_label == true_label),
            }
            for column in ("participant", "condition", "stimulus_class"):
                if column in metadata.columns:
                    row[column] = metadata.loc[row_idx, column]
            for class_idx, class_name in enumerate(encoder.classes_):
                row[f"class_{class_idx}"] = str(class_name)
                row[f"prob_class_{class_idx}"] = float(probabilities[row_idx, class_idx])
            prediction_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    inner = pd.DataFrame(inner_rows)
    predictions = pd.DataFrame(prediction_rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    inner_out.parent.mkdir(parents=True, exist_ok=True)
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    inner.to_csv(inner_out, index=False)
    predictions.to_csv(predictions_out, index=False)
    if cue_source_weights is not None:
        write_cue_source_weight_csv(cue_source_weights, sorted(subjects), cue_weights_out)
    _write_json_sidecar(
        out,
        {
            "config_path": str(config_path),
            "selection_metric": selection_metric,
            "n_subjects": len(subjects),
            "n_candidates": len(candidates),
            "feature_families": sorted({normalize_source_feature_family(candidate.feature_family) for candidate in candidates}),
            "normalization_scope": "subject_unlabeled_baseline",
            "epoch_normalization": _base.normalize_epoch_normalization(
                _preprocessing_normalization_name(_section(config, "preprocessing"))
            ),
            "sample_weighting_modes": sorted({candidate.sample_weighting for candidate in candidates}),
            "class_bias_modes": sorted({candidate.class_bias for candidate in candidates}),
            "cue_files_used": cue_source_weights is not None,
            "cue_files_used_for_classifier_training": False,
            "target_labels_used_for_selection": False,
            "cue_source_weighting_config": {} if cue_source_weights is None else dict(cue_source_weights.config or {}),
            "random_seed": DEFAULT_RANDOM_SEED,
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run strict cue-free source-only nested LOSO decoding for BUSH-MEG main-task FieldTrip MAT files."
    )
    parser.add_argument("config", type=Path, help="Dataset/workflow config, for example configs/bush_meg/source_loso_decoding.yml.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a dotted config key, e.g. --set source_loso.selection_metric=accuracy.")
    parser.add_argument("--out", type=Path, help="Summary CSV path. Defaults to outputs.source_loso_summary_csv.")
    parser.add_argument("--inner-cv-out", type=Path, help="Inner LOSO candidate-score CSV path.")
    parser.add_argument("--predictions-out", type=Path, help="Held-out trial probability CSV path.")
    args = parser.parse_args(argv)

    summary = run_bushmeg_source_loso(
        args.config,
        overrides=args.overrides,
        out_path=args.out,
        inner_cv_out_path=args.inner_cv_out,
        predictions_out_path=args.predictions_out,
    )
    mean_balanced = float(summary["balanced_accuracy"].mean())
    mean_top2 = float(summary["top2_accuracy"].mean())
    mean_top3 = float(summary["top3_accuracy"].mean())
    print(f"Wrote {len(summary)} LOSO rows")
    print(f"Mean balanced accuracy: {mean_balanced:.6f}")
    print(f"Mean top-2/top-3 accuracy: {mean_top2:.6f} / {mean_top3:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

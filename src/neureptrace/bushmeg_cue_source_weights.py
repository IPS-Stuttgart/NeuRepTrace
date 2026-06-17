"""Cue-derived source-subject weighting for strict BUSH-MEG source LOSO.

This module intentionally does *not* expose cue epochs as classifier training
or test examples.  Cue files are used only to estimate subject-level nuisance
similarity, then the main-task source trials are reweighted accordingly inside
the normal source-only LOSO decoder.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neureptrace import mne_time_decode as _base
from neureptrace.dataset_config import (
    _fieldtrip_file_specs,
    _validation_section,
    validate_dataset_config,
)
from neureptrace.io.fieldtrip_mat import load_fieldtrip_mat_epochs

DEFAULT_CUE_FEATURE_KINDS = ("baseline_logvar", "evoked_gfp", "evoked_mean")
DEFAULT_CUE_RESPONSE_WINDOW = (0.050, 0.250)
DEFAULT_CUE_TEMPORAL_BINS = 8
DEFAULT_CUE_MODE = "softmax_top_k"
DEFAULT_CUE_TEMPERATURE = 0.25
DEFAULT_CUE_TOP_K = 12
DEFAULT_CUE_BLEND = 0.50

CUE_SOURCE_WEIGHTING_MODES = {"uniform", "softmax", "top_k", "softmax_top_k"}
CUE_FEATURE_KINDS = {
    "baseline_logvar",
    "post_logvar",
    "evoked_mean",
    "evoked_gfp",
}


@dataclass(slots=True)
class CueSubjectData:
    """Cue epochs for one participant after subject-local normalization."""

    subject: str
    data: np.ndarray
    times: np.ndarray
    metadata: pd.DataFrame


@dataclass(slots=True)
class CueSourceWeights:
    """Fold-local source weights derived from cue subject-similarity vectors."""

    features: Mapping[str, np.ndarray]
    mode: str = DEFAULT_CUE_MODE
    temperature: float = DEFAULT_CUE_TEMPERATURE
    top_k: int | None = DEFAULT_CUE_TOP_K
    blend: float = DEFAULT_CUE_BLEND
    config: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        mode = normalize_cue_source_weighting_mode(self.mode)
        temperature = _positive_float(self.temperature, name="cue_source_weighting.temperature")
        blend = _unit_interval_float(self.blend, name="cue_source_weighting.blend")
        top_k = _normalize_optional_positive_integer(self.top_k, name="cue_source_weighting.top_k")
        normalized_features = {
            str(subject): _unit_feature_vector(vector)
            for subject, vector in self.features.items()
        }
        if not normalized_features:
            raise ValueError("Cue source weighting requires at least one cue subject feature vector.")
        self.features = normalized_features
        self.mode = mode
        self.temperature = temperature
        self.top_k = top_k
        self.blend = blend
        self.config = dict(self.config or {})

    def distance(self, target_subject: str, source_subject: str) -> float:
        """Return cosine distance between two cue feature vectors."""

        target = str(target_subject)
        source = str(source_subject)
        try:
            target_vector = self.features[target]
            source_vector = self.features[source]
        except KeyError as exc:
            raise ValueError(f"Missing cue calibration features for subject {exc.args[0]!r}.") from exc
        cosine = float(np.dot(target_vector, source_vector))
        return float(np.clip(1.0 - cosine, 0.0, 2.0))

    def for_fold(self, test_subject: str, train_subjects: Sequence[str]) -> dict[str, float]:
        """Return mean-one source-subject multipliers for one LOSO fold."""

        target = str(test_subject)
        sources = [str(subject) for subject in train_subjects if str(subject) != target]
        if not sources:
            return {}
        if self.mode == "uniform":
            return {source: 1.0 for source in sources}

        distances = np.asarray([self.distance(target, source) for source in sources], dtype=float)
        if not np.all(np.isfinite(distances)):
            raise ValueError(f"Non-finite cue distances for target subject {target!r}.")

        keep = np.ones(len(sources), dtype=bool)
        if self.top_k is not None and self.mode in {"top_k", "softmax_top_k"}:
            keep[:] = False
            keep[np.argsort(distances)[: min(int(self.top_k), len(sources))]] = True

        if self.mode == "top_k":
            raw = keep.astype(float)
        else:
            shifted = distances - float(np.min(distances))
            raw = np.exp(np.clip(-shifted / float(self.temperature), -60.0, 0.0))
            if self.mode == "softmax_top_k":
                raw = np.where(keep, raw, 0.0)

        if float(raw.mean()) <= 0.0 or not np.all(np.isfinite(raw)):
            raw = np.ones(len(sources), dtype=float)
        else:
            raw = raw / float(raw.mean())

        if self.blend < 1.0:
            raw = (1.0 - float(self.blend)) * np.ones_like(raw) + float(self.blend) * raw
        raw = raw / float(raw.mean())
        return {source: float(weight) for source, weight in zip(sources, raw, strict=True)}

    def rows(self, subjects: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return a long-form diagnostic table of all fold-local weights."""

        subject_ids = sorted(str(subject) for subject in (subjects or self.features.keys()))
        rows: list[dict[str, Any]] = []
        for target in subject_ids:
            train_subjects = [subject for subject in subject_ids if subject != target]
            weights = self.for_fold(target, train_subjects)
            for source in train_subjects:
                rows.append(
                    {
                        "outer_test_subject": target,
                        "source_subject": source,
                        "cue_distance": self.distance(target, source),
                        "cue_source_weight": weights[source],
                        "cue_weighting_mode": self.mode,
                        "cue_weighting_temperature": self.temperature,
                        "cue_weighting_top_k": "" if self.top_k is None else self.top_k,
                        "cue_weighting_blend": self.blend,
                    }
                )
        return rows


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


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _positive_float(value: Any, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return number


def _unit_interval_float(value: Any, *, name: str) -> float:
    number = _finite_float(value, name=name)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{name} must be in [0, 1].")
    return number


def _normalize_integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(number) or number % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    integer = int(number)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return integer


def _normalize_optional_positive_integer(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() in {"", "none", "None"}:
        return None
    return _normalize_integer(value, name=name, minimum=1)


def _normalize_temporal_bins(value: Any) -> int:
    return _normalize_integer(value, name="cue_source_weighting.temporal_bins", minimum=1)


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"", "0", "false", "no", "n", "off", "none", "null"}:
            return False
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        if int(value) in {0, 1}:
            return bool(value)
    raise ValueError(f"Cannot interpret {value!r} as a boolean flag.")


def _optional_float(value: Any, *, name: str) -> float | None:
    return None if value is None else _finite_float(value, name=name)


def normalize_cue_source_weighting_mode(value: Any) -> str:
    normalized = DEFAULT_CUE_MODE if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "nearest": "top_k",
        "nearest_sources": "top_k",
        "nearest_subjects": "top_k",
        "softmax_nearest": "softmax_top_k",
        "softmax_topk": "softmax_top_k",
        "exp": "softmax",
        "none": "uniform",
        "off": "uniform",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in CUE_SOURCE_WEIGHTING_MODES:
        raise ValueError(
            f"Unknown cue source-weighting mode {value!r}; choose one of {sorted(CUE_SOURCE_WEIGHTING_MODES)}."
        )
    return normalized


def normalize_cue_feature_kinds(value: Any) -> tuple[str, ...]:
    normalized: list[str] = []
    for item in _list_value(value, DEFAULT_CUE_FEATURE_KINDS):
        feature = str(item).strip().lower().replace("-", "_")
        aliases = {
            "baseline_var": "baseline_logvar",
            "baseline_log_variance": "baseline_logvar",
            "response_var": "post_logvar",
            "poststim_logvar": "post_logvar",
            "evoked": "evoked_mean",
            "evoked_bin_mean": "evoked_mean",
            "gfp": "evoked_gfp",
            "global_field_power": "evoked_gfp",
        }
        feature = aliases.get(feature, feature)
        if feature not in CUE_FEATURE_KINDS:
            raise ValueError(f"Unknown cue feature kind {item!r}; choose one of {sorted(CUE_FEATURE_KINDS)}.")
        normalized.append(feature)
    return tuple(dict.fromkeys(normalized))


def _window_tuple(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if value is None:
        items = list(default)
    elif isinstance(value, (str, bytes)):
        raise ValueError("Cue calibration windows must contain exactly [start, stop].")
    else:
        try:
            items = list(value)
        except TypeError as exc:
            raise ValueError("Cue calibration windows must contain exactly [start, stop].") from exc
    if len(items) != 2:
        raise ValueError("Cue calibration windows must contain exactly [start, stop].")
    start = _finite_float(items[0], name="cue_window_start")
    stop = _finite_float(items[1], name="cue_window_stop")
    if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
        raise ValueError("Cue calibration windows must be finite and have stop > start.")
    return start, stop


def _preprocessing_normalization_name(preprocessing: Mapping[str, Any]) -> str:
    if "normalization" in preprocessing:
        return str(preprocessing["normalization"])
    return str(preprocessing.get("epoch_normalization", "none"))


def _time_mask(times: np.ndarray, window: tuple[float, float], *, name: str) -> np.ndarray:
    tolerance = 1e-12
    mask = (times >= float(window[0]) - tolerance) & (times <= float(window[1]) + tolerance)
    if not np.any(mask):
        raise ValueError(f"{name} window {window} does not overlap cue epoch times [{times[0]}, {times[-1]}].")
    return mask


def _crop_data(data: np.ndarray, times: np.ndarray, *, tmin: float | None, tmax: float | None) -> tuple[np.ndarray, np.ndarray]:
    mask = np.ones(len(times), dtype=bool)
    tolerance = 1e-12
    tmin = _optional_float(tmin, name="cue_tmin")
    tmax = _optional_float(tmax, name="cue_tmax")
    if tmin is not None:
        mask &= times >= tmin - tolerance
    if tmax is not None:
        mask &= times <= tmax + tolerance
    if not np.any(mask):
        raise ValueError(f"Cue crop window [{tmin}, {tmax}] does not overlap the epoch time axis.")
    return data[:, :, mask], times[mask]


def _apply_subject_epoch_normalization(
    data: np.ndarray,
    times: np.ndarray,
    normalization: str,
    *,
    baseline_window: tuple[float, float],
) -> np.ndarray:
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
    raise ValueError(f"Unsupported cue normalization: {normalization}")


def _default_cue_participant_file(dataset: Mapping[str, Any]) -> str | None:
    for key in ("cue_participant_file", "cue_file_template"):
        value = dataset.get(key)
        if value:
            return str(value)
    template = dataset.get("participant_file") or dataset.get("file_template")
    if not template:
        return None
    text = str(template)
    candidate = text.replace("Data.mat", "CueData.mat")
    return candidate if candidate != text else None


def _cue_dataset_config(config: Mapping[str, Any], cue_config: Mapping[str, Any]) -> dict[str, Any]:
    updated = copy.deepcopy(dict(config))
    dataset = updated.setdefault("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("Config section 'dataset' must be a mapping.")
    if "files" in cue_config:
        dataset["files"] = cue_config["files"]
        for key in ("participant_file", "file_template", "file_templates", "participant_files"):
            dataset.pop(key, None)
    else:
        participant_file = cue_config.get("participant_file") or cue_config.get("file_template") or _default_cue_participant_file(dataset)
        if not participant_file:
            raise ValueError(
                "source_loso.cue_source_weighting is enabled, but no cue participant_file was configured. "
                "Set source_loso.cue_source_weighting.participant_file, e.g. 'Part{participant}CueData.mat'."
            )
        for key in ("files", "file_templates", "participant_files"):
            dataset.pop(key, None)
        dataset["participant_file"] = str(participant_file)
    return updated


def _load_cue_subjects_from_config(config: Mapping[str, Any], *, config_dir: Path, cue_config: Mapping[str, Any]) -> dict[str, CueSubjectData]:
    cue_dataset_config = _cue_dataset_config(config, cue_config)
    validate_dataset_config(cue_dataset_config, base_dir=config_dir, check_files=True)
    dataset = _section(cue_dataset_config, "dataset")
    metadata_config = _section(cue_dataset_config, "metadata")
    preprocessing = _section(cue_dataset_config, "preprocessing")
    matlab_config = _section(cue_dataset_config, "matlab")
    decoding = _section(cue_dataset_config, "decoding") or _section(cue_dataset_config, "workflow")
    source_loso = _section(cue_dataset_config, "source_loso")
    group_column = str(source_loso.get("group_column", decoding.get("group_column", "participant")))
    baseline_window = _window_tuple(cue_config.get("baseline_window", preprocessing.get("baseline_window", _base.DEFAULT_BASELINE_WINDOW)), _base.DEFAULT_BASELINE_WINDOW)
    normalization = str(cue_config.get("normalization", _preprocessing_normalization_name(preprocessing)))
    tmin = cue_config.get("tmin", preprocessing.get("tmin"))
    tmax = cue_config.get("tmax", preprocessing.get("tmax"))

    loader_config: dict[str, Any] = {**matlab_config, **dataset, "metadata": metadata_config}
    if "validation" in cue_dataset_config:
        loader_config["validation"] = _validation_section(cue_dataset_config)

    loaded: dict[str, CueSubjectData] = {}
    for path, extra_metadata in _fieldtrip_file_specs(cue_dataset_config, base_dir=config_dir):
        dataset_epochs = load_fieldtrip_mat_epochs(path, loader_config, extra_metadata=extra_metadata)
        metadata = dataset_epochs.metadata.reset_index(drop=True).copy()
        if group_column not in metadata.columns:
            metadata[group_column] = str(extra_metadata.get("participant", path.stem))
        subject_values = pd.unique(metadata[group_column].astype(str))
        if len(subject_values) != 1:
            raise ValueError(f"Expected one {group_column} value per cue file; found {subject_values.tolist()} in {path}.")
        subject = str(subject_values[0])
        if subject in loaded:
            raise ValueError(f"Duplicate cue calibration file for subject {subject!r}.")
        data, times = _crop_data(dataset_epochs.data, dataset_epochs.times, tmin=tmin, tmax=tmax)
        normalized = _apply_subject_epoch_normalization(
            data,
            times,
            normalization,
            baseline_window=baseline_window,
        )
        loaded[subject] = CueSubjectData(
            subject=subject,
            data=normalized,
            times=times.astype(float, copy=True),
            metadata=metadata,
        )
    return loaded


def _channel_logvar(data: np.ndarray, times: np.ndarray, window: tuple[float, float], *, name: str, epsilon: float = 1e-12) -> np.ndarray:
    mask = _time_mask(times, window, name=name)
    epsilon = _positive_float(epsilon, name="cue_feature_epsilon")
    variance = np.var(data[:, :, mask], axis=(0, 2), ddof=1 if data.shape[0] * int(mask.sum()) > 1 else 0)
    return np.log(np.maximum(variance, epsilon))


def _evoked_bin_means(data: np.ndarray, times: np.ndarray, window: tuple[float, float], *, temporal_bins: int) -> np.ndarray:
    temporal_bins = _normalize_temporal_bins(temporal_bins)
    mask = _time_mask(times, window, name="cue response")
    evoked = np.asarray(data[:, :, mask], dtype=np.float64).mean(axis=0)
    bins = np.array_split(np.arange(evoked.shape[1]), temporal_bins)
    return np.concatenate([evoked[:, bin_indices].mean(axis=1) for bin_indices in bins], axis=0)


def _evoked_gfp_bins(data: np.ndarray, times: np.ndarray, window: tuple[float, float], *, temporal_bins: int) -> np.ndarray:
    temporal_bins = _normalize_temporal_bins(temporal_bins)
    mask = _time_mask(times, window, name="cue response")
    evoked = np.asarray(data[:, :, mask], dtype=np.float64).mean(axis=0)
    gfp = np.sqrt(np.mean(evoked * evoked, axis=0))
    bins = np.array_split(np.arange(gfp.shape[0]), temporal_bins)
    return np.asarray([gfp[bin_indices].mean() for bin_indices in bins], dtype=np.float64)


def cue_subject_feature_vector(
    subject: CueSubjectData,
    *,
    feature_kinds: Sequence[str] = DEFAULT_CUE_FEATURE_KINDS,
    baseline_window: tuple[float, float] = _base.DEFAULT_BASELINE_WINDOW,
    response_window: tuple[float, float] = DEFAULT_CUE_RESPONSE_WINDOW,
    temporal_bins: int = DEFAULT_CUE_TEMPORAL_BINS,
) -> np.ndarray:
    """Return one subject-level cue calibration vector."""

    temporal_bins = _normalize_temporal_bins(temporal_bins)
    parts: list[np.ndarray] = []
    for feature_kind in normalize_cue_feature_kinds(feature_kinds):
        if feature_kind == "baseline_logvar":
            parts.append(_channel_logvar(subject.data, subject.times, baseline_window, name="cue baseline"))
        elif feature_kind == "post_logvar":
            parts.append(_channel_logvar(subject.data, subject.times, response_window, name="cue response"))
        elif feature_kind == "evoked_mean":
            parts.append(_evoked_bin_means(subject.data, subject.times, response_window, temporal_bins=temporal_bins))
        elif feature_kind == "evoked_gfp":
            parts.append(_evoked_gfp_bins(subject.data, subject.times, response_window, temporal_bins=temporal_bins))
        else:  # pragma: no cover - guarded by normalize_cue_feature_kinds
            raise ValueError(f"Unsupported cue feature kind: {feature_kind}")
    vector = np.concatenate([np.ravel(part).astype(np.float64, copy=False) for part in parts], axis=0)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


def _unit_feature_vector(vector: np.ndarray, *, epsilon: float = 1e-12) -> np.ndarray:
    epsilon = _positive_float(epsilon, name="cue_feature_epsilon")
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vector.size == 0:
        raise ValueError("Cue feature vector must not be empty.")
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    vector = vector - float(vector.mean())
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > epsilon else np.zeros_like(vector)


def resolve_cue_source_weights(
    config: Mapping[str, Any],
    *,
    config_dir: Path,
    known_subjects: Mapping[str, Any] | Sequence[str],
) -> CueSourceWeights | None:
    """Build cue-derived source weights when enabled in the config."""

    source_loso = _section(config, "source_loso")
    raw = source_loso.get("cue_source_weighting", {}) or {}
    if isinstance(raw, bool):
        raw = {"enabled": raw}
    if not isinstance(raw, Mapping):
        raise ValueError("source_loso.cue_source_weighting must be a mapping or boolean.")
    if not _truthy(raw.get("enabled", False)):
        return None

    cue_config = dict(raw)
    cue_subjects = _load_cue_subjects_from_config(config, config_dir=config_dir, cue_config=cue_config)
    expected_subjects = {str(subject) for subject in (known_subjects.keys() if isinstance(known_subjects, Mapping) else known_subjects)}
    missing = sorted(expected_subjects.difference(cue_subjects))
    if missing:
        raise ValueError(f"Cue source weighting is enabled, but cue calibration files are missing for subjects: {missing}.")

    preprocessing = _section(config, "preprocessing")
    baseline_window = _window_tuple(raw.get("baseline_window", preprocessing.get("baseline_window", _base.DEFAULT_BASELINE_WINDOW)), _base.DEFAULT_BASELINE_WINDOW)
    response_window = _window_tuple(raw.get("response_window", DEFAULT_CUE_RESPONSE_WINDOW), DEFAULT_CUE_RESPONSE_WINDOW)
    temporal_bins = _normalize_temporal_bins(raw.get("temporal_bins", DEFAULT_CUE_TEMPORAL_BINS))
    feature_kinds = normalize_cue_feature_kinds(raw.get("feature_kinds", DEFAULT_CUE_FEATURE_KINDS))
    features = {
        subject: cue_subject_feature_vector(
            cue_subjects[subject],
            feature_kinds=feature_kinds,
            baseline_window=baseline_window,
            response_window=response_window,
            temporal_bins=temporal_bins,
        )
        for subject in sorted(expected_subjects)
    }
    return CueSourceWeights(
        features=features,
        mode=normalize_cue_source_weighting_mode(raw.get("mode", DEFAULT_CUE_MODE)),
        temperature=_positive_float(raw.get("temperature", DEFAULT_CUE_TEMPERATURE), name="cue_source_weighting.temperature"),
        top_k=raw.get("top_k", DEFAULT_CUE_TOP_K),
        blend=_unit_interval_float(raw.get("blend", DEFAULT_CUE_BLEND), name="cue_source_weighting.blend"),
        config={
            "participant_file": raw.get("participant_file"),
            "feature_kinds": feature_kinds,
            "baseline_window": baseline_window,
            "response_window": response_window,
            "temporal_bins": temporal_bins,
            "mode": raw.get("mode", DEFAULT_CUE_MODE),
            "temperature": _positive_float(raw.get("temperature", DEFAULT_CUE_TEMPERATURE), name="cue_source_weighting.temperature"),
            "top_k": raw.get("top_k", DEFAULT_CUE_TOP_K),
            "blend": _unit_interval_float(raw.get("blend", DEFAULT_CUE_BLEND), name="cue_source_weighting.blend"),
        },
    )


def write_cue_source_weight_csv(cue_source_weights: CueSourceWeights, subjects: Sequence[str], path: str | Path) -> None:
    """Write fold-local cue source weights for provenance/debugging."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cue_source_weights.rows(subjects)).to_csv(output, index=False)

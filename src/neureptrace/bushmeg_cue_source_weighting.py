"""Cue-task summaries for target-specific BUSH-MEG source weighting.

The cue files are used only as calibration data: they produce one summary vector
per subject, and those summaries weight source-subject training rows for a held
out subject. No cue trials are used as classifier training/testing examples.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from neureptrace.bushmeg_source_loso import (
    SubjectEpochs,
    WindowSpec,
    _load_subjects_from_config,
    _sample_indices_for_window,
    _section,
)

DEFAULT_CUE_SOURCE_WEIGHTING = "none"
DEFAULT_CUE_SOURCE_TEMPERATURE = 0.25
DEFAULT_CUE_SUMMARY_WINDOW = (-0.05, 0.25)
DEFAULT_CUE_BASELINE_WINDOW = (-0.35, -0.05)
CUE_SOURCE_WEIGHTING_MODES = {"none", "cue_evoked_correlation", "cue_covariance_correlation", "cue_hybrid_correlation"}


def normalize_cue_source_weighting(value: Any) -> str:
    mode = DEFAULT_CUE_SOURCE_WEIGHTING if value is None else str(value).strip().lower().replace("-", "_")
    mode = {
        "": "none",
        "false": "none",
        "off": "none",
        "no": "none",
        "cue": "cue_evoked_correlation",
        "cue_similarity": "cue_evoked_correlation",
        "cue_evoked": "cue_evoked_correlation",
        "evoked": "cue_evoked_correlation",
        "cue_covariance": "cue_covariance_correlation",
        "covariance": "cue_covariance_correlation",
        "cue_hybrid": "cue_hybrid_correlation",
        "hybrid": "cue_hybrid_correlation",
    }.get(mode, mode)
    if mode not in CUE_SOURCE_WEIGHTING_MODES:
        raise ValueError(f"Unknown cue source-weighting mode {value!r}; choose one of {sorted(CUE_SOURCE_WEIGHTING_MODES)}.")
    return mode


def normalize_cue_source_temperature(value: Any, cue_source_weighting: str) -> float | None:
    if cue_source_weighting == "none":
        return None
    temperature = DEFAULT_CUE_SOURCE_TEMPERATURE if value is None else float(value)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("source_loso.cue_source_temperature must be positive and finite.")
    return temperature


def cue_window_tuple(value: Any, default: tuple[float, float], *, key: str) -> tuple[float, float]:
    if value is None:
        return tuple(float(part) for part in default)
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"source_loso.{key} must contain exactly two numeric values.")
    window = (float(parts[0]), float(parts[1]))
    if not np.isfinite(window[0]) or not np.isfinite(window[1]) or window[0] >= window[1]:
        raise ValueError(f"source_loso.{key} must be a finite [start, stop] window with start < stop.")
    return window


def _unit_centered_vector(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    vector = vector - float(vector.mean())
    norm = float(np.linalg.norm(vector))
    return vector / (norm if norm > 1e-12 else 1.0)


def _window_from_tuple(window: tuple[float, float]) -> WindowSpec:
    return WindowSpec(center=float(np.mean(window)), width=float(window[1] - window[0]))


def cue_evoked_summary(subject: SubjectEpochs, *, summary_window: tuple[float, float] = DEFAULT_CUE_SUMMARY_WINDOW) -> np.ndarray:
    indices = _sample_indices_for_window(subject.times, _window_from_tuple(summary_window))
    return np.asarray(subject.data[:, :, indices], dtype=np.float64).mean(axis=0).reshape(-1)


def cue_covariance_summary(subject: SubjectEpochs, *, baseline_window: tuple[float, float] = DEFAULT_CUE_BASELINE_WINDOW) -> np.ndarray:
    indices = _sample_indices_for_window(subject.times, _window_from_tuple(baseline_window))
    baseline = np.asarray(subject.data[:, :, indices], dtype=np.float64)
    n_channels = baseline.shape[1]
    flattened = np.transpose(baseline, (1, 0, 2)).reshape(n_channels, -1)
    flattened -= flattened.mean(axis=1, keepdims=True)
    covariance = (flattened @ flattened.T) / float(max(flattened.shape[1] - 1, 1))
    trace = float(np.trace(covariance))
    if np.isfinite(trace) and trace > 0.0:
        covariance /= trace
    return covariance[np.triu_indices(n_channels)]


def cue_summary_vector(
    subject: SubjectEpochs,
    *,
    mode: str,
    summary_window: tuple[float, float] = DEFAULT_CUE_SUMMARY_WINDOW,
    baseline_window: tuple[float, float] = DEFAULT_CUE_BASELINE_WINDOW,
) -> np.ndarray:
    mode = normalize_cue_source_weighting(mode)
    if mode == "cue_evoked_correlation":
        return _unit_centered_vector(cue_evoked_summary(subject, summary_window=summary_window))
    if mode == "cue_covariance_correlation":
        return _unit_centered_vector(cue_covariance_summary(subject, baseline_window=baseline_window))
    if mode == "cue_hybrid_correlation":
        return np.concatenate(
            [
                _unit_centered_vector(cue_evoked_summary(subject, summary_window=summary_window)),
                _unit_centered_vector(cue_covariance_summary(subject, baseline_window=baseline_window)),
            ]
        )
    raise ValueError("Cue summaries are only defined for non-'none' cue weighting modes.")


def cue_subject_file_pattern(config: Mapping[str, Any]) -> str:
    source_loso = _section(config, "source_loso")
    if source_loso.get("cue_participant_file"):
        return str(source_loso["cue_participant_file"])
    participant_file = str(_section(config, "dataset").get("participant_file", "Part{participant}Data.mat"))
    if "CueData" in participant_file:
        return participant_file
    if "Data.mat" in participant_file:
        return participant_file.replace("Data.mat", "CueData.mat")
    raise ValueError("Set source_loso.cue_participant_file because dataset.participant_file could not be converted to a cue pattern.")


def load_cue_summaries_from_config(config: Mapping[str, Any], *, config_dir: Path, mode: str) -> dict[str, np.ndarray]:
    mode = normalize_cue_source_weighting(mode)
    if mode == "none":
        return {}
    source_loso = _section(config, "source_loso")
    summary_window = cue_window_tuple(source_loso.get("cue_summary_window"), DEFAULT_CUE_SUMMARY_WINDOW, key="cue_summary_window")
    baseline_window = cue_window_tuple(source_loso.get("cue_baseline_window"), DEFAULT_CUE_BASELINE_WINDOW, key="cue_baseline_window")
    cue_config = deepcopy(dict(config))
    cue_config["dataset"] = dict(_section(config, "dataset"))
    cue_config["dataset"]["participant_file"] = cue_subject_file_pattern(config)
    cue_subjects, _ = _load_subjects_from_config(cue_config, config_dir=config_dir)
    return {
        subject_id: cue_summary_vector(subject, mode=mode, summary_window=summary_window, baseline_window=baseline_window)
        for subject_id, subject in cue_subjects.items()
    }


def cue_source_weights_from_summaries(
    cue_summaries: Mapping[str, np.ndarray],
    *,
    test_subject: str,
    train_subjects: Sequence[str],
    mode: str,
    temperature: float | None,
) -> dict[str, float] | None:
    mode = normalize_cue_source_weighting(mode)
    if mode == "none":
        return None
    if test_subject not in cue_summaries:
        raise ValueError(f"Cue summary for test subject {test_subject!r} is missing.")
    missing = [subject for subject in train_subjects if subject not in cue_summaries]
    if missing:
        raise ValueError(f"Cue summaries for source subjects are missing: {missing}.")
    temperature = normalize_cue_source_temperature(temperature, mode)
    target = _unit_centered_vector(cue_summaries[test_subject])
    distances = np.asarray([1.0 - float(np.clip(np.dot(target, _unit_centered_vector(cue_summaries[subject])), -1.0, 1.0)) for subject in train_subjects], dtype=float)
    scores = np.exp(np.clip(-(distances - float(np.min(distances))) / float(temperature), -60.0, 0.0))
    scores /= float(scores.mean())
    return {str(subject): float(weight) for subject, weight in zip(train_subjects, scores, strict=True)}

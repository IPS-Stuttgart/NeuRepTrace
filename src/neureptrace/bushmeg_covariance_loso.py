"""Strict source-only BUSH-MEG covariance-feature LOSO decoding.

This module carries the reusable covariance-feature benchmark from the
deprecated PyMEGDec project into NeuRepTrace.  It keeps the same leakage
discipline as the other BUSH-MEG workflows: only main-task FieldTrip MATLAB
files are loaded, candidate selection is performed by source-subject inner LOSO,
and cue/localizer files are not touched.

The feature extractor supports the PyMEGDec covariance variants:

* log-Euclidean covariance features;
* upper-triangular covariance features;
* upper-triangular correlation features; and
* log-variance diagonal features.

All classifier fitting and optional label-shuffle null controls happen inside
the source-subject training folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from neureptrace import mne_time_decode as _base
from neureptrace.bushmeg_source_loso import (
    DEFAULT_RANDOM_SEED,
    DEFAULT_SELECTION_METRIC,
    MINIMIZE_SELECTION_METRICS,
    SUPPORTED_SELECTION_METRICS,
    SubjectEpochs,
    _candidate_metrics,
    _load_subjects_from_config,
    _resolve_output,
    _section,
    _write_json_sidecar,
)
from neureptrace.dataset_config import apply_overrides, load_config
from neureptrace.decoding import (
    make_decoder,
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    normalize_pca_components,
    parse_c_grid,
    predict_emission_probabilities,
)

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
    """One absolute-time covariance feature window."""

    name: str
    start: float
    stop: float


@dataclass(frozen=True, slots=True)
class CovarianceCandidateSpec:
    """One covariance-feature source-only decoder candidate."""

    name: str
    decoder: str
    emission_mode: str
    feature_preprocessor: str
    pca_components: int | float | None
    classifier_param: float | None
    window: CovarianceWindow
    covariance_feature_mode: str = DEFAULT_COVARIANCE_FEATURE_MODE
    covariance_shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE
    covariance_epsilon: float = DEFAULT_COVARIANCE_EPSILON
    covariance_max_channels: int = DEFAULT_COVARIANCE_MAX_CHANNELS


class CovarianceFeatureCache:
    """Per-subject cache for covariance-window feature matrices."""

    def __init__(self, subjects: Mapping[str, SubjectEpochs]):
        self._subjects = dict(subjects)
        self._cache: dict[tuple[str, CovarianceWindow, str, float, float, int], np.ndarray] = {}

    def get(self, subject: str, candidate: CovarianceCandidateSpec) -> np.ndarray:
        key = (
            str(subject),
            candidate.window,
            normalize_covariance_feature_mode(candidate.covariance_feature_mode),
            float(candidate.covariance_shrinkage),
            float(candidate.covariance_epsilon),
            int(candidate.covariance_max_channels),
        )
        if key not in self._cache:
            subject_epochs = self._subjects[subject]
            self._cache[key] = _window_covariance_features(
                subject_epochs.data,
                subject_epochs.times,
                candidate.window,
                mode=candidate.covariance_feature_mode,
                shrinkage=candidate.covariance_shrinkage,
                epsilon=candidate.covariance_epsilon,
                covariance_max_channels=candidate.covariance_max_channels,
            )
        return self._cache[key]


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


def _normalize_covariance_shrinkage(value: Any) -> float:
    shrinkage = float(value)
    if not np.isfinite(shrinkage) or not 0.0 <= shrinkage <= 1.0:
        raise ValueError("covariance_shrinkage must be a finite value in [0, 1].")
    return shrinkage


def _normalize_covariance_epsilon(value: Any) -> float:
    epsilon = float(value)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("covariance_epsilon must be a positive finite value.")
    return epsilon


def _sample_indices_for_window(times: np.ndarray, window: CovarianceWindow) -> np.ndarray:
    times = np.asarray(times, dtype=float).reshape(-1)
    tolerance = 1e-12
    indices = np.flatnonzero((times >= float(window.start) - tolerance) & (times <= float(window.stop) + tolerance))
    if indices.size == 0:
        raise ValueError(
            f"Covariance window '{window.name}' [{window.start:.6g}, {window.stop:.6g}] "
            f"does not overlap available times [{times[0]:.6g}, {times[-1]:.6g}]."
        )
    return indices


def _channel_subset_indices(n_channels: int, max_channels: int) -> np.ndarray:
    max_channels = max(1, int(max_channels))
    if int(n_channels) <= max_channels:
        return np.arange(int(n_channels), dtype=int)
    return np.unique(np.linspace(0, int(n_channels) - 1, max_channels, dtype=int))


def _trial_covariance(signal: np.ndarray, *, shrinkage: float, epsilon: float) -> np.ndarray:
    """Return a shrunken SPD covariance matrix for one channels x time trial."""

    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2:
        raise ValueError("Trial signal must be a channels x time matrix.")
    centered = signal - np.mean(signal, axis=1, keepdims=True)
    denominator = max(1, int(centered.shape[1]) - 1)
    covariance = (centered @ centered.T) / float(denominator)
    covariance = 0.5 * (covariance + covariance.T)
    n_channels = covariance.shape[0]
    trace_mean = float(np.trace(covariance) / max(1, n_channels))
    if not np.isfinite(trace_mean) or trace_mean <= 0.0:
        trace_mean = 1.0
    shrinkage = _normalize_covariance_shrinkage(shrinkage)
    covariance = (1.0 - shrinkage) * covariance + shrinkage * trace_mean * np.eye(n_channels)
    covariance = covariance + _normalize_covariance_epsilon(epsilon) * trace_mean * np.eye(n_channels)
    return 0.5 * (covariance + covariance.T)


def _eigen_floor(matrix: np.ndarray, epsilon: float) -> float:
    scale = float(np.trace(matrix) / max(1, matrix.shape[0]))
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return _normalize_covariance_epsilon(epsilon) * scale


def _matrix_log_spd(matrix: np.ndarray, epsilon: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    floor = _eigen_floor(matrix, epsilon)
    log_values = np.log(np.maximum(eigenvalues, floor))
    return (eigenvectors * log_values[None, :]) @ eigenvectors.T


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    std = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
    correlation = covariance / np.outer(std, std)
    np.fill_diagonal(correlation, 1.0)
    return 0.5 * (correlation + correlation.T)


def _vectorize_symmetric(matrix: np.ndarray, *, scale_off_diagonal: bool = True) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    rows, cols = np.triu_indices(matrix.shape[0])
    values = np.asarray(matrix[rows, cols], dtype=float)
    if scale_off_diagonal:
        values = values.copy()
        values[rows != cols] *= np.sqrt(2.0)
    return values


def covariance_feature_vector(
    signal: np.ndarray,
    mode: str = DEFAULT_COVARIANCE_FEATURE_MODE,
    *,
    shrinkage: float = DEFAULT_COVARIANCE_SHRINKAGE,
    epsilon: float = DEFAULT_COVARIANCE_EPSILON,
) -> np.ndarray:
    """Return one PyMEGDec-compatible covariance feature vector."""

    mode = normalize_covariance_feature_mode(mode)
    covariance = _trial_covariance(signal, shrinkage=shrinkage, epsilon=epsilon)
    if mode == "variance":
        return np.log(np.maximum(np.diag(covariance), _eigen_floor(covariance, epsilon)))
    if mode == "correlation_upper":
        return _vectorize_symmetric(_covariance_to_correlation(covariance), scale_off_diagonal=True)
    if mode == "covariance_upper":
        return _vectorize_symmetric(covariance, scale_off_diagonal=True)
    if mode == "logeuclidean_covariance":
        return _vectorize_symmetric(_matrix_log_spd(covariance, epsilon), scale_off_diagonal=True)
    raise ValueError(f"Unsupported covariance feature mode: {mode}")  # pragma: no cover - guarded above


def _window_covariance_features(
    data: np.ndarray,
    times: np.ndarray,
    window: CovarianceWindow,
    *,
    mode: str,
    shrinkage: float,
    epsilon: float,
    covariance_max_channels: int,
) -> np.ndarray:
    """Return trial x feature covariance representations for one time window."""

    data = np.asarray(data, dtype=float)
    if data.ndim != 3:
        raise ValueError("BUSH-MEG covariance features expect data shaped trials x channels x time.")
    time_indices = _sample_indices_for_window(times, window)
    channel_indices = _channel_subset_indices(data.shape[1], covariance_max_channels)
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


def _as_list(value: Any, default: Sequence[Any]) -> list[Any]:
    if value is None:
        return list(default)
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _as_bool(value: Any, *, default: bool = False) -> bool:
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


def _float_grid(value: Any, default: Sequence[float]) -> list[float]:
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        values = [float(token) for token in tokens] if tokens else list(default)
    else:
        values = [float(item) for item in _as_list(value, default)]
    if not values or not np.all(np.isfinite(values)):
        raise ValueError("Expected at least one finite numeric grid value.")
    return values


def _normalize_pca_value(value: Any) -> int | float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return normalize_pca_components(value)


def _time_window_from_mapping(item: Mapping[str, Any], *, index: int) -> CovarianceWindow:
    if "range" in item:
        raw_range = item["range"]
        if not isinstance(raw_range, Sequence) or isinstance(raw_range, (str, bytes)) or len(raw_range) != 2:
            raise ValueError("Covariance window range must contain exactly two values.")
        start, stop = map(float, raw_range)
    else:
        start = float(item.get("start", item.get("tmin")))
        stop = float(item.get("stop", item.get("tmax")))
    if not np.all(np.isfinite([start, stop])) or stop <= start:
        raise ValueError("Covariance window stop must be finite and greater than start.")
    name = str(item.get("name", f"cov_{index:02d}_{start:g}_{stop:g}"))
    return CovarianceWindow(name=name, start=start, stop=stop)


def _candidate_windows(grid: Mapping[str, Any]) -> list[CovarianceWindow]:
    raw_windows = grid.get("time_windows", grid.get("windows"))
    if raw_windows is None:
        raw_windows = [{"name": "cov_050_300ms", "start": 0.05, "stop": 0.30}]
    if isinstance(raw_windows, str):
        items: list[Mapping[str, Any]] = []
        for index, token in enumerate(token.strip() for token in raw_windows.split(",") if token.strip()):
            start_text, stop_text = token.split(":", maxsplit=1)
            items.append({"name": f"cov_{index:02d}", "start": float(start_text), "stop": float(stop_text)})
        raw_windows = items
    windows = []
    for index, item in enumerate(_as_list(raw_windows, [])):
        if not isinstance(item, Mapping):
            raise ValueError("covariance_loso.candidate_grid.time_windows entries must be mappings or start:stop strings.")
        windows.append(_time_window_from_mapping(item, index=index))
    if not windows:
        raise ValueError("At least one covariance window is required.")
    return windows


def _candidate_grid(config: Mapping[str, Any]) -> list[CovarianceCandidateSpec]:
    decoding = _section(config, "decoding") or _section(config, "workflow")
    covariance_loso = _section(config, "covariance_loso")
    source_loso = _section(config, "source_loso")
    grid = covariance_loso.get("candidate_grid", {}) or {}
    if not isinstance(grid, Mapping):
        raise ValueError("covariance_loso.candidate_grid must be a mapping.")

    windows = _candidate_windows(grid)
    decoders = [str(value) for value in _as_list(grid.get("decoders"), [decoding.get("decoder", decoding.get("classifier", "multinomial-logistic"))])]
    emission_modes = [str(value) for value in _as_list(grid.get("emission_modes"), [decoding.get("emission_mode", "uncalibrated")])]
    feature_preprocessors = [
        str(value)
        for value in _as_list(
            grid.get("feature_preprocessors"),
            [decoding.get("feature_preprocessor", "pca")],
        )
    ]
    pca_values = [_normalize_pca_value(value) for value in _as_list(grid.get("pca_components"), [decoding.get("pca_components", 64)])]
    classifier_grid = [float(value) for value in parse_c_grid(grid.get("c_grid", decoding.get("tuning_c_grid", "0.1,1.0,10.0")))]
    feature_modes = [normalize_covariance_feature_mode(value) for value in _as_list(grid.get("feature_modes", grid.get("covariance_feature_modes")), [covariance_loso.get("feature_mode", DEFAULT_COVARIANCE_FEATURE_MODE)])]
    shrinkages = [_normalize_covariance_shrinkage(value) for value in _float_grid(grid.get("covariance_shrinkages", grid.get("shrinkages")), [covariance_loso.get("covariance_shrinkage", DEFAULT_COVARIANCE_SHRINKAGE)])]
    epsilons = [_normalize_covariance_epsilon(value) for value in _float_grid(grid.get("covariance_epsilons", grid.get("epsilons")), [covariance_loso.get("covariance_epsilon", DEFAULT_COVARIANCE_EPSILON)])]
    max_channels_values = [int(value) for value in _as_list(grid.get("covariance_max_channels"), [covariance_loso.get("covariance_max_channels", DEFAULT_COVARIANCE_MAX_CHANNELS)])]

    candidates: list[CovarianceCandidateSpec] = []
    for window in windows:
        for mode in feature_modes:
            for shrinkage in shrinkages:
                for epsilon in epsilons:
                    for max_channels in max_channels_values:
                        for decoder in decoders:
                            for emission_mode in emission_modes:
                                for feature_preprocessor in feature_preprocessors:
                                    for pca_components in pca_values:
                                        for classifier_param in classifier_grid:
                                            normalized_decoder = normalize_decoder_name(decoder)
                                            name = "__".join(
                                                [
                                                    window.name,
                                                    mode,
                                                    f"shrink{shrinkage:g}",
                                                    f"eps{epsilon:g}",
                                                    f"covch{max_channels}",
                                                    normalized_decoder,
                                                    normalize_emission_mode(emission_mode),
                                                    normalize_feature_preprocessor(feature_preprocessor),
                                                    "pca" + ("none" if pca_components is None else str(pca_components)),
                                                    f"c{classifier_param:g}",
                                                ]
                                            )
                                            candidates.append(
                                                CovarianceCandidateSpec(
                                                    name=name,
                                                    decoder=decoder,
                                                    emission_mode=emission_mode,
                                                    feature_preprocessor=feature_preprocessor,
                                                    pca_components=pca_components,
                                                    classifier_param=classifier_param,
                                                    window=window,
                                                    covariance_feature_mode=mode,
                                                    covariance_shrinkage=shrinkage,
                                                    covariance_epsilon=epsilon,
                                                    covariance_max_channels=max_channels,
                                                )
                                            )
    if not candidates:
        raise ValueError("No covariance LOSO candidates were configured.")
    # Touch source_loso so configs that keep the section for related workflows still
    # fail early if it is malformed.
    _ = source_loso
    return candidates


def _effective_pca_components(candidate: CovarianceCandidateSpec, n_features: int, n_samples: int) -> int | float | None:
    pca_components = candidate.pca_components
    if pca_components is None:
        return None
    if normalize_feature_preprocessor(candidate.feature_preprocessor) not in {"pca", "pca_whiten"}:
        return pca_components
    if isinstance(pca_components, (int, np.integer)):
        return min(int(pca_components), max(1, min(int(n_features), int(n_samples))))
    return pca_components


def _make_model(candidate: CovarianceCandidateSpec, *, n_features: int, n_samples: int, max_iter: int):
    return make_decoder(
        candidate.decoder,
        max_iter=max_iter,
        emission_mode=candidate.emission_mode,
        feature_preprocessor=candidate.feature_preprocessor,
        pca_components=_effective_pca_components(candidate, n_features, n_samples),
        classifier_param=candidate.classifier_param,
        random_state=DEFAULT_RANDOM_SEED,
    )


def _stack_features(cache: CovarianceFeatureCache, subject_ids: Sequence[str], candidate: CovarianceCandidateSpec) -> np.ndarray:
    return np.concatenate([cache.get(subject_id, candidate) for subject_id in subject_ids], axis=0)


def _stack_labels(subjects: Mapping[str, SubjectEpochs], subject_ids: Sequence[str]) -> np.ndarray:
    return np.concatenate([subjects[subject_id].labels for subject_id in subject_ids], axis=0)


def _stable_seed(seed: int, context: Sequence[Any]) -> int:
    payload = json.dumps([int(seed), *[str(item) for item in context]], sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


def _shuffle_training_labels(labels: np.ndarray, *, seed: int, context: Sequence[Any]) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    rng = np.random.default_rng(_stable_seed(seed, context))
    return rng.permutation(labels)


def _predict_candidate(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: CovarianceFeatureCache,
    candidate: CovarianceCandidateSpec,
    train_subjects: Sequence[str],
    test_subject: str,
    n_classes: int,
    max_iter: int,
    label_shuffle_control: bool = False,
    label_shuffle_seed: int = 0,
    label_shuffle_context: Sequence[Any] = (),
) -> np.ndarray:
    train_features = _stack_features(cache, train_subjects, candidate)
    train_labels = _stack_labels(subjects, train_subjects)
    if label_shuffle_control:
        train_labels = _shuffle_training_labels(train_labels, seed=int(label_shuffle_seed), context=label_shuffle_context)
    test_features = cache.get(test_subject, candidate)
    model = _make_model(candidate, n_features=train_features.shape[1], n_samples=train_features.shape[0], max_iter=max_iter)
    model.fit(train_features, train_labels)
    probabilities = predict_emission_probabilities(model, test_features, emission_mode=candidate.emission_mode)
    return _base._align_probability_columns(probabilities, model=model, classes=np.arange(int(n_classes)))


def _score_is_better(candidate_score: float, incumbent_score: float | None, *, metric: str) -> bool:
    if incumbent_score is None:
        return True
    if metric in MINIMIZE_SELECTION_METRICS:
        return candidate_score < incumbent_score
    return candidate_score > incumbent_score


def _candidate_rowspec(candidate: CovarianceCandidateSpec) -> dict[str, Any]:
    return {
        "candidate": candidate.name,
        "decoder": normalize_decoder_name(candidate.decoder),
        "emission_mode": normalize_emission_mode(candidate.emission_mode),
        "feature_preprocessor": normalize_feature_preprocessor(candidate.feature_preprocessor),
        "pca_components": "" if candidate.pca_components is None else candidate.pca_components,
        "classifier_param": "" if candidate.classifier_param is None else candidate.classifier_param,
        "feature_family": "covariance",
        "covariance_feature_mode": normalize_covariance_feature_mode(candidate.covariance_feature_mode),
        "covariance_shrinkage": candidate.covariance_shrinkage,
        "covariance_epsilon": candidate.covariance_epsilon,
        "covariance_max_channels": candidate.covariance_max_channels,
        "window_name": candidate.window.name,
        "window_start": candidate.window.start,
        "window_stop": candidate.window.stop,
        "window_width": candidate.window.stop - candidate.window.start,
    }


def _inner_loso_scores(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: CovarianceFeatureCache,
    candidate: CovarianceCandidateSpec,
    outer_test_subject: str,
    n_classes: int,
    max_iter: int,
    label_shuffle_control: bool,
    label_shuffle_seed: int,
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
            label_shuffle_control=label_shuffle_control,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=(outer_test_subject, inner_test_subject, candidate.name),
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
                "label_shuffle_control": bool(label_shuffle_control),
                "label_shuffle_seed": int(label_shuffle_seed),
            }
        )
    return rows


def _select_candidate(
    *,
    subjects: Mapping[str, SubjectEpochs],
    cache: CovarianceFeatureCache,
    candidates: Sequence[CovarianceCandidateSpec],
    outer_test_subject: str,
    n_classes: int,
    max_iter: int,
    selection_metric: str,
    label_shuffle_control: bool,
    label_shuffle_seed: int,
) -> tuple[CovarianceCandidateSpec, list[dict[str, Any]], dict[str, Any]]:
    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(f"Unknown selection metric '{selection_metric}'. Available metrics: {sorted(SUPPORTED_SELECTION_METRICS)}.")
    all_rows: list[dict[str, Any]] = []
    selected: CovarianceCandidateSpec | None = None
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
            label_shuffle_control=label_shuffle_control,
            label_shuffle_seed=label_shuffle_seed,
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
        raise ValueError("No candidates were available for covariance LOSO selection.")
    return selected, all_rows, selected_summary


def run_covariance_loso_subjects(
    subjects: Mapping[str, SubjectEpochs],
    *,
    candidates: Sequence[CovarianceCandidateSpec],
    class_names: Sequence[Any],
    selection_metric: str = DEFAULT_SELECTION_METRIC,
    max_iter: int = 1000,
    label_shuffle_control: bool = False,
    label_shuffle_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run nested source-only covariance LOSO on already-loaded subjects."""

    if len(subjects) < 3:
        raise ValueError("Need at least three subjects for nested source-only LOSO.")
    if not candidates:
        raise ValueError("At least one covariance candidate is required.")
    cache = CovarianceFeatureCache(subjects)
    n_classes = len(class_names)
    class_names = list(class_names)

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
            label_shuffle_control=label_shuffle_control,
            label_shuffle_seed=label_shuffle_seed,
        )
        inner_rows.extend(candidate_inner_rows)
        train_subjects = [subject for subject in sorted(subjects) if subject != outer_test_subject]
        probabilities = _predict_candidate(
            subjects=subjects,
            cache=cache,
            candidate=selected,
            train_subjects=train_subjects,
            test_subject=outer_test_subject,
            n_classes=n_classes,
            max_iter=max_iter,
            label_shuffle_control=label_shuffle_control,
            label_shuffle_seed=label_shuffle_seed,
            label_shuffle_context=(outer_test_subject, selected.name, "outer"),
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
                "n_test_trials": len(labels),
                "n_classes": n_classes,
                "class_names": "|".join(map(str, class_names)),
                "label_shuffle_control": bool(label_shuffle_control),
                "label_shuffle_seed": int(label_shuffle_seed),
            }
        )

        metadata = subjects[outer_test_subject].metadata.reset_index(drop=True)
        for row_idx, (true_label, predicted_label) in enumerate(zip(labels, predictions, strict=True)):
            row: dict[str, Any] = {
                "outer_test_subject": outer_test_subject,
                "trial_index": int(row_idx),
                "candidate": selected.name,
                "true_label": int(true_label),
                "true_class": str(class_names[true_label]),
                "predicted_label": int(predicted_label),
                "predicted_class": str(class_names[predicted_label]),
                "probability_true_class": float(probabilities[row_idx, true_label]),
                "confidence": float(np.max(probabilities[row_idx])),
                "is_correct": bool(predicted_label == true_label),
                "feature_family": "covariance",
                "covariance_feature_mode": normalize_covariance_feature_mode(selected.covariance_feature_mode),
                "label_shuffle_control": bool(label_shuffle_control),
            }
            for column in ("participant", "condition", "stimulus_class"):
                if column in metadata.columns:
                    row[column] = metadata.loc[row_idx, column]
            for class_idx, class_name in enumerate(class_names):
                row[f"class_{class_idx}"] = str(class_name)
                row[f"prob_class_{class_idx}"] = float(probabilities[row_idx, class_idx])
            prediction_rows.append(row)

    return pd.DataFrame(summary_rows), pd.DataFrame(inner_rows), pd.DataFrame(prediction_rows)


def run_bushmeg_covariance_loso(
    config_path: str | Path,
    *,
    overrides: Sequence[str] | None = None,
    out_path: str | Path | None = None,
    inner_cv_out_path: str | Path | None = None,
    predictions_out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run strict cue-free BUSH-MEG covariance-feature nested LOSO decoding."""

    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    covariance_loso = _section(config, "covariance_loso")
    selection_metric = str(covariance_loso.get("selection_metric", DEFAULT_SELECTION_METRIC))
    label_shuffle_control = _as_bool(covariance_loso.get("label_shuffle_control"), default=False)
    label_shuffle_seed = int(covariance_loso.get("label_shuffle_seed", 0))
    max_iter = int((_section(config, "decoding") or {}).get("max_iter", covariance_loso.get("max_iter", 1000)))

    subjects, encoder = _load_subjects_from_config(config, config_dir=config_path.parent)
    candidates = _candidate_grid(config)
    summary, inner, predictions = run_covariance_loso_subjects(
        subjects,
        candidates=candidates,
        class_names=encoder.classes_,
        selection_metric=selection_metric,
        max_iter=max_iter,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )

    out = Path(out_path) if out_path is not None else _resolve_output(
        config,
        config_dir=config_path.parent,
        key="covariance_loso_summary_csv",
        default="covariance_loso_summary.csv",
    )
    inner_out = Path(inner_cv_out_path) if inner_cv_out_path is not None else _resolve_output(
        config,
        config_dir=config_path.parent,
        key="covariance_loso_inner_cv_csv",
        default="covariance_loso_inner_cv.csv",
    )
    predictions_out = Path(predictions_out_path) if predictions_out_path is not None else _resolve_output(
        config,
        config_dir=config_path.parent,
        key="covariance_loso_predictions_csv",
        default="covariance_loso_predictions.csv",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    inner_out.parent.mkdir(parents=True, exist_ok=True)
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    inner.to_csv(inner_out, index=False)
    predictions.to_csv(predictions_out, index=False)
    _write_json_sidecar(
        out,
        {
            "config_path": str(config_path),
            "selection_metric": selection_metric,
            "n_subjects": len(subjects),
            "n_candidates": len(candidates),
            "feature_family": "covariance",
            "covariance_feature_modes": sorted({normalize_covariance_feature_mode(candidate.covariance_feature_mode) for candidate in candidates}),
            "covariance_shrinkages": sorted({float(candidate.covariance_shrinkage) for candidate in candidates}),
            "covariance_epsilons": sorted({float(candidate.covariance_epsilon) for candidate in candidates}),
            "normalization_scope": "subject_unlabeled_baseline",
            "cue_files_used": False,
            "label_shuffle_control": label_shuffle_control,
            "label_shuffle_seed": label_shuffle_seed,
            "random_seed": DEFAULT_RANDOM_SEED,
        },
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run strict cue-free covariance-feature nested LOSO decoding for BUSH-MEG main-task FieldTrip MAT files."
    )
    parser.add_argument("config", type=Path, help="Dataset/workflow config, for example configs/bush_meg/covariance_loso.yml.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a dotted config key.")
    parser.add_argument("--out", type=Path, help="Summary CSV path.")
    parser.add_argument("--inner-cv-out", type=Path, help="Inner LOSO candidate-score CSV path.")
    parser.add_argument("--predictions-out", type=Path, help="Held-out trial probability CSV path.")
    args = parser.parse_args(argv)

    summary = run_bushmeg_covariance_loso(
        args.config,
        overrides=args.overrides,
        out_path=args.out,
        inner_cv_out_path=args.inner_cv_out,
        predictions_out_path=args.predictions_out,
    )
    print(f"Wrote {len(summary)} covariance LOSO rows")
    print(f"Mean balanced accuracy: {float(summary['balanced_accuracy'].mean()):.6f}")
    print(f"Mean top-2/top-3 accuracy: {float(summary['top2_accuracy'].mean()):.6f} / {float(summary['top3_accuracy'].mean()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

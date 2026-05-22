"""Strict source-only LOSO decoding workflow for the BUSH-MEG main task.

The workflow is intentionally cue-free: it loads only the configured main-task
FieldTrip MATLAB files, performs leave-one-subject-out evaluation, and selects
window/model hyperparameters by an inner LOSO loop over source subjects only.

The implementation differs from the generic time-resolved decoder in two ways
that matter for BUSH-MEG:

* temporal features are compact per-channel bin means rather than very large
  sensor-by-sample windows;
* a candidate may average probabilities from several nearby post-stimulus
  windows, giving a strict source-only temporal bagging baseline.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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

DEFAULT_SELECTION_METRIC = "balanced_accuracy"
DEFAULT_RANDOM_SEED = 13
SUPPORTED_SELECTION_METRICS = {"balanced_accuracy", "accuracy", "log_loss"}
MINIMIZE_SELECTION_METRICS = {"log_loss"}


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
        self._cache: dict[tuple[str, WindowSpec, int], np.ndarray] = {}

    def get(self, subject: str, window: WindowSpec, temporal_bins: int) -> np.ndarray:
        key = (subject, window, int(temporal_bins))
        if key not in self._cache:
            subject_epochs = self._subjects[subject]
            self._cache[key] = _window_bin_mean_features(
                subject_epochs.data,
                subject_epochs.times,
                window,
                temporal_bins=int(temporal_bins),
            )
        return self._cache[key]


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


def _window_size_seconds(preprocessing: Mapping[str, Any], default: float = 0.100) -> float:
    if "window_size" in preprocessing:
        return float(preprocessing["window_size"])
    if "window_ms" in preprocessing:
        return float(preprocessing["window_ms"]) / 1000.0
    return float(default)


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
    preprocessing = _section(config, "preprocessing")
    decoding = _section(config, "decoding") or _section(config, "workflow")
    source_loso = _section(config, "source_loso")
    label_column = str(decoding.get("label_column", "stimulus_class"))
    group_column = str(source_loso.get("group_column", decoding.get("group_column", "participant")))
    baseline_window = tuple(preprocessing.get("baseline_window", _base.DEFAULT_BASELINE_WINDOW))
    normalization = str(preprocessing.get("normalization", "none"))
    tmin = preprocessing.get("tmin")
    tmax = preprocessing.get("tmax")

    loader_config: dict[str, Any] = {**dataset, "metadata": metadata_config}
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


def _stack_subject_features(cache: FeatureCache, subjects: Mapping[str, SubjectEpochs], subject_ids: Sequence[str], window: WindowSpec, temporal_bins: int) -> np.ndarray:
    return np.concatenate([cache.get(subject_id, window, temporal_bins) for subject_id in subject_ids], axis=0)


def _stack_subject_labels(subjects: Mapping[str, SubjectEpochs], subject_ids: Sequence[str]) -> np.ndarray:
    return np.concatenate([subjects[subject_id].labels for subject_id in subject_ids], axis=0)


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    effective_k = min(int(k), probabilities.shape[1])
    top_columns = np.argsort(probabilities, axis=1)[:, ::-1][:, :effective_k]
    return float(np.mean(np.any(top_columns == labels[:, None], axis=1)))


def _candidate_model(candidate: CandidateSpec, *, max_iter: int):
    return make_decoder(
        candidate.decoder,
        max_iter=max_iter,
        emission_mode=candidate.emission_mode,
        feature_preprocessor=candidate.feature_preprocessor,
        pca_components=candidate.pca_components,
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
) -> np.ndarray:
    train_labels = _stack_subject_labels(subjects, train_subjects)
    test_n = len(subjects[test_subject].labels)
    probabilities_sum = np.zeros((test_n, n_classes), dtype=float)
    classes = np.arange(n_classes)
    for window in candidate.windows:
        train_features = _stack_subject_features(cache, subjects, train_subjects, window, candidate.temporal_bins)
        test_features = cache.get(test_subject, window, candidate.temporal_bins)
        model = _candidate_model(candidate, max_iter=max_iter)
        model.fit(train_features, train_labels)
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
    return _base._probability_average(probabilities_sum, len(candidate.windows))


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
        "decoder": normalize_decoder_name(candidate.decoder),
        "emission_mode": normalize_emission_mode(candidate.emission_mode),
        "feature_preprocessor": normalize_feature_preprocessor(candidate.feature_preprocessor),
        "pca_components": "" if candidate.pca_components is None else candidate.pca_components,
        "classifier_param": "" if candidate.classifier_param is None else candidate.classifier_param,
        "temporal_bins": candidate.temporal_bins,
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
        name = str(item.get("name", "windows"))
        width = float(item.get("window_size", item.get("width", default_window_width)))
        centers = [float(center) for center in _list_value(item.get("centers"), [])]
        if not centers:
            raise ValueError(f"window set '{name}' must contain at least one center.")
        window_sets.append((name, tuple(WindowSpec(center=center, width=width) for center in centers)))

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
    normalized_pca_values = [None if value in {None, "", "none", "None"} else normalize_pca_components(value) for value in pca_values]
    temporal_bins_values = [int(value) for value in _list_value(grid.get("temporal_bins"), [4])]
    c_grid = [float(value) for value in parse_c_grid(grid.get("c_grid", decoding.get("tuning_c_grid", "0.1,1.0,10.0")))]

    candidates: list[CandidateSpec] = []
    for window_name, windows in window_sets:
        for decoder in decoders:
            for emission_mode in emission_modes:
                for feature_preprocessor in feature_preprocessors:
                    for pca_components in normalized_pca_values:
                        for temporal_bins in temporal_bins_values:
                            for c_value in c_grid:
                                normalized_decoder = normalize_decoder_name(decoder)
                                name = "__".join(
                                    [
                                        window_name,
                                        normalized_decoder,
                                        normalize_emission_mode(emission_mode),
                                        normalize_feature_preprocessor(feature_preprocessor),
                                        "pca" + ("none" if pca_components is None else str(pca_components)),
                                        f"bins{temporal_bins}",
                                        f"c{c_value:g}",
                                    ]
                                )
                                candidates.append(
                                    CandidateSpec(
                                        name=name,
                                        decoder=decoder,
                                        emission_mode=emission_mode,
                                        feature_preprocessor=feature_preprocessor,
                                        pca_components=pca_components,
                                        classifier_param=c_value,
                                        temporal_bins=temporal_bins,
                                        windows=windows,
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
    _write_json_sidecar(
        out,
        {
            "config_path": str(config_path),
            "selection_metric": selection_metric,
            "n_subjects": len(subjects),
            "n_candidates": len(candidates),
            "normalization_scope": "subject_unlabeled_baseline",
            "cue_files_used": False,
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

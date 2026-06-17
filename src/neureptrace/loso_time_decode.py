"""Leave-one-subject-out source-only time decoding from a dataset config.

This module is intended for datasets such as BUSH-MEG where the scientific
target is held-out-subject generalisation, not within-subject cross-validation.
It deliberately keeps cue/calibration files out of the training target by using
metadata filters and supports a training-subject-only temporal window selector.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from neureptrace.dataset_config import apply_overrides, effective_config, load_config, load_epoch_dataset_from_config
from neureptrace.decode_from_config import _bool_value, _resolve_output, _section, _window_ms, _write_provenance_sidecars
from neureptrace.decoding import (
    make_decoder,
    make_tuning_cross_validator,
    normalize_anova_select_percentile,
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    normalize_pca_components,
    normalize_tuning_scoring,
    parse_c_grid,
    predict_emission_probabilities,
    time_windows,
)
from neureptrace.metrics import brier_score_multiclass, expected_calibration_error
from neureptrace.mne_time_decode import (
    DEFAULT_BASELINE_WINDOW,
    _align_probability_columns,
    _apply_epoch_normalization,
    _features_for_window,
    _normalize_baseline_window,
    _normalize_integer,
    _normalize_positive_int,
    _probability_average,
    _top_k_accuracy,
)
from neureptrace.mne_time_decode_foldlocal import _apply_fold_epoch_normalization, _fit_fold_epoch_normalization
from neureptrace.observations import ProbabilityObservationTable, stable_hash

LOSO_METRIC_CHOICES = (
    "accuracy",
    "balanced_accuracy",
    "top2_accuracy",
    "top3_accuracy",
    "log_loss",
    "brier",
    "ece",
)
LOSO_MINIMIZE_METRICS = {"log_loss", "brier", "ece"}
NORMALIZATION_SCOPE_CHOICES = ("per_group", "train", "global")
DEFAULT_SOURCE_SELECT_TOP_K = 0
DEFAULT_SOURCE_SELECT_INNER_SPLITS = 4


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and "," in value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return [value]


def _loso_section(config: Mapping[str, Any]) -> dict[str, Any]:
    value = config.get("loso") or config.get("workflow") or config.get("decoding") or {}
    if not isinstance(value, Mapping):
        raise ValueError("Config section 'loso' must be a mapping when present.")
    return dict(value)


def _decoder_names(value: Any) -> tuple[str, ...]:
    raw_decoders = _as_list(value if value is not None else "logistic")
    decoders = tuple(normalize_decoder_name(str(decoder)) for decoder in raw_decoders)
    if not decoders:
        raise ValueError("At least one LOSO decoder is required.")
    return decoders


def _normalization_scope(value: str | None) -> str:
    scope = "per_group" if value is None else str(value).strip().lower().replace("-", "_")
    aliases = {
        "subject": "per_group",
        "participant": "per_group",
        "group": "per_group",
        "per_subject": "per_group",
        "per_participant": "per_group",
        "source_only": "train",
        "train_fold": "train",
        "outer_train": "train",
    }
    scope = aliases.get(scope, scope)
    if scope not in NORMALIZATION_SCOPE_CHOICES:
        raise ValueError(f"Unknown normalization scope '{value}'. Available scopes: {', '.join(NORMALIZATION_SCOPE_CHOICES)}.")
    return scope


def _filter_mask(metadata: pd.DataFrame, filter_spec: Mapping[str, Any] | None, *, name: str) -> np.ndarray:
    mask = np.ones(len(metadata), dtype=bool)
    if not filter_spec:
        return mask
    if not isinstance(filter_spec, Mapping):
        raise ValueError(f"{name} filter must be a mapping from metadata columns to allowed values.")
    for column_name, expected in filter_spec.items():
        if column_name not in metadata.columns:
            raise ValueError(f"{name} filter references unknown metadata column '{column_name}'.")
        mask &= metadata[column_name].isin(_as_list(expected)).to_numpy()
    return mask


def _infer_group_column(metadata: pd.DataFrame, configured: str | None) -> str:
    if configured:
        if configured not in metadata.columns:
            raise ValueError(f"LOSO group column '{configured}' not found in metadata.")
        return configured
    for candidate in ("subject", "participant", "participant_id", "subject_id"):
        if candidate in metadata.columns:
            return candidate
    raise ValueError("Could not infer LOSO group column. Set loso.group_column, e.g. 'participant' or 'subject'.")


def _finite_time_bound(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _time_interval(value: Any, *, name: str = "time interval") -> tuple[float, float] | None:
    if value is None:
        return None
    values = _as_list(value)
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two entries: start and stop.")
    start = _finite_time_bound(values[0], name=name)
    stop = _finite_time_bound(values[1], name=name)
    if stop < start:
        raise ValueError(f"{name} stop must be greater than or equal to start.")
    return start, stop


def _select_windows_by_interval(windows: Sequence[tuple[int, int, float]], interval: tuple[float, float] | None) -> list[tuple[int, int, float]]:
    if interval is None:
        return list(windows)
    start, stop = interval
    selected = [window for window in windows if start <= window[2] <= stop]
    if not selected:
        centers = [window[2] for window in windows]
        raise ValueError(
            f"No time-window centers fall in [{start}, {stop}]. Available centers span [{min(centers)}, {max(centers)}]."
        )
    return selected


def _preprocessed_data_for_outer_fold(
    data: np.ndarray,
    times: np.ndarray,
    metadata: pd.DataFrame,
    *,
    normalization: str,
    normalization_scope: str,
    baseline_window: tuple[float, float],
    train_indices: np.ndarray,
    group_column: str,
) -> np.ndarray:
    """Return normalized data for one outer LOSO fold.

    ``per_group`` is the default because it matches the common BUSH/MEG
    source-only setting: labels from the held-out subject are never used, but
    label-free baseline statistics from that subject may standardize its sensor
    scale. Set ``normalization_scope: train`` for the stricter variant where all
    normalization parameters come only from source subjects.
    """

    if normalization_scope == "global":
        return _apply_epoch_normalization(data, times, normalization, baseline_window=baseline_window)

    if normalization_scope == "train":
        params = _fit_fold_epoch_normalization(data, times, normalization, baseline_window=baseline_window, train_idx=train_indices)
        return _apply_fold_epoch_normalization(data, normalization, params)

    if normalization_scope != "per_group":
        raise ValueError(f"Unsupported normalization scope: {normalization_scope}")

    normalized = np.asarray(data, dtype=float).copy()
    groups = metadata[group_column].to_numpy()
    for group_value in pd.unique(groups):
        group_indices = np.flatnonzero(groups == group_value)
        params = _fit_fold_epoch_normalization(data, times, normalization, baseline_window=baseline_window, train_idx=group_indices)
        normalized[group_indices] = _apply_fold_epoch_normalization(data[group_indices], normalization, params)
    return normalized


def _feasible_source_cv_splits(labels: np.ndarray, groups: np.ndarray | None, requested_splits: int):
    _, class_counts = np.unique(labels, return_counts=True)
    if len(class_counts) < 2:
        raise ValueError("Need at least two classes for source-window selection.")
    feasible_splits = min(int(requested_splits), int(np.min(class_counts)))
    if groups is not None:
        feasible_splits = min(feasible_splits, len(np.unique(groups)))
    if feasible_splits < 2:
        raise ValueError("Need at least two inner folds for source-window selection.")
    if groups is not None:
        return StratifiedGroupKFold(n_splits=feasible_splits).split(np.zeros_like(labels), labels, groups)
    return StratifiedKFold(n_splits=feasible_splits, shuffle=True, random_state=13).split(np.zeros_like(labels), labels)


def _metric_from_probabilities(metric: str, probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray) -> float:
    predictions = probabilities.argmax(axis=1)
    if metric == "accuracy":
        return float(accuracy_score(labels, predictions))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(labels, predictions))
    if metric == "top2_accuracy":
        return _top_k_accuracy(probabilities, labels, k=2)
    if metric == "top3_accuracy":
        return _top_k_accuracy(probabilities, labels, k=3)
    if metric == "log_loss":
        return float(log_loss(labels, probabilities, labels=classes))
    if metric == "brier":
        return float(brier_score_multiclass(probabilities, labels))
    if metric == "ece":
        return float(expected_calibration_error(probabilities, labels))
    raise ValueError(f"Unknown metric '{metric}'.")


def _sort_window_scores(scores: list[tuple[float, tuple[int, int, float]]], metric: str) -> list[tuple[float, tuple[int, int, float]]]:
    reverse = metric not in LOSO_MINIMIZE_METRICS
    return sorted(scores, key=lambda item: item[0], reverse=reverse)


def _make_model(
    decoder_name: str,
    *,
    max_iter: int,
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: int | float | None,
    tune_hyperparameters: bool,
    tuning_cv,
    tuning_scoring: str,
    tuning_c_grid: Sequence[float],
):
    return make_decoder(
        decoder_name,
        max_iter=max_iter,
        emission_mode=emission_mode,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv=tuning_cv,
        tuning_scoring=tuning_scoring,
        tuning_c_grid=tuning_c_grid,
    )


def _predict_for_train_window(
    *,
    data: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    train_window: tuple[int, int, float],
    test_window: tuple[int, int, float],
    decoder_name: str,
    max_iter: int,
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: int | float | None,
    tune_hyperparameters: bool,
    tuning_cv,
    tuning_scoring: str,
    tuning_c_grid: Sequence[float],
):
    train_features = _features_for_window(data, train_window)
    test_features = _features_for_window(data, test_window)
    model = _make_model(
        decoder_name,
        max_iter=max_iter,
        emission_mode=emission_mode,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv=tuning_cv,
        tuning_scoring=tuning_scoring,
        tuning_c_grid=tuning_c_grid,
    )
    model.fit(train_features[train_indices], labels[train_indices])
    probabilities = _align_probability_columns(
        predict_emission_probabilities(model, test_features[test_indices], emission_mode=emission_mode),
        model=model,
        classes=classes,
    )
    return model, probabilities


def _rank_source_windows(
    *,
    data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    source_indices: np.ndarray,
    windows: Sequence[tuple[int, int, float]],
    decoder_names: Sequence[str],
    classes: np.ndarray,
    max_iter: int,
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: int | float | None,
    tune_hyperparameters: bool,
    tuning_cv_splits: int,
    tuning_scoring: str,
    tuning_c_grid: Sequence[float],
    source_select_inner_splits: int,
    source_select_metric: str,
) -> pd.DataFrame:
    source_labels = labels[source_indices]
    source_groups = groups[source_indices]
    split_iter = _feasible_source_cv_splits(source_labels, source_groups, source_select_inner_splits)
    inner_splits = [(source_indices[train_local], source_indices[val_local]) for train_local, val_local in split_iter]
    score_rows: list[dict[str, Any]] = []
    for window in windows:
        fold_scores: list[float] = []
        for inner_train_indices, inner_val_indices in inner_splits:
            probability_sum = np.zeros((len(inner_val_indices), len(classes)), dtype=float)
            for decoder_name in decoder_names:
                tuning_cv = (
                    make_tuning_cross_validator(labels[inner_train_indices], groups[inner_train_indices], tuning_cv_splits)
                    if tune_hyperparameters
                    else 3
                )
                _model, probabilities = _predict_for_train_window(
                    data=data,
                    labels=labels,
                    classes=classes,
                    train_indices=inner_train_indices,
                    test_indices=inner_val_indices,
                    train_window=window,
                    test_window=window,
                    decoder_name=decoder_name,
                    max_iter=max_iter,
                    emission_mode=emission_mode,
                    feature_preprocessor=feature_preprocessor,
                    pca_components=pca_components,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv=tuning_cv,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid,
                )
                probability_sum += probabilities
            probabilities = _probability_average(probability_sum, len(decoder_names))
            fold_scores.append(_metric_from_probabilities(source_select_metric, probabilities, labels[inner_val_indices], classes))
        _start, _stop, center = window
        score_rows.append(
            {
                "train_time": center,
                "source_select_metric": source_select_metric,
                "source_select_score": float(np.mean(fold_scores)),
                "source_select_score_std": float(np.std(fold_scores)),
                "source_select_inner_folds": len(fold_scores),
            }
        )
    return pd.DataFrame(score_rows)


def _selected_windows_for_outer_fold(
    *,
    data: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    source_indices: np.ndarray,
    windows: Sequence[tuple[int, int, float]],
    decoder_names: Sequence[str],
    classes: np.ndarray,
    source_select_top_k: int,
    source_select_metric: str,
    max_iter: int,
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: int | float | None,
    tune_hyperparameters: bool,
    tuning_cv_splits: int,
    tuning_scoring: str,
    tuning_c_grid: Sequence[float],
    source_select_inner_splits: int,
) -> tuple[list[tuple[int, int, float]], pd.DataFrame]:
    if source_select_top_k <= 0:
        return list(windows), pd.DataFrame()
    ranking = _rank_source_windows(
        data=data,
        labels=labels,
        groups=groups,
        source_indices=source_indices,
        windows=windows,
        decoder_names=decoder_names,
        classes=classes,
        max_iter=max_iter,
        emission_mode=emission_mode,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv_splits=tuning_cv_splits,
        tuning_scoring=tuning_scoring,
        tuning_c_grid=tuning_c_grid,
        source_select_inner_splits=source_select_inner_splits,
        source_select_metric=source_select_metric,
    )
    score_by_time = {float(row.train_time): float(row.source_select_score) for row in ranking.itertuples(index=False)}
    ranked = _sort_window_scores([(score_by_time[float(window[2])], window) for window in windows], source_select_metric)
    selected = [window for _score, window in ranked[: int(source_select_top_k)]]
    return selected, ranking


def _common_row(
    *,
    outer_fold: int,
    outer_group: Any,
    decoder_names: Sequence[str],
    emission_mode: str,
    feature_preprocessor: str,
    pca_components: int | float | None,
    normalization: str,
    normalization_scope: str,
    baseline_window: tuple[float, float],
    source_select_top_k: int,
    source_select_metric: str,
    selected_train_windows: Sequence[tuple[int, int, float]],
    test_window: tuple[int, int, float],
    times: np.ndarray,
) -> dict[str, Any]:
    start, stop, center = test_window
    selected_times = [float(window[2]) for window in selected_train_windows]
    if source_select_top_k > 0:
        temporal_mode = "source_selected_train_window_ensemble"
    elif len(selected_train_windows) > 1:
        temporal_mode = "fixed_train_window_ensemble"
    else:
        temporal_mode = "same_time_loso"
    return {
        "fold": int(outer_fold),
        "outer_group": outer_group,
        "decoder": decoder_names[0] if len(decoder_names) == 1 else "decoder_ensemble",
        "source_decoders": "|".join(decoder_names),
        "emission_mode": emission_mode,
        "feature_preprocessor": feature_preprocessor,
        "pca_components": "" if pca_components is None else pca_components,
        "normalization": normalization,
        "normalization_scope": normalization_scope,
        "baseline_window_start": baseline_window[0],
        "baseline_window_stop": baseline_window[1],
        "temporal_mode": temporal_mode,
        "time": float(center),
        "test_time": float(center),
        "window_start": float(times[start]),
        "window_stop": float(times[stop - 1]),
        "n_train_windows": int(len(selected_train_windows)),
        "train_time": float(np.mean(selected_times)),
        "selected_train_times": "|".join(f"{value:.12g}" for value in selected_times),
        "source_select_top_k": int(source_select_top_k),
        "source_select_metric": source_select_metric if source_select_top_k > 0 else "",
    }


def _append_observation_rows(
    rows: list[dict[str, Any]],
    *,
    metadata: pd.DataFrame,
    test_indices: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: np.ndarray,
    common: Mapping[str, Any],
    preprocessing_hash: str,
    model_hash: str,
    group_column: str,
) -> None:
    predictions = probabilities.argmax(axis=1)
    for local_position, sample_index in enumerate(test_indices):
        true_label = int(labels[sample_index])
        predicted_label = int(predictions[local_position])
        row = {
            **common,
            "sample_index": int(sample_index),
            "sequence_id": int(sample_index),
            "session": metadata.iloc[sample_index].get("session", metadata.iloc[sample_index].get(group_column, "")),
            "group": metadata.iloc[sample_index].get(group_column, ""),
            "true_label": true_label,
            "true_class": str(class_names[true_label]),
            "predicted_label": predicted_label,
            "predicted_class": str(class_names[predicted_label]),
            "probability_true_class": float(probabilities[local_position, true_label]),
            "confidence": float(probabilities[local_position].max()),
            "is_correct": bool(predicted_label == true_label),
            "backend": "sklearn",
            "split_id": f"loso:{group_column}",
            "seed": 13,
            "calibration_fold": "",
            "preprocessing_hash": preprocessing_hash,
            "model_hash": model_hash,
        }
        for class_index, class_name in enumerate(class_names):
            row[f"class_{class_index}"] = str(class_name)
            row[f"prob_class_{class_index}"] = float(probabilities[local_position, class_index])
        rows.append(row)


def _output_paths(config: Mapping[str, Any], *, config_dir: Path) -> tuple[Path, Path | None, Path | None]:
    summary = _resolve_output(config, config_dir=config_dir, key="summary_csv", default="results/{dataset}_loso_summary.csv")
    observations = _resolve_output(config, config_dir=config_dir, key="observations_csv")
    source_scores = _resolve_output(config, config_dir=config_dir, key="source_window_scores_csv")
    if summary is None:
        raise ValueError("LOSO decoding requires outputs.summary_csv.")
    return summary, observations, source_scores


def run_loso_time_decode(
    config_path: str | Path,
    *,
    overrides: Sequence[str] | None = None,
    write_provenance: bool | None = None,
) -> pd.DataFrame:
    """Run leave-one-group-out source-only decoding from a dataset config."""

    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    if write_provenance is not None:
        config.setdefault("outputs", {})["provenance"] = _bool_value(write_provenance, name="write_provenance")

    dataset = load_epoch_dataset_from_config(config, base_dir=config_path.parent, check_files=True)
    preprocessing = _section(config, "preprocessing")
    loso = _loso_section(config)
    label_column = loso.get("label_column") or _section(config, "decoding").get("label_column")
    if not label_column:
        raise ValueError("LOSO decoding requires loso.label_column or decoding.label_column.")
    if label_column not in dataset.metadata.columns:
        raise ValueError(f"Label column '{label_column}' not found in metadata.")

    group_column = _infer_group_column(dataset.metadata, loso.get("group_column") or loso.get("subject_column"))
    trial_mask = _filter_mask(dataset.metadata, loso.get("trial_filter") or loso.get("filter"), name="loso.trial_filter")
    raw_labels = dataset.metadata[label_column].to_numpy()
    trial_mask &= pd.notna(raw_labels)
    if not np.any(trial_mask):
        raise ValueError("LOSO trial filter selected no labeled trials.")

    metadata = dataset.metadata.loc[trial_mask].reset_index(drop=True)
    data = np.asarray(dataset.data, dtype=float)[trial_mask]
    raw_labels = raw_labels[trial_mask]
    groups = metadata[group_column].to_numpy()
    test_groups = _as_list(loso.get("test_groups") or loso.get("heldout_groups"))
    outer_groups = [group for group in pd.unique(groups) if not test_groups or group in test_groups]
    if len(outer_groups) < 1:
        raise ValueError("No LOSO groups are available after filtering.")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(raw_labels)
    classes = np.arange(len(encoder.classes_))
    decoder_names = _decoder_names(loso.get("decoders", loso.get("decoder", loso.get("classifier", "logistic"))))
    emission_mode = normalize_emission_mode(loso.get("emission_mode", "calibrated"))
    feature_preprocessor = normalize_feature_preprocessor(loso.get("feature_preprocessor", preprocessing.get("feature_preprocessor", "none")))
    if feature_preprocessor == "none":
        pca_components = None
        if loso.get("pca_components", preprocessing.get("pca_components")) is not None:
            raise ValueError("pca_components can only be set when a feature_preprocessor is active.")
    elif feature_preprocessor == "anova_select":
        pca_components = normalize_anova_select_percentile(loso.get("pca_components", preprocessing.get("pca_components")))
    else:
        pca_components = normalize_pca_components(loso.get("pca_components", preprocessing.get("pca_components")))
    normalization = str(preprocessing.get("normalization", "none")).strip().lower().replace("-", "_")
    normalization_scope = _normalization_scope(loso.get("normalization_scope", preprocessing.get("normalization_scope")))
    baseline_window = _normalize_baseline_window(preprocessing.get("baseline_window", DEFAULT_BASELINE_WINDOW))
    max_iter = _normalize_positive_int(
        loso.get("max_iter", _section(config, "decoding").get("max_iter", 1000)),
        name="loso.max_iter",
    )
    tune_hyperparameters = _bool_value(loso.get("tune_hyperparameters"), name="loso.tune_hyperparameters")
    tuning_cv_splits = _normalize_positive_int(loso.get("tuning_cv_splits", 3), name="loso.tuning_cv_splits")
    tuning_scoring = normalize_tuning_scoring(loso.get("tuning_scoring", "accuracy"))
    tuning_c_grid = parse_c_grid(loso.get("tuning_c_grid"))
    source_select_top_k = _normalize_integer(
        loso.get("source_select_top_k", DEFAULT_SOURCE_SELECT_TOP_K),
        name="loso.source_select_top_k",
        minimum=0,
    )
    source_select_metric = str(loso.get("source_select_metric", "balanced_accuracy")).strip().lower().replace("-", "_")
    if source_select_metric not in LOSO_METRIC_CHOICES:
        raise ValueError(f"Unknown source_select_metric '{source_select_metric}'. Available metrics: {', '.join(LOSO_METRIC_CHOICES)}.")
    source_select_inner_splits = _normalize_integer(
        loso.get("source_select_inner_splits", DEFAULT_SOURCE_SELECT_INNER_SPLITS),
        name="loso.source_select_inner_splits",
        minimum=2,
    )

    windows = time_windows(
        np.asarray(dataset.times, dtype=float),
        window_ms=_window_ms(preprocessing, key_ms="window_ms", key_seconds="window_size", default=20.0),
        step_ms=_window_ms(preprocessing, key_ms="step_ms", key_seconds="window_step", default=10.0),
    )
    decode_interval = _time_interval(
        loso.get("decode_window", preprocessing.get("decode_window")),
        name="loso.decode_window",
    )
    windows = _select_windows_by_interval(windows, decode_interval)
    if "source_select_window" in loso:
        candidate_interval_value = loso.get("source_select_window")
        candidate_interval_name = "loso.source_select_window"
    elif "temporal_train_window" in loso:
        candidate_interval_value = loso.get("temporal_train_window")
        candidate_interval_name = "loso.temporal_train_window"
    else:
        candidate_interval_value = preprocessing.get("temporal_train_window")
        candidate_interval_name = "preprocessing.temporal_train_window"
    candidate_interval = _time_interval(candidate_interval_value, name=candidate_interval_name)
    candidate_train_windows = _select_windows_by_interval(windows, candidate_interval)
    fixed_train_interval = _time_interval(
        loso.get("temporal_train_window", preprocessing.get("temporal_train_window")),
        name="loso.temporal_train_window",
    )
    fixed_train_windows = _select_windows_by_interval(windows, fixed_train_interval) if fixed_train_interval is not None else []

    summary_out, observations_out, source_scores_out = _output_paths(config, config_dir=config_path.parent)
    preprocessing_hash = stable_hash(
        {
            "config": {
                "preprocessing": preprocessing,
                "loso": loso,
                "group_column": group_column,
                "trial_filter": loso.get("trial_filter") or loso.get("filter"),
            }
        }
    )

    metric_rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    source_score_rows: list[pd.DataFrame] = []
    for outer_fold, outer_group in enumerate(outer_groups):
        test_indices = np.flatnonzero(groups == outer_group)
        train_indices = np.flatnonzero(groups != outer_group)
        if len(test_indices) == 0 or len(train_indices) == 0:
            continue
        fold_data = _preprocessed_data_for_outer_fold(
            data,
            np.asarray(dataset.times, dtype=float),
            metadata,
            normalization=normalization,
            normalization_scope=normalization_scope,
            baseline_window=baseline_window,
            train_indices=train_indices,
            group_column=group_column,
        )

        if source_select_top_k > 0:
            selected_train_windows, ranking = _selected_windows_for_outer_fold(
                data=fold_data,
                labels=labels,
                groups=groups,
                source_indices=train_indices,
                windows=candidate_train_windows,
                decoder_names=decoder_names,
                classes=classes,
                source_select_top_k=source_select_top_k,
                source_select_metric=source_select_metric,
                max_iter=max_iter,
                emission_mode=emission_mode,
                feature_preprocessor=feature_preprocessor,
                pca_components=pca_components,
                tune_hyperparameters=tune_hyperparameters,
                tuning_cv_splits=tuning_cv_splits,
                tuning_scoring=tuning_scoring,
                tuning_c_grid=tuning_c_grid,
                source_select_inner_splits=source_select_inner_splits,
            )
            if not ranking.empty:
                ranking = ranking.copy()
                ranking.insert(0, "outer_group", outer_group)
                ranking.insert(0, "fold", outer_fold)
                source_score_rows.append(ranking)
        else:
            selected_train_windows = fixed_train_windows

        prefitted: list[tuple[str, tuple[int, int, float], Any]] = []
        prefitted_tuning_metadata: list[dict[str, Any]] = []

        def _fit_models(train_windows: Sequence[tuple[int, int, float]]):
            fitted: list[tuple[str, tuple[int, int, float], Any]] = []
            tuning_metadata_items: list[dict[str, Any]] = []
            for decoder_name in decoder_names:
                tuning_cv = (
                    make_tuning_cross_validator(labels[train_indices], groups[train_indices], tuning_cv_splits)
                    if tune_hyperparameters
                    else 3
                )
                for train_window in train_windows:
                    model, _unused_probabilities = _predict_for_train_window(
                        data=fold_data,
                        labels=labels,
                        classes=classes,
                        train_indices=train_indices,
                        test_indices=test_indices,
                        train_window=train_window,
                        test_window=train_window,
                        decoder_name=decoder_name,
                        max_iter=max_iter,
                        emission_mode=emission_mode,
                        feature_preprocessor=feature_preprocessor,
                        pca_components=pca_components,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv=tuning_cv,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid,
                    )
                    fitted.append((decoder_name, train_window, model))
                    if tune_hyperparameters:
                        from neureptrace.mne_time_decode import _tuning_metadata

                        tuning_metadata_items.append(
                            _tuning_metadata(
                                model,
                                tune_hyperparameters=True,
                                tuning_cv_splits=tuning_cv_splits,
                                tuning_scoring=tuning_scoring,
                                tuning_c_grid=tuning_c_grid,
                            )
                        )
            return fitted, tuning_metadata_items

        if selected_train_windows:
            prefitted, prefitted_tuning_metadata = _fit_models(selected_train_windows)

        for test_window in windows:
            train_windows_for_test = selected_train_windows or [test_window]
            fitted, tuning_metadata_items = (prefitted, prefitted_tuning_metadata) if prefitted else _fit_models(train_windows_for_test)
            probability_sum = np.zeros((len(test_indices), len(classes)), dtype=float)
            for _decoder_name, _train_window, model in fitted:
                test_features = _features_for_window(fold_data, test_window)
                probability_sum += _align_probability_columns(
                    predict_emission_probabilities(model, test_features[test_indices], emission_mode=emission_mode),
                    model=model,
                    classes=classes,
                )
            probabilities = _probability_average(probability_sum, len(fitted))
            predictions = probabilities.argmax(axis=1)
            common = _common_row(
                outer_fold=outer_fold,
                outer_group=outer_group,
                decoder_names=decoder_names,
                emission_mode=emission_mode,
                feature_preprocessor=feature_preprocessor,
                pca_components=pca_components,
                normalization=normalization,
                normalization_scope=normalization_scope,
                baseline_window=baseline_window,
                source_select_top_k=source_select_top_k,
                source_select_metric=source_select_metric,
                selected_train_windows=train_windows_for_test,
                test_window=test_window,
                times=np.asarray(dataset.times, dtype=float),
            )
            model_hash = stable_hash(
                {
                    **common,
                    "max_iter": max_iter,
                    "tune_hyperparameters": tune_hyperparameters,
                    "tuning_cv_splits": tuning_cv_splits,
                    "tuning_scoring": tuning_scoring,
                    "tuning_c_grid": tuple(tuning_c_grid),
                    "best_params": [item.get("best_params", "") for item in tuning_metadata_items],
                }
            )
            metric_rows.append(
                {
                    **common,
                    "accuracy": accuracy_score(labels[test_indices], predictions),
                    "balanced_accuracy": balanced_accuracy_score(labels[test_indices], predictions),
                    "top2_accuracy": _top_k_accuracy(probabilities, labels[test_indices], k=2),
                    "top3_accuracy": _top_k_accuracy(probabilities, labels[test_indices], k=3),
                    "log_loss": log_loss(labels[test_indices], probabilities, labels=classes),
                    "brier": brier_score_multiclass(probabilities, labels[test_indices]),
                    "ece": expected_calibration_error(probabilities, labels[test_indices]),
                    "n_train": int(len(train_indices)),
                    "n_test": int(len(test_indices)),
                    "n_classes": int(len(classes)),
                    "class_names": "|".join(map(str, encoder.classes_)),
                    "preprocessing_hash": preprocessing_hash,
                    "model_hash": model_hash,
                }
            )
            _append_observation_rows(
                observation_rows,
                metadata=metadata,
                test_indices=test_indices,
                probabilities=probabilities,
                labels=labels,
                class_names=encoder.classes_,
                common=common,
                preprocessing_hash=preprocessing_hash,
                model_hash=model_hash,
                group_column=group_column,
            )

    results = pd.DataFrame(metric_rows)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(summary_out, index=False)
    if observations_out is not None:
        observations_out.parent.mkdir(parents=True, exist_ok=True)
        ProbabilityObservationTable(pd.DataFrame(observation_rows)).standardized(
            defaults={"backend": "sklearn", "split_id": f"loso:{group_column}", "seed": 13, "preprocessing_hash": preprocessing_hash}
        ).to_csv(observations_out)
    if source_scores_out is not None and source_score_rows:
        source_scores_out.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(source_score_rows, ignore_index=True).to_csv(source_scores_out, index=False)
    _write_provenance_sidecars(
        config,
        config_path=config_path,
        config_dir=config_path.parent,
        output_paths=[path for path in (summary_out, observations_out, source_scores_out) if path is not None],
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run leave-one-subject-out source-only time decoding from a NeuRepTrace dataset config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a dotted config key, e.g. --set loso.source_select_top_k=5.")
    parser.add_argument("--print-effective-config", action="store_true", help="Print the effective config and exit without decoding.")
    parser.add_argument("--no-provenance", action="store_true", help="Do not write .provenance.json sidecars next to output CSVs.")
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(args.config), args.overrides)
    if args.print_effective_config:
        print(json.dumps(effective_config(config, base_dir=args.config.parent), indent=2, sort_keys=True, default=str))
        return 0
    results = run_loso_time_decode(args.config, overrides=args.overrides, write_provenance=not args.no_provenance)
    print(f"Wrote {len(results)} LOSO rows from {args.config}")
    if not results.empty:
        pooled = results.groupby("time")["balanced_accuracy"].mean()
        best_time = float(pooled.idxmax())
        print(f"Best mean balanced_accuracy: {pooled.loc[best_time]:.4f} at {best_time:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

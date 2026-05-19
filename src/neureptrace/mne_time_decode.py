from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
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
    normalize_tuning_scoring,
    parse_c_grid,
    predict_emission_probabilities,
    time_windows,
)
from neureptrace.metrics import (
    brier_score_multiclass,
    confusion_counts,
    expected_calibration_error,
    per_class_accuracy,
    rank_class_scores,
    reliability_bins,
)
from neureptrace.observations import ProbabilityObservationTable, stable_hash

EMISSION_RUN_CHOICES = (*EMISSION_MODE_CHOICES, "both")
FEATURE_PREPROCESSOR_RUN_CHOICES = (*FEATURE_PREPROCESSOR_CHOICES, "pca-whiten", "anova-select", "select-percentile")
RESULT_SELECTION_METRIC_CHOICES = (
    "accuracy",
    "top_2_accuracy",
    "top_3_accuracy",
    "mean_true_label_rank",
    "median_true_label_rank",
    "log_loss",
    "brier",
    "ece",
)
RESULT_SELECTION_MINIMIZE_METRICS = {"log_loss", "brier", "ece", "mean_true_label_rank", "median_true_label_rank"}
INPUT_FORMAT_CHOICES = ("mne-epochs", "fieldtrip-mat")
TimeWindow = tuple[int, int, float]
TemporalTrainWindow = tuple[float, float]
DEFAULT_RANKING_TOP_K = (2, 3)
DEFAULT_RANKING_ROW_TOP_K = 3


def _add_subject(row: dict, subject: str | None) -> dict:
    if subject is not None:
        row = {"subject": subject, **row}
    return row


def _parse_positive_int_list(
    value: Sequence[int] | str | None,
    *,
    default: Sequence[int],
) -> tuple[int, ...]:
    if value is None:
        values = tuple(default)
    elif isinstance(value, str):
        values = tuple(part.strip() for part in value.split(",") if part.strip())
        if not values:
            values = tuple(default)
    else:
        values = tuple(value)

    parsed = tuple(dict.fromkeys(int(item) for item in values))
    if not parsed:
        raise ValueError("At least one top-k value must be provided.")
    if any(item < 1 for item in parsed):
        raise ValueError("top-k values must be positive integers.")
    return parsed


def _ranking_diagnostics(
    probabilities: np.ndarray,
    test_labels: np.ndarray,
    *,
    classes: np.ndarray,
    class_names: np.ndarray,
    ranking_top_k: Sequence[int],
    ranking_row_top_k: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    ranking = rank_class_scores(
        probabilities,
        classes,
        test_labels,
        top_k=ranking_top_k,
        row_top_k=ranking_row_top_k,
        class_column="label",
    )
    summary: dict[str, object] = {
        "mean_true_label_rank": ranking["mean_true_label_rank"],
        "median_true_label_rank": ranking["median_true_label_rank"],
    }
    summary.update(
        {f"top_{top_k}_accuracy": value for top_k, value in ranking["top_k_accuracy"].items()}
    )

    row_diagnostics: list[dict[str, object]] = []
    for rank_row in ranking["rows"]:
        row: dict[str, object] = {
            "true_label_rank": rank_row.get("true_label_rank", np.nan),
            "true_label_score": rank_row.get("true_label_score", np.nan),
        }
        for position in range(1, ranking_row_top_k + 1):
            label_key = f"rank{position}_label"
            score_key = f"rank{position}_score"
            if label_key not in rank_row:
                continue
            label = int(rank_row[label_key])
            row[label_key] = label
            row[f"rank{position}_class"] = str(class_names[label])
            row[score_key] = float(rank_row[score_key])
        row_diagnostics.append(row)
    return summary, row_diagnostics


def _available_columns(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(column for column in columns if column in frame.columns)


def _write_prediction_diagnostic_tables(
    observation_rows: Sequence[dict],
    *,
    confusion_out_path: Path | None,
    per_class_out_path: Path | None,
) -> None:
    if confusion_out_path is None and per_class_out_path is None:
        return

    observations = pd.DataFrame(observation_rows)
    group_columns = _available_columns(
        observations,
        (
            "subject",
            "fold",
            "decoder",
            "emission_mode",
            "feature_preprocessor",
            "pca_components",
            "temporal_mode",
            "train_time",
            "test_time",
            "time",
        ),
    )

    if confusion_out_path is not None:
        confusion_out_path.parent.mkdir(parents=True, exist_ok=True)
        if observations.empty:
            confusion = pd.DataFrame(columns=[*group_columns, "true_label", "predicted_label", "count"])
        else:
            confusion = confusion_counts(
                observations,
                true_column="true_class",
                predicted_column="predicted_class",
                group_columns=group_columns,
            )
        confusion.to_csv(confusion_out_path, index=False)

    if per_class_out_path is not None:
        per_class_out_path.parent.mkdir(parents=True, exist_ok=True)
        if observations.empty:
            per_class = pd.DataFrame(columns=[*group_columns, "true_label", "n_trials", "n_correct", "accuracy"])
        else:
            per_class = per_class_accuracy(
                observations,
                true_column="true_class",
                predicted_column="predicted_class",
                group_columns=group_columns,
            )
        per_class.to_csv(per_class_out_path, index=False)


def _default_observation_path(out_path: Path) -> Path:
    """Return the canonical probability-observation path paired with a metric CSV."""
    return out_path.with_name(f"{out_path.stem}_observations.csv")


def _resolve_observation_out_path(out_path: Path, observation_out_path: Path | None, no_observations: bool) -> Path | None:
    """Resolve CLI observation output behavior while keeping the Python API explicit."""
    return None if no_observations else observation_out_path or _default_observation_path(out_path)


def _load_epochs_and_metadata(
    epochs_path: Path,
    metadata_csv: Path | None,
    *,
    input_format: str = "mne-epochs",
    fieldtrip_root_path: Sequence[str | int] | str | None = None,
    fieldtrip_label_base: int | None = 1,
    fieldtrip_trim_overlong_labels: bool = True,
    fieldtrip_ch_type: str = "grad",
) -> tuple[mne.Epochs, pd.DataFrame]:
    if input_format == "mne-epochs":
        epochs = mne.read_epochs(epochs_path, preload=True, verbose="error")
        metadata = epochs.metadata.copy() if epochs.metadata is not None else None
    elif input_format == "fieldtrip-mat":
        from neureptrace.fieldtrip_mat import FieldTripRawMatConfig, load_fieldtrip_raw_mat_epochs

        epochs, metadata = load_fieldtrip_raw_mat_epochs(
            epochs_path,
            config=FieldTripRawMatConfig(
                label_base=fieldtrip_label_base,
                trim_overlong_labels=fieldtrip_trim_overlong_labels,
                ch_type=fieldtrip_ch_type,
            ),
            root_path=fieldtrip_root_path,
        )
    else:
        raise ValueError(f"Unsupported input format: {input_format!r}. Supported formats: {', '.join(INPUT_FORMAT_CHOICES)}")

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


def _features_for_window(data: np.ndarray, window: TimeWindow) -> np.ndarray:
    start, stop, _center = window
    return data[:, :, start:stop].reshape(data.shape[0], -1)


def _probability_average(probability_sum: np.ndarray, n_models: int) -> np.ndarray:
    probabilities = probability_sum / float(n_models)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Averaged probabilities must have positive row sums.")
    return probabilities / row_sums


def _estimator_classes(model) -> np.ndarray | None:
    """Return the class order used by an estimator's probability columns."""

    classes = getattr(model, "classes_", None)
    if classes is not None:
        return np.asarray(classes).ravel()

    best_estimator = getattr(model, "best_estimator_", None)
    if best_estimator is not None:
        classes = _estimator_classes(best_estimator)
        if classes is not None:
            return classes

    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            classes = _estimator_classes(step)
            if classes is not None:
                return classes

    return None


def _align_probabilities_to_classes(
    probabilities: np.ndarray,
    *,
    model,
    classes: np.ndarray,
) -> np.ndarray:
    """Align estimator probability columns to the global encoded class order."""

    probabilities = np.asarray(probabilities, dtype=float)
    classes = np.asarray(classes).ravel()
    if probabilities.ndim != 2:
        raise ValueError("Predicted probabilities must be a two-dimensional matrix.")

    estimator_classes = _estimator_classes(model)
    if probabilities.shape[1] == classes.size and np.array_equal(estimator_classes, classes):
        return probabilities

    if estimator_classes is None:
        if probabilities.shape[1] == classes.size:
            return probabilities
        raise ValueError(
            "Cannot align probability columns because the fitted estimator does "
            "not expose classes_ and the probability width differs from the global class count."
        )
    if estimator_classes.shape[0] != probabilities.shape[1]:
        raise ValueError(
            "Estimator class count does not match predicted probability columns: "
            f"{estimator_classes.shape[0]} != {probabilities.shape[1]}."
        )

    class_to_column = {class_label: column for column, class_label in enumerate(classes.tolist())}
    aligned = np.zeros((probabilities.shape[0], classes.size), dtype=float)
    for source_column, class_label in enumerate(estimator_classes.tolist()):
        target_column = class_to_column.get(class_label)
        if target_column is None:
            raise ValueError(f"Estimator produced probability column for unknown class {class_label!r}.")
        aligned[:, target_column] = probabilities[:, source_column]

    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Aligned probabilities must have positive row sums.")
    return aligned / row_sums


def _predict_aligned_emission_probabilities(
    model,
    features: np.ndarray,
    *,
    emission_mode: str,
    classes: np.ndarray,
) -> np.ndarray:
    """Predict emissions and align columns to the global LabelEncoder classes."""

    probabilities = predict_emission_probabilities(
        model,
        features,
        emission_mode=emission_mode,
    )
    return _align_probabilities_to_classes(probabilities, model=model, classes=classes)


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
    """Return the best time index for a metric aggregated over folds."""
    if metric not in RESULT_SELECTION_METRIC_CHOICES:
        raise ValueError(f"Unknown selection metric '{metric}'.")
    if metric in RESULT_SELECTION_MINIMIZE_METRICS:
        return float(time_summary[metric].idxmin())
    return float(time_summary[metric].idxmax())


def _metric_value(labels: np.ndarray, probabilities: np.ndarray, classes: np.ndarray, metric: str) -> float:
    predictions = probabilities.argmax(axis=1)
    if metric == "accuracy":
        return float(accuracy_score(labels, predictions))
    if metric == "log_loss":
        return float(log_loss(labels, probabilities, labels=classes))
    if metric == "brier":
        return float(brier_score_multiclass(probabilities, labels))
    if metric == "ece":
        return float(expected_calibration_error(probabilities, labels))
    raise ValueError(f"Unknown temporal selection metric '{metric}'.")


def _rank_temporal_scores(scores: list[dict[str, object]], metric: str) -> list[dict[str, object]]:
    reverse = metric not in RESULT_SELECTION_MINIMIZE_METRICS
    return sorted(
        scores,
        key=lambda score: (float(score["score"]), -float(score["center"])),
        reverse=reverse,
    )


def _compact_float(value: float) -> str:
    return f"{float(value):.9g}"


def _temporal_selection_metadata(
    *,
    temporal_selection_window: TemporalTrainWindow | None,
    temporal_selection_metric: str,
    temporal_selection_cv_splits: int,
    temporal_selection_top_k: int,
    selected_train_windows: list[TimeWindow] | None = None,
    scores: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if temporal_selection_window is None:
        return {
            "temporal_selection_window_start": "",
            "temporal_selection_window_stop": "",
            "temporal_selection_metric": "",
            "temporal_selection_cv_splits": "",
            "temporal_selection_top_k": "",
            "temporal_selected_train_times": "",
            "temporal_selection_scores": "",
        }
    selected_centers = [] if selected_train_windows is None else [window[2] for window in selected_train_windows]
    return {
        "temporal_selection_window_start": temporal_selection_window[0],
        "temporal_selection_window_stop": temporal_selection_window[1],
        "temporal_selection_metric": temporal_selection_metric,
        "temporal_selection_cv_splits": temporal_selection_cv_splits,
        "temporal_selection_top_k": temporal_selection_top_k,
        "temporal_selected_train_times": "|".join(_compact_float(center) for center in selected_centers),
        "temporal_selection_scores": "" if not scores else "|".join(
            f"{_compact_float(float(score['center']))}:{_compact_float(float(score['score']))}" for score in scores
        ),
    }


def _select_temporal_train_windows_nested_cv(
    *,
    feature_cache: dict[TimeWindow, np.ndarray],
    candidate_windows: list[TimeWindow],
    labels: np.ndarray,
    groups: np.ndarray | None,
    train_idx: np.ndarray,
    classes: np.ndarray,
    decoder_name: str,
    current_emission_mode: str,
    max_iter: int,
    feature_preprocessor_name: str,
    pca_components_value: int | float | None,
    tune_hyperparameters: bool,
    tuning_cv_splits: int,
    tuning_scoring: str,
    tuning_c_grid_values: Sequence[float],
    temporal_selection_cv_splits: int,
    temporal_selection_metric: str,
    temporal_selection_top_k: int,
) -> tuple[list[TimeWindow], list[dict[str, object]]]:
    if temporal_selection_top_k < 1:
        raise ValueError("temporal_selection_top_k must be at least 1.")
    if temporal_selection_cv_splits < 2:
        raise ValueError("temporal_selection_cv_splits must be at least 2.")
    if not candidate_windows:
        raise ValueError("At least one temporal-selection candidate window is required.")

    train_idx = np.asarray(train_idx)
    inner_groups = None if groups is None else groups[train_idx]
    inner_splits = list(make_tuning_cross_validator(labels[train_idx], inner_groups, temporal_selection_cv_splits))
    if not inner_splits:
        raise ValueError("Temporal selection requires at least one valid inner-CV split.")

    window_scores: list[dict[str, object]] = []
    for candidate_window in candidate_windows:
        features = feature_cache[candidate_window]
        fold_scores: list[float] = []
        for inner_train_local, inner_validation_local in inner_splits:
            inner_train_idx = train_idx[inner_train_local]
            inner_validation_idx = train_idx[inner_validation_local]
            inner_tuning_cv = (
                make_tuning_cross_validator(
                    labels[inner_train_idx],
                    None if groups is None else groups[inner_train_idx],
                    tuning_cv_splits,
                )
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
                tuning_cv=inner_tuning_cv,
                tuning_scoring=tuning_scoring,
                tuning_c_grid=tuning_c_grid_values,
            )
            model.fit(features[inner_train_idx], labels[inner_train_idx])
            probabilities = _predict_aligned_emission_probabilities(
                model,
                features[inner_validation_idx],
                emission_mode=current_emission_mode,
                classes=classes,
            )
            fold_scores.append(
                _metric_value(
                    labels[inner_validation_idx],
                    probabilities,
                    classes,
                    temporal_selection_metric,
                )
            )
        window_scores.append(
            {
                "center": float(candidate_window[2]),
                "score": float(np.mean(fold_scores)),
                "scores": [float(score) for score in fold_scores],
            }
        )

    ranked_scores = _rank_temporal_scores(window_scores, temporal_selection_metric)
    score_by_center = {float(score["center"]): score for score in ranked_scores}
    selected_windows = [
        candidate_window
        for candidate_window in candidate_windows
        if float(candidate_window[2]) in {float(score["center"]) for score in ranked_scores[:temporal_selection_top_k]}
    ]
    selected_windows = sorted(
        selected_windows,
        key=lambda window: ranked_scores.index(score_by_center[float(window[2])]),
    )
    return selected_windows, ranked_scores


def _model_hash(
    *,
    decoder_name: str,
    emission_mode: str,
    max_iter: int,
    feature_preprocessor: str,
    pca_components: int | float | None,
    temporal_mode: str,
    temporal_train_window: TemporalTrainWindow | None,
    temporal_selection_window: TemporalTrainWindow | None = None,
    temporal_selection_metric: str | None = None,
    temporal_selection_cv_splits: int | None = None,
    temporal_selection_top_k: int | None = None,
    train_window_centers: list[float] | None = None,
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int | None = None,
    tuning_scoring: str | None = None,
    tuning_c_grid: Sequence[float] | None = None,
    tuning_metadata: dict[str, object] | None = None,
) -> str:
    payload: dict[str, object] = {
        "backend": "sklearn",
        "decoder": decoder_name,
        "emission_mode": emission_mode,
        "max_iter": max_iter,
        "feature_preprocessor": feature_preprocessor,
        "pca_components": pca_components,
        "temporal_mode": temporal_mode,
        "temporal_train_window": temporal_train_window,
        "temporal_selection_window": temporal_selection_window,
        "temporal_selection_metric": temporal_selection_metric,
        "temporal_selection_cv_splits": temporal_selection_cv_splits,
        "temporal_selection_top_k": temporal_selection_top_k,
        "train_window_centers": train_window_centers,
    }
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
    collect_prediction_diagnostics: bool = False,
    ranking_top_k: Sequence[int] = DEFAULT_RANKING_TOP_K,
    ranking_row_top_k: int = DEFAULT_RANKING_ROW_TOP_K,
    tuning_metadata: dict[str, object] | None = None,
) -> None:
    tuning_metadata = {} if tuning_metadata is None else tuning_metadata
    start, stop, center = time_window
    predictions = probabilities.argmax(axis=1)
    ranking_summary, ranking_rows = _ranking_diagnostics(
        probabilities,
        test_labels,
        classes=classes,
        class_names=class_names,
        ranking_top_k=ranking_top_k,
        ranking_row_top_k=ranking_row_top_k,
    )
    row = {
        "fold": fold,
        "decoder": decoder_name,
        "emission_mode": emission_mode,
        "feature_preprocessor": feature_preprocessor_name,
        "pca_components": "" if pca_components_value is None else pca_components_value,
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
        "accuracy": accuracy_score(test_labels, predictions),
        "log_loss": log_loss(test_labels, probabilities, labels=classes),
        "brier": brier_score_multiclass(probabilities, test_labels),
        "ece": expected_calibration_error(probabilities, test_labels),
        **ranking_summary,
        "n_train": n_train,
        "n_test": len(test_idx),
        "n_classes": len(classes),
        "class_names": "|".join(map(str, class_names)),
    }
    row.update(tuning_metadata)
    rows.append(_add_subject(row, subject))

    if calibration_out_path is not None:
        for bin_row in reliability_bins(probabilities, test_labels, n_bins=calibration_bins):
            calibration_row = {
                "fold": fold,
                "decoder": decoder_name,
                "emission_mode": emission_mode,
                "feature_preprocessor": feature_preprocessor_name,
                "pca_components": "" if pca_components_value is None else pca_components_value,
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
                **bin_row,
            }
            calibration_row.update(tuning_metadata)
            calibration_rows.append(_add_subject(calibration_row, subject))
    if observation_out_path is not None or collect_prediction_diagnostics:
        for local_position, filtered_index in enumerate(test_idx):
            true_label = int(test_labels[local_position])
            predicted_label = int(predictions[local_position])
            observation = {
                "fold": fold,
                "split_id": split_id,
                "seed": 13,
                "decoder": decoder_name,
                "backend": "sklearn",
                "emission_mode": emission_mode,
                "feature_preprocessor": feature_preprocessor_name,
                "pca_components": "" if pca_components_value is None else pca_components_value,
                "temporal_mode": temporal_mode,
                "temporal_train_window_start": "" if temporal_train_window is None else temporal_train_window[0],
                "temporal_train_window_stop": "" if temporal_train_window is None else temporal_train_window[1],
                "train_time": train_time,
                "test_time": center,
                "time": center,
                "train_window_start": train_window_start,
                "train_window_stop": train_window_stop,
                "n_train_windows": n_train_windows,
                "window_start": float(epochs.times[start]),
                "window_stop": float(epochs.times[stop - 1]),
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
                **ranking_rows[local_position],
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
    fieldtrip_root_path: Sequence[str | int] | str | None = None,
    fieldtrip_label_base: int | None = 1,
    fieldtrip_trim_overlong_labels: bool = True,
    fieldtrip_ch_type: str = "grad",
    group_column: str | None = None,
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
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int = 3,
    tuning_scoring: str = "accuracy",
    tuning_c_grid: Sequence[float] | str | None = None,
    calibration_out_path: Path | None = None,
    calibration_bins: int = 10,
    observation_out_path: Path | None = None,
    subject: str | None = None,
    temporal_train_window: tuple[float, float] | None = None,
    temporal_selection_window: tuple[float, float] | None = None,
    temporal_selection_metric: str = "accuracy",
    temporal_selection_cv_splits: int = 3,
    temporal_selection_top_k: int = 1,
    ranking_top_k: Sequence[int] | str | None = DEFAULT_RANKING_TOP_K,
    ranking_row_top_k: int = DEFAULT_RANKING_ROW_TOP_K,
    confusion_out_path: Path | None = None,
    per_class_out_path: Path | None = None,
) -> pd.DataFrame:
    """Load epochs input, run time-resolved decoding, and save metrics as CSV."""
    epochs, metadata = _load_epochs_and_metadata(
        epochs_path,
        metadata_csv,
        input_format=input_format,
        fieldtrip_root_path=fieldtrip_root_path,
        fieldtrip_label_base=fieldtrip_label_base,
        fieldtrip_trim_overlong_labels=fieldtrip_trim_overlong_labels,
        fieldtrip_ch_type=fieldtrip_ch_type,
    )
    return run_time_resolved_decode_from_epochs(
        epochs=epochs,
        metadata=metadata,
        label_column=label_column,
        out_path=out_path,
        group_column=group_column,
        picks=picks,
        tmin=tmin,
        tmax=tmax,
        window_ms=window_ms,
        step_ms=step_ms,
        n_splits=n_splits,
        max_iter=max_iter,
        decoder=decoder,
        emission_mode=emission_mode,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv_splits=tuning_cv_splits,
        tuning_scoring=tuning_scoring,
        tuning_c_grid=tuning_c_grid,
        calibration_out_path=calibration_out_path,
        calibration_bins=calibration_bins,
        observation_out_path=observation_out_path,
        subject=subject,
        temporal_train_window=temporal_train_window,
        temporal_selection_window=temporal_selection_window,
        temporal_selection_metric=temporal_selection_metric,
        temporal_selection_cv_splits=temporal_selection_cv_splits,
        temporal_selection_top_k=temporal_selection_top_k,
        ranking_top_k=ranking_top_k,
        ranking_row_top_k=ranking_row_top_k,
        confusion_out_path=confusion_out_path,
        per_class_out_path=per_class_out_path,
    )


def run_time_resolved_decode_from_epochs(
    epochs: mne.Epochs,
    metadata: pd.DataFrame,
    label_column: str,
    out_path: Path,
    *,
    group_column: str | None = None,
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
    tune_hyperparameters: bool = False,
    tuning_cv_splits: int = 3,
    tuning_scoring: str = "accuracy",
    tuning_c_grid: Sequence[float] | str | None = None,
    calibration_out_path: Path | None = None,
    calibration_bins: int = 10,
    observation_out_path: Path | None = None,
    subject: str | None = None,
    temporal_train_window: tuple[float, float] | None = None,
    temporal_selection_window: tuple[float, float] | None = None,
    temporal_selection_metric: str = "accuracy",
    temporal_selection_cv_splits: int = 3,
    temporal_selection_top_k: int = 1,
    ranking_top_k: Sequence[int] | str | None = DEFAULT_RANKING_TOP_K,
    ranking_row_top_k: int = DEFAULT_RANKING_ROW_TOP_K,
    confusion_out_path: Path | None = None,
    per_class_out_path: Path | None = None,
) -> pd.DataFrame:
    """Run time-resolved decoding on preloaded epochs and metadata.

    File-specific loaders normalize their inputs to ``epochs`` plus a
    trial-aligned metadata frame and call this function. If
    ``temporal_train_window`` is set, models are trained on every decoding
    window whose center lies in that interval and are evaluated at every test
    time. If ``temporal_selection_window`` is set, train-time windows inside
    that interval are selected independently inside each outer train fold using
    nested cross-validation on training trials only.
    """
    if len(metadata) != len(epochs):
        raise ValueError(
            f"Metadata row count ({len(metadata)}) does not match epochs ({len(epochs)})."
        )
    metadata = metadata.copy().reset_index(drop=True)
    decoder_name = normalize_decoder_name(decoder)
    emission_modes = list(EMISSION_MODE_CHOICES) if emission_mode == "both" else [normalize_emission_mode(emission_mode)]
    feature_preprocessor_name = normalize_feature_preprocessor(feature_preprocessor)
    if feature_preprocessor_name == "none" and pca_components is not None:
        raise ValueError(
            "pca_components can only be set when feature_preprocessor is 'pca', 'pca_whiten', or 'anova_select'."
        )
    if feature_preprocessor_name == "anova_select":
        pca_components_value = normalize_anova_select_percentile(pca_components)
    elif feature_preprocessor_name != "none":
        pca_components_value = normalize_pca_components(pca_components)
    else:
        pca_components_value = None
    tuning_scoring = normalize_tuning_scoring(tuning_scoring)
    tuning_c_grid_values = parse_c_grid(tuning_c_grid)
    temporal_selection_metric = str(temporal_selection_metric).strip()
    if temporal_selection_metric not in RESULT_SELECTION_METRIC_CHOICES:
        raise ValueError(f"Unknown temporal_selection_metric '{temporal_selection_metric}'.")
    if temporal_selection_top_k < 1:
        raise ValueError("temporal_selection_top_k must be at least 1.")
    if temporal_selection_cv_splits < 2:
        raise ValueError("temporal_selection_cv_splits must be at least 2.")
    normalized_temporal_train_window = _normalize_temporal_train_window(temporal_train_window)
    normalized_temporal_selection_window = _normalize_temporal_train_window(temporal_selection_window)
    if normalized_temporal_train_window is not None and normalized_temporal_selection_window is not None:
        raise ValueError("temporal_train_window and temporal_selection_window are mutually exclusive.")
    ranking_top_k_values = _parse_positive_int_list(ranking_top_k, default=DEFAULT_RANKING_TOP_K)
    ranking_row_top_k = int(ranking_row_top_k)
    if ranking_row_top_k < 0:
        raise ValueError("ranking_row_top_k must be non-negative.")
    collect_prediction_diagnostics = (
        observation_out_path is not None or confusion_out_path is not None or per_class_out_path is not None
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
    if normalized_temporal_selection_window is not None:
        temporal_mode = "nested_train_window_selection"
    elif normalized_temporal_train_window is not None:
        temporal_mode = "train_window_ensemble"
    else:
        temporal_mode = "same_time"
    preprocessing_hash = stable_hash(
        {
            "picks": picks,
            "tmin": tmin,
            "tmax": tmax,
            "window_ms": window_ms,
            "step_ms": step_ms,
            "feature_preprocessor": feature_preprocessor_name,
            "pca_components": pca_components_value,
            "temporal_train_window": normalized_temporal_train_window,
            "temporal_selection_window": normalized_temporal_selection_window,
            "temporal_selection_metric": temporal_selection_metric,
            "temporal_selection_cv_splits": temporal_selection_cv_splits,
            "temporal_selection_top_k": temporal_selection_top_k,
        }
    )
    default_model_hash = _model_hash(
        decoder_name=decoder_name,
        emission_mode=emission_mode,
        max_iter=max_iter,
        feature_preprocessor=feature_preprocessor_name,
        pca_components=pca_components_value,
        temporal_mode=temporal_mode,
        temporal_train_window=normalized_temporal_train_window,
        temporal_selection_window=normalized_temporal_selection_window,
        temporal_selection_metric=temporal_selection_metric,
        temporal_selection_cv_splits=temporal_selection_cv_splits,
        temporal_selection_top_k=temporal_selection_top_k,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv_splits=tuning_cv_splits,
        tuning_scoring=tuning_scoring,
        tuning_c_grid=tuning_c_grid_values,
    )

    data = epochs.get_data(copy=False)
    classes = np.arange(len(encoder.classes_))
    rows = []
    calibration_rows = []
    observation_rows = []
    windows = time_windows(epochs.times, window_ms=window_ms, step_ms=step_ms)
    selected_train_windows = _select_temporal_train_windows(windows, normalized_temporal_train_window)
    candidate_selection_windows = _select_temporal_train_windows(windows, normalized_temporal_selection_window)
    if candidate_selection_windows is not None and temporal_selection_top_k > len(candidate_selection_windows):
        raise ValueError(
            "temporal_selection_top_k cannot exceed the number of temporal-selection candidate windows "
            f"({len(candidate_selection_windows)})."
        )
    splits = list(make_cross_validator(labels, groups, n_splits))

    if selected_train_windows is None and candidate_selection_windows is None:
        for time_window in windows:
            features = _features_for_window(data, time_window)
            start, stop, center = time_window
            for fold, (train_idx, test_idx) in enumerate(splits):
                test_labels = labels[test_idx]
                for current_emission_mode in emission_modes:
                    tuning_cv = (
                        make_tuning_cross_validator(labels[train_idx], None if groups is None else groups[train_idx], tuning_cv_splits)
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
                    model.fit(features[train_idx], labels[train_idx])

                    probabilities = _predict_aligned_emission_probabilities(
                        model,
                        features[test_idx],
                        emission_mode=current_emission_mode,
                        classes=classes,
                    )
                    tuning_metadata = _tuning_metadata(
                        model,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                    )
                    row_metadata = {
                        **tuning_metadata,
                        **_temporal_selection_metadata(
                            temporal_selection_window=None,
                            temporal_selection_metric=temporal_selection_metric,
                            temporal_selection_cv_splits=temporal_selection_cv_splits,
                            temporal_selection_top_k=temporal_selection_top_k,
                        ),
                    }
                    current_model_hash = _model_hash(
                        decoder_name=decoder_name,
                        emission_mode=current_emission_mode,
                        max_iter=max_iter,
                        feature_preprocessor=feature_preprocessor_name,
                        pca_components=pca_components_value,
                        temporal_mode=temporal_mode,
                        temporal_train_window=None,
                        temporal_selection_window=None,
                        train_window_centers=[center],
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                        tuning_metadata=tuning_metadata,
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
                        collect_prediction_diagnostics=collect_prediction_diagnostics,
                        ranking_top_k=ranking_top_k_values,
                        ranking_row_top_k=ranking_row_top_k,
                        subject=subject,
                        tuning_metadata=row_metadata,
                    )
    else:
        feature_cache = {time_window: _features_for_window(data, time_window) for time_window in windows}
        for fold, (train_idx, test_idx) in enumerate(splits):
            test_labels = labels[test_idx]
            for current_emission_mode in emission_modes:
                if candidate_selection_windows is not None:
                    current_train_windows, selection_scores = _select_temporal_train_windows_nested_cv(
                        feature_cache=feature_cache,
                        candidate_windows=candidate_selection_windows,
                        labels=labels,
                        groups=groups,
                        train_idx=train_idx,
                        classes=classes,
                        decoder_name=decoder_name,
                        current_emission_mode=current_emission_mode,
                        max_iter=max_iter,
                        feature_preprocessor_name=feature_preprocessor_name,
                        pca_components_value=pca_components_value,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid_values=tuning_c_grid_values,
                        temporal_selection_cv_splits=temporal_selection_cv_splits,
                        temporal_selection_metric=temporal_selection_metric,
                        temporal_selection_top_k=temporal_selection_top_k,
                    )
                else:
                    current_train_windows = selected_train_windows or []
                    selection_scores = []

                train_time, train_window_start, train_window_stop = _train_window_summary(epochs, current_train_windows)
                train_window_centers = [window[2] for window in current_train_windows]
                tuning_cv = (
                    make_tuning_cross_validator(labels[train_idx], None if groups is None else groups[train_idx], tuning_cv_splits)
                    if tune_hyperparameters
                    else 3
                )
                fitted_models = []
                probability_sums = {
                    time_window: np.zeros((len(test_idx), len(classes)), dtype=float)
                    for time_window in windows
                }
                for train_window in current_train_windows:
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
                    model.fit(train_features[train_idx], labels[train_idx])
                    fitted_models.append(model)
                    for test_window in windows:
                        probability_sums[test_window] += _predict_aligned_emission_probabilities(
                            model,
                            feature_cache[test_window][test_idx],
                            emission_mode=current_emission_mode,
                            classes=classes,
                        )

                tuning_metadata = _tuning_metadata(
                    fitted_models,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                )
                row_metadata = {
                    **tuning_metadata,
                    **_temporal_selection_metadata(
                        temporal_selection_window=normalized_temporal_selection_window,
                        temporal_selection_metric=temporal_selection_metric,
                        temporal_selection_cv_splits=temporal_selection_cv_splits,
                        temporal_selection_top_k=temporal_selection_top_k,
                        selected_train_windows=current_train_windows,
                        scores=selection_scores,
                    ),
                }
                current_model_hash = _model_hash(
                    decoder_name=decoder_name,
                    emission_mode=current_emission_mode,
                    max_iter=max_iter,
                    feature_preprocessor=feature_preprocessor_name,
                    pca_components=pca_components_value,
                    temporal_mode=temporal_mode,
                    temporal_train_window=normalized_temporal_train_window,
                    temporal_selection_window=normalized_temporal_selection_window,
                    temporal_selection_metric=temporal_selection_metric,
                    temporal_selection_cv_splits=temporal_selection_cv_splits,
                    temporal_selection_top_k=temporal_selection_top_k,
                    train_window_centers=train_window_centers,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                    tuning_metadata=tuning_metadata,
                )
                for test_window in windows:
                    probabilities = _probability_average(probability_sums[test_window], len(current_train_windows))
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
                        n_train_windows=len(current_train_windows),
                        calibration_out_path=calibration_out_path,
                        calibration_bins=calibration_bins,
                        observation_out_path=observation_out_path,
                        collect_prediction_diagnostics=collect_prediction_diagnostics,
                        ranking_top_k=ranking_top_k_values,
                        ranking_row_top_k=ranking_row_top_k,
                        subject=subject,
                        tuning_metadata=row_metadata,
                    )

    results = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    if calibration_out_path is not None:
        calibration_out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(calibration_rows).to_csv(calibration_out_path, index=False)
    if observation_out_path is not None:
        observation_table = ProbabilityObservationTable(pd.DataFrame(observation_rows)).standardized(
            defaults={
                "backend": "sklearn",
                "split_id": split_id,
                "seed": 13,
                "calibration_fold": "",
                "preprocessing_hash": preprocessing_hash,
                "model_hash": default_model_hash,
            }
        )
        observation_table.validate(profile="canonical", require_normalized=True).raise_for_errors()
        observation_table.to_csv(observation_out_path)
    _write_prediction_diagnostic_tables(
        observation_rows,
        confusion_out_path=confusion_out_path,
        per_class_out_path=per_class_out_path,
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run calibrated time-resolved decoding on an MNE Epochs FIF file or FieldTrip raw MATLAB file."
    )
    parser.add_argument("--epochs", type=Path, required=True)
    parser.add_argument("--input-format", choices=INPUT_FORMAT_CHOICES, default="mne-epochs")
    parser.add_argument(
        "--fieldtrip-root-path",
        default="data,0",
        help="MATLAB path to the FieldTrip raw struct when --input-format fieldtrip-mat is used, e.g. 'data,0'.",
    )
    parser.add_argument(
        "--fieldtrip-label-base",
        type=int,
        default=1,
        help="Subtract this value from numeric trialinfo to create the default 'condition' metadata column.",
    )
    parser.add_argument(
        "--fieldtrip-no-trim-overlong-labels",
        action="store_true",
        help="Fail instead of trimming data.label when it is longer than the trial channel count.",
    )
    parser.add_argument("--fieldtrip-ch-type", default="grad", help="MNE channel type assigned to FieldTrip trial rows.")
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metadata-csv", type=Path)
    parser.add_argument("--group-column")
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
    parser.add_argument("--tune-hyperparameters", action="store_true", help="Use nested inner-CV hyperparameter selection inside each outer train fold.")
    parser.add_argument("--tuning-cv-splits", type=int, default=3, help="Maximum number of inner CV folds for --tune-hyperparameters.")
    parser.add_argument("--tuning-scoring", choices=TUNING_SCORING_CHOICES, default="accuracy", help="Inner-CV objective for --tune-hyperparameters.")
    parser.add_argument("--selection-metric", choices=RESULT_SELECTION_METRIC_CHOICES, default="accuracy", help="Metric used only for the console 'best time' summary.")
    parser.add_argument("--top-k", default="2,3", help="Comma-separated top-k values to report, for example '2,3,5'.")
    parser.add_argument(
        "--rank-row-top-k",
        type=int,
        default=DEFAULT_RANKING_ROW_TOP_K,
        help="Number of ranked class alternatives to include in per-trial observation rows.",
    )
    parser.add_argument(
        "--tuning-c-grid",
        default=",".join(str(value) for value in parse_c_grid(None)),
        help="Comma-separated positive C values for tuned logistic regression and linear SVM.",
    )
    parser.add_argument("--calibration-out", type=Path)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument(
        "--observations-out",
        type=Path,
        help="Canonical held-out trial/time probability observation CSV. Defaults to '<out stem>_observations.csv'.",
    )
    parser.add_argument("--no-observations", action="store_true", help="Disable canonical probability-observation CSV output.")
    parser.add_argument("--confusion-out", type=Path, help="Optional true/predicted class-pair count CSV.")
    parser.add_argument("--per-class-out", type=Path, help="Optional per-class recall/accuracy CSV.")
    parser.add_argument("--subject", help="Optional subject identifier to include in output CSVs.")
    parser.add_argument(
        "--temporal-train-window",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        help=(
            "Train one model per time-window center in START..STOP seconds, "
            "evaluate each model at every test time, and average probabilities."
        ),
    )
    parser.add_argument(
        "--temporal-selection-window",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        help=(
            "Select train-time windows within START..STOP seconds by inner CV on each outer train fold, "
            "then refit the selected top-k train-time models and average probabilities at every test time."
        ),
    )
    parser.add_argument(
        "--temporal-selection-metric",
        choices=RESULT_SELECTION_METRIC_CHOICES,
        default="accuracy",
        help="Inner-CV metric used to rank candidate train-time windows for --temporal-selection-window.",
    )
    parser.add_argument(
        "--temporal-selection-cv-splits",
        type=int,
        default=3,
        help="Maximum number of inner CV folds used to select train-time windows.",
    )
    parser.add_argument(
        "--temporal-selection-top-k",
        type=int,
        default=1,
        help="Number of selected train-time windows to refit and probability-ensemble per outer fold.",
    )
    args = parser.parse_args()

    observations_out = _resolve_observation_out_path(args.out, args.observations_out, args.no_observations)

    results = run_time_resolved_decode(
        epochs_path=args.epochs,
        metadata_csv=args.metadata_csv,
        input_format=args.input_format,
        fieldtrip_root_path=args.fieldtrip_root_path,
        fieldtrip_label_base=args.fieldtrip_label_base,
        fieldtrip_trim_overlong_labels=not args.fieldtrip_no_trim_overlong_labels,
        fieldtrip_ch_type=args.fieldtrip_ch_type,
        label_column=args.label_column,
        group_column=args.group_column,
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
        tune_hyperparameters=args.tune_hyperparameters,
        tuning_cv_splits=args.tuning_cv_splits,
        tuning_scoring=args.tuning_scoring,
        tuning_c_grid=args.tuning_c_grid,
        calibration_out_path=args.calibration_out,
        calibration_bins=args.calibration_bins,
        observation_out_path=observations_out,
        subject=args.subject,
        temporal_train_window=tuple(args.temporal_train_window) if args.temporal_train_window is not None else None,
        temporal_selection_window=tuple(args.temporal_selection_window) if args.temporal_selection_window is not None else None,
        temporal_selection_metric=args.temporal_selection_metric,
        temporal_selection_cv_splits=args.temporal_selection_cv_splits,
        temporal_selection_top_k=args.temporal_selection_top_k,
        ranking_top_k=args.top_k,
        ranking_row_top_k=args.rank_row_top_k,
        confusion_out_path=args.confusion_out,
        per_class_out_path=args.per_class_out,
    )
    print(f"Wrote {args.out}")
    if observations_out is not None:
        print(f"Wrote probability observations: {observations_out}")
    if args.confusion_out is not None:
        print(f"Wrote confusion counts: {args.confusion_out}")
    if args.per_class_out is not None:
        print(f"Wrote per-class accuracy: {args.per_class_out}")
    for emission_mode_name, summary in results.groupby("emission_mode", sort=True):
        summary_metrics = [column for column in RESULT_SELECTION_METRIC_CHOICES if column in summary.columns]
        time_summary = summary.groupby("time")[summary_metrics].mean()
        if args.selection_metric not in time_summary.columns:
            available = ", ".join(time_summary.columns)
            raise ValueError(f"Selection metric '{args.selection_metric}' is unavailable. Available metrics: {available}")
        best_time = _best_time_by_metric(time_summary, args.selection_metric)
        best_value = time_summary.loc[best_time, args.selection_metric]
        direction = "lowest" if args.selection_metric in RESULT_SELECTION_MINIMIZE_METRICS else "highest"
        print(
            f"Best {emission_mode_name} mean {args.selection_metric} "
            f"({direction}): {best_value:.3f} at {best_time:.3f}s"
        )


if __name__ == "__main__":
    main()

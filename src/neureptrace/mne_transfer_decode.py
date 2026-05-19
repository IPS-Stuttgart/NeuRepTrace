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
from neureptrace.metrics import brier_score_multiclass, expected_calibration_error, reliability_bins
from neureptrace.mne_time_decode import (
    DEFAULT_BASELINE_WINDOW,
    EPOCH_NORMALIZATION_RUN_CHOICES,
    FEATURE_PREPROCESSOR_RUN_CHOICES,
    RESULT_SELECTION_METRIC_CHOICES,
    RESULT_SELECTION_MINIMIZE_METRICS,
    _add_subject,
    _align_probability_columns,
    _apply_epoch_normalization,
    _best_params_json,
    _best_scores,
    _best_time_by_metric,
    _features_for_window,
    _load_epochs_and_metadata,
    _model_hash,
    _normalize_baseline_window,
    _normalize_temporal_train_window,
    _probability_average,
    _select_temporal_train_windows,
    _train_window_summary,
)
from neureptrace.observations import ProbabilityObservationTable, stable_hash

EMISSION_RUN_CHOICES = (*EMISSION_MODE_CHOICES, "both")
TRANSFER_SPLIT_ID_PREFIX = "transfer"


def _fit_label_encoder(train_values: np.ndarray, test_values: np.ndarray) -> tuple[LabelEncoder, np.ndarray, np.ndarray]:
    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(train_values)
    train_classes = set(encoder.classes_.tolist())
    missing = sorted({value for value in pd.unique(test_values) if value not in train_classes}, key=str)
    if missing:
        raise ValueError(
            "Test metadata contains class(es) that are absent from the transfer training set: "
            + ", ".join(map(str, missing))
        )
    test_labels = encoder.transform(test_values)
    return encoder, train_labels, test_labels


def _filtered_epochs_metadata(
    epochs: mne.Epochs,
    metadata: pd.DataFrame,
    *,
    label_column: str,
) -> tuple[mne.Epochs, pd.DataFrame, np.ndarray, np.ndarray]:
    if label_column not in metadata.columns:
        raise ValueError(f"Label column '{label_column}' not found in metadata.")
    raw_labels = metadata[label_column].to_numpy()
    keep = pd.notna(raw_labels)
    original_indices = np.arange(len(raw_labels))[keep]
    return epochs[keep], metadata.loc[keep].reset_index(drop=True), raw_labels[keep], original_indices


def _validate_transfer_axes(train_epochs: mne.Epochs, test_epochs: mne.Epochs) -> None:
    if len(train_epochs.ch_names) != len(test_epochs.ch_names):
        raise ValueError(
            f"Transfer decoding requires the same channel count; training has {len(train_epochs.ch_names)} channels, "
            f"test has {len(test_epochs.ch_names)} channels."
        )
    if list(train_epochs.ch_names) != list(test_epochs.ch_names):
        raise ValueError("Transfer decoding requires identical channel order between training and test epochs.")
    if len(train_epochs.times) != len(test_epochs.times) or not np.allclose(train_epochs.times, test_epochs.times, rtol=1e-7, atol=1e-12):
        raise ValueError("Transfer decoding requires identical time axes between training and test epochs.")


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


def _append_transfer_outputs(
    *,
    rows: list[dict],
    calibration_rows: list[dict],
    observation_rows: list[dict],
    probabilities: np.ndarray,
    test_labels: np.ndarray,
    original_indices: np.ndarray,
    session_values: np.ndarray | None,
    groups: np.ndarray | None,
    group_column: str | None,
    classes: np.ndarray,
    class_names: np.ndarray,
    n_train: int,
    decoder_name: str,
    emission_mode: str,
    feature_preprocessor_name: str,
    pca_components_value: int | float | None,
    normalization_name: str,
    baseline_window: tuple[float, float],
    time_window: tuple[int, int, float],
    epochs: mne.Epochs,
    split_id: str,
    preprocessing_hash: str,
    model_hash: str,
    temporal_mode: str,
    temporal_train_window: tuple[float, float] | None,
    train_time: float,
    train_window_start: float,
    train_window_stop: float,
    n_train_windows: int,
    calibration_out_path: Path | None,
    calibration_bins: int,
    observation_out_path: Path | None,
    train_subject: str | None,
    test_subject: str | None,
    train_recording: str,
    test_recording: str,
    tuning_metadata: dict[str, object] | None = None,
) -> None:
    tuning_metadata = {} if tuning_metadata is None else tuning_metadata
    start, stop, center = time_window
    test_idx = np.arange(len(test_labels), dtype=int)
    predictions = probabilities.argmax(axis=1)
    common = {
        "fold": 0,
        "split_kind": "transfer",
        "train_recording": train_recording,
        "test_recording": test_recording,
        "train_subject": "" if train_subject is None else train_subject,
        "test_subject": "" if test_subject is None else test_subject,
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
    }
    row = {
        **common,
        "accuracy": accuracy_score(test_labels, predictions),
        "log_loss": log_loss(test_labels, probabilities, labels=classes),
        "brier": brier_score_multiclass(probabilities, test_labels),
        "ece": expected_calibration_error(probabilities, test_labels),
        "n_train": n_train,
        "n_test": len(test_idx),
        "n_classes": len(classes),
        "class_names": "|".join(map(str, class_names)),
    }
    row.update(tuning_metadata)
    rows.append(_add_subject(row, test_subject))

    if calibration_out_path is not None:
        for bin_row in reliability_bins(probabilities, test_labels, n_bins=calibration_bins):
            calibration_row = {**common, **bin_row}
            calibration_row.update(tuning_metadata)
            calibration_rows.append(_add_subject(calibration_row, test_subject))

    if observation_out_path is not None:
        for local_position, filtered_index in enumerate(test_idx):
            true_label = int(test_labels[local_position])
            predicted_label = int(predictions[local_position])
            observation = {
                **common,
                "split_id": split_id,
                "seed": 13,
                "backend": "sklearn",
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
            observation_rows.append(_add_subject(observation, test_subject))


def run_time_resolved_transfer_decode(
    train_epochs_path: Path,
    test_epochs_path: Path,
    label_column: str,
    out_path: Path,
    *,
    train_metadata_csv: Path | None = None,
    test_metadata_csv: Path | None = None,
    group_column: str | None = None,
    picks: str = "data",
    tmin: float | None = None,
    tmax: float | None = None,
    window_ms: float = 20.0,
    step_ms: float = 10.0,
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
    train_subject: str | None = None,
    test_subject: str | None = None,
    train_recording: str = "train",
    test_recording: str = "test",
    temporal_train_window: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Train time-resolved decoders on one epochs file and evaluate on another.

    This is the NeuRepTrace replacement for PyMEGDec-style main->cue and cue->main
    transfer decoding. It deliberately uses all labelled training trials and all
    labelled test trials instead of within-recording cross-validation folds.
    """

    train_epochs, train_metadata = _load_epochs_and_metadata(train_epochs_path, train_metadata_csv)
    test_epochs, test_metadata = _load_epochs_and_metadata(test_epochs_path, test_metadata_csv)
    decoder_name = normalize_decoder_name(decoder)
    emission_modes = list(EMISSION_MODE_CHOICES) if emission_mode == "both" else [normalize_emission_mode(emission_mode)]
    feature_preprocessor_name = normalize_feature_preprocessor(feature_preprocessor)
    normalization_name = normalization.replace("-", "_")
    baseline_window_value = _normalize_baseline_window(baseline_window)
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
    normalized_temporal_train_window = _normalize_temporal_train_window(temporal_train_window)

    train_epochs = train_epochs.copy().pick(picks)
    test_epochs = test_epochs.copy().pick(picks)
    if tmin is not None or tmax is not None:
        train_epochs.crop(tmin=tmin, tmax=tmax)
        test_epochs.crop(tmin=tmin, tmax=tmax)
    _validate_transfer_axes(train_epochs, test_epochs)

    train_epochs, train_metadata, train_raw_labels, train_original_indices = _filtered_epochs_metadata(
        train_epochs,
        train_metadata,
        label_column=label_column,
    )
    test_epochs, test_metadata, test_raw_labels, test_original_indices = _filtered_epochs_metadata(
        test_epochs,
        test_metadata,
        label_column=label_column,
    )
    encoder, train_labels, test_labels = _fit_label_encoder(train_raw_labels, test_raw_labels)

    train_groups = train_metadata[group_column].to_numpy() if group_column else None
    test_groups = test_metadata[group_column].to_numpy() if group_column else None
    test_session_values = test_metadata["session"].to_numpy() if "session" in test_metadata.columns else test_groups
    classes = np.arange(len(encoder.classes_))
    split_id = f"{TRANSFER_SPLIT_ID_PREFIX}:{train_recording}->{test_recording}"
    temporal_mode = "same_time_transfer" if normalized_temporal_train_window is None else "train_window_ensemble_transfer"
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
            "temporal_train_window": normalized_temporal_train_window,
            "transfer": {"train_recording": train_recording, "test_recording": test_recording},
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
    )

    train_data = _apply_epoch_normalization(
        train_epochs.get_data(copy=False),
        train_epochs.times,
        normalization_name,
        baseline_window=baseline_window_value,
    )
    test_data = _apply_epoch_normalization(
        test_epochs.get_data(copy=False),
        test_epochs.times,
        normalization_name,
        baseline_window=baseline_window_value,
    )
    rows: list[dict] = []
    calibration_rows: list[dict] = []
    observation_rows: list[dict] = []
    windows = time_windows(test_epochs.times, window_ms=window_ms, step_ms=step_ms)
    selected_train_windows = _select_temporal_train_windows(windows, normalized_temporal_train_window)

    if selected_train_windows is None:
        for time_window in windows:
            train_features = _features_for_window(train_data, time_window)
            test_features = _features_for_window(test_data, time_window)
            start, stop, center = time_window
            for current_emission_mode in emission_modes:
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
                model.fit(train_features, train_labels)
                probabilities = _align_probability_columns(
                    predict_emission_probabilities(model, test_features, emission_mode=current_emission_mode),
                    model=model,
                    classes=classes,
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
                )
                _append_transfer_outputs(
                    rows=rows,
                    calibration_rows=calibration_rows,
                    observation_rows=observation_rows,
                    probabilities=probabilities,
                    test_labels=test_labels,
                    original_indices=test_original_indices,
                    session_values=test_session_values,
                    groups=test_groups,
                    group_column=group_column,
                    classes=classes,
                    class_names=encoder.classes_,
                    n_train=len(train_labels),
                    decoder_name=decoder_name,
                    emission_mode=current_emission_mode,
                    feature_preprocessor_name=feature_preprocessor_name,
                    pca_components_value=pca_components_value,
                    normalization_name=normalization_name,
                    baseline_window=baseline_window_value,
                    time_window=time_window,
                    epochs=test_epochs,
                    split_id=split_id,
                    preprocessing_hash=preprocessing_hash,
                    model_hash=current_model_hash,
                    temporal_mode=temporal_mode,
                    temporal_train_window=normalized_temporal_train_window,
                    train_time=center,
                    train_window_start=float(train_epochs.times[start]),
                    train_window_stop=float(train_epochs.times[stop - 1]),
                    n_train_windows=1,
                    calibration_out_path=calibration_out_path,
                    calibration_bins=calibration_bins,
                    observation_out_path=observation_out_path,
                    train_subject=train_subject,
                    test_subject=test_subject,
                    train_recording=train_recording,
                    test_recording=test_recording,
                    tuning_metadata=tuning_metadata,
                )
    else:
        train_feature_cache = {time_window: _features_for_window(train_data, time_window) for time_window in windows}
        test_feature_cache = {time_window: _features_for_window(test_data, time_window) for time_window in windows}
        train_time, train_window_start, train_window_stop = _train_window_summary(train_epochs, selected_train_windows)
        train_window_centers = [window[2] for window in selected_train_windows]
        for current_emission_mode in emission_modes:
            tuning_cv = (
                make_tuning_cross_validator(train_labels, train_groups, tuning_cv_splits)
                if tune_hyperparameters
                else 3
            )
            fitted_models = []
            probability_sums = {
                time_window: np.zeros((len(test_labels), len(classes)), dtype=float)
                for time_window in windows
            }
            for train_window in selected_train_windows:
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
                model.fit(train_feature_cache[train_window], train_labels)
                fitted_models.append(model)
                for test_window in windows:
                    probability_sums[test_window] += _align_probability_columns(
                        predict_emission_probabilities(
                            model,
                            test_feature_cache[test_window],
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
            )
            for test_window in windows:
                probabilities = _probability_average(probability_sums[test_window], len(selected_train_windows))
                _append_transfer_outputs(
                    rows=rows,
                    calibration_rows=calibration_rows,
                    observation_rows=observation_rows,
                    probabilities=probabilities,
                    test_labels=test_labels,
                    original_indices=test_original_indices,
                    session_values=test_session_values,
                    groups=test_groups,
                    group_column=group_column,
                    classes=classes,
                    class_names=encoder.classes_,
                    n_train=len(train_labels),
                    decoder_name=decoder_name,
                    emission_mode=current_emission_mode,
                    feature_preprocessor_name=feature_preprocessor_name,
                    pca_components_value=pca_components_value,
                    normalization_name=normalization_name,
                    baseline_window=baseline_window_value,
                    time_window=test_window,
                    epochs=test_epochs,
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
                    train_subject=train_subject,
                    test_subject=test_subject,
                    train_recording=train_recording,
                    test_recording=test_recording,
                    tuning_metadata=tuning_metadata,
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
                "backend": "sklearn",
                "split_id": split_id,
                "seed": 13,
                "calibration_fold": "",
                "preprocessing_hash": preprocessing_hash,
                "model_hash": default_model_hash,
            }
        ).to_csv(observation_out_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train time-resolved decoders on one MNE Epochs FIF file and evaluate on another.")
    parser.add_argument("--train-epochs", type=Path, required=True)
    parser.add_argument("--test-epochs", type=Path, required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-metadata-csv", type=Path)
    parser.add_argument("--test-metadata-csv", type=Path)
    parser.add_argument("--group-column")
    parser.add_argument("--picks", default="data")
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--step-ms", type=float, default=10.0)
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
        help="Per-recording normalization applied before time-window feature extraction.",
    )
    parser.add_argument(
        "--baseline-window",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        default=DEFAULT_BASELINE_WINDOW,
        help="Baseline time window in seconds for subject_baseline_z and subject_baseline_whiten.",
    )
    parser.add_argument("--tune-hyperparameters", action="store_true", help="Use inner-CV hyperparameter selection on the transfer training recording.")
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
    parser.add_argument("--observations-out", type=Path, help="Optional transfer trial/time probability observation CSV.")
    parser.add_argument("--train-subject", help="Optional training subject identifier to include in output CSVs.")
    parser.add_argument("--test-subject", help="Optional test subject identifier to include in output CSVs.")
    parser.add_argument("--train-recording", default="train", help="Training recording label, e.g. main or cue.")
    parser.add_argument("--test-recording", default="test", help="Test recording label, e.g. cue or main.")
    parser.add_argument(
        "--temporal-train-window",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        help=(
            "Train one model per time-window center in START..STOP seconds on the training recording, "
            "evaluate each model at every test time, and average probabilities."
        ),
    )
    args = parser.parse_args()

    results = run_time_resolved_transfer_decode(
        train_epochs_path=args.train_epochs,
        test_epochs_path=args.test_epochs,
        train_metadata_csv=args.train_metadata_csv,
        test_metadata_csv=args.test_metadata_csv,
        label_column=args.label_column,
        group_column=args.group_column,
        out_path=args.out,
        picks=args.picks,
        tmin=args.tmin,
        tmax=args.tmax,
        window_ms=args.window_ms,
        step_ms=args.step_ms,
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
        train_subject=args.train_subject,
        test_subject=args.test_subject,
        train_recording=args.train_recording,
        test_recording=args.test_recording,
        temporal_train_window=tuple(args.temporal_train_window) if args.temporal_train_window is not None else None,
    )
    print(f"Wrote {args.out}")
    if args.observations_out is not None:
        print(f"Wrote probability observations: {args.observations_out}")
    for emission_mode_name, summary in results.groupby("emission_mode", sort=True):
        time_summary = summary.groupby("time")[["accuracy", "log_loss", "brier", "ece"]].mean()
        best_time = _best_time_by_metric(time_summary, args.selection_metric)
        best_value = time_summary.loc[best_time, args.selection_metric]
        direction = "lowest" if args.selection_metric in RESULT_SELECTION_MINIMIZE_METRICS else "highest"
        print(
            f"Best transfer {emission_mode_name} mean {args.selection_metric} "
            f"({direction}): {best_value:.3f} at {best_time:.3f}s"
        )


if __name__ == "__main__":
    main()

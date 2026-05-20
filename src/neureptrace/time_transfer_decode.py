from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from neureptrace.decoding import (
    DECODER_CLI_CHOICES,
    EMISSION_MODE_CHOICES,
    TUNING_SCORING_CHOICES,
    make_decoder,
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
    normalize_epoch_normalization,
)
from neureptrace.observations import ProbabilityObservationTable, stable_hash

EMISSION_RUN_CHOICES = (*EMISSION_MODE_CHOICES, "both")
TimeWindow = tuple[int, int, float]


def _tuning_metadata(
    models,
    *,
    tune_hyperparameters: bool,
    tuning_cv_splits: int,
    tuning_scoring: str,
    tuning_c_grid: tuple[float, ...],
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
        metadata["best_scores"] = "|".join(str(score) for score in scores)
    return metadata


def _check_matching_time_axes(train_epochs: mne.Epochs, validation_epochs: mne.Epochs) -> None:
    if len(train_epochs.times) != len(validation_epochs.times) or not np.allclose(train_epochs.times, validation_epochs.times, rtol=1e-7, atol=1e-12):
        raise ValueError("Train and validation epochs must have matching time axes after cropping.")


def _encoded_labels(metadata: pd.DataFrame, *, label_column: str, encoder: LabelEncoder | None = None) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    if label_column not in metadata.columns:
        raise ValueError(f"Label column '{label_column}' not found in metadata.")
    raw_labels = metadata[label_column].to_numpy()
    keep = pd.notna(raw_labels)
    if not np.any(keep):
        raise ValueError(f"Label column '{label_column}' contains no non-missing labels.")
    kept_labels = raw_labels[keep]
    if encoder is None:
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(kept_labels)
    else:
        unknown = sorted(set(map(str, kept_labels)) - set(map(str, encoder.classes_)))
        if unknown:
            raise ValueError(f"Validation labels contain classes that were not present in training: {unknown}.")
        encoded = encoder.transform(kept_labels)
    return keep, encoded, encoder


def _load_transfer_inputs(
    *,
    train_epochs_path: Path,
    validation_epochs_path: Path,
    train_metadata_csv: Path | None,
    validation_metadata_csv: Path | None,
    label_column: str,
    picks: str,
    tmin: float | None,
    tmax: float | None,
) -> tuple[mne.Epochs, pd.DataFrame, np.ndarray, mne.Epochs, pd.DataFrame, np.ndarray, np.ndarray, LabelEncoder]:
    train_epochs, train_metadata = _load_epochs_and_metadata(train_epochs_path, train_metadata_csv)
    validation_epochs, validation_metadata = _load_epochs_and_metadata(validation_epochs_path, validation_metadata_csv)

    train_epochs = train_epochs.copy().pick(picks)
    validation_epochs = validation_epochs.copy().pick(picks)
    if tmin is not None or tmax is not None:
        train_epochs.crop(tmin=tmin, tmax=tmax)
        validation_epochs.crop(tmin=tmin, tmax=tmax)
    _check_matching_time_axes(train_epochs, validation_epochs)

    train_keep, train_labels, encoder = _encoded_labels(train_metadata, label_column=label_column)
    validation_keep, validation_labels, encoder = _encoded_labels(validation_metadata, label_column=label_column, encoder=encoder)
    validation_original_indices = np.arange(len(validation_metadata))[validation_keep]

    train_epochs = train_epochs[train_keep]
    validation_epochs = validation_epochs[validation_keep]
    train_metadata = train_metadata.loc[train_keep].reset_index(drop=True)
    validation_metadata = validation_metadata.loc[validation_keep].reset_index(drop=True)
    return train_epochs, train_metadata, train_labels, validation_epochs, validation_metadata, validation_labels, validation_original_indices, encoder


def _append_transfer_outputs(
    *,
    rows: list[dict],
    calibration_rows: list[dict],
    observation_rows: list[dict],
    probabilities: np.ndarray,
    validation_labels: np.ndarray,
    validation_original_indices: np.ndarray,
    validation_metadata: pd.DataFrame,
    classes: np.ndarray,
    class_names: np.ndarray,
    n_train: int,
    decoder_name: str,
    emission_mode: str,
    feature_preprocessor_name: str,
    pca_components_value: int | float | None,
    normalization_name: str,
    baseline_window: tuple[float, float],
    time_window: TimeWindow,
    validation_epochs: mne.Epochs,
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
    subject: str | None,
    train_subject: str | None,
    validation_subject: str | None,
    transfer_label: str,
    tuning_metadata: dict[str, object],
) -> None:
    start, stop, center = time_window
    predictions = probabilities.argmax(axis=1)
    common = {
        "transfer": transfer_label,
        "train_subject": "" if train_subject is None else train_subject,
        "validation_subject": "" if validation_subject is None else validation_subject,
        "fold": 0,
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
        "window_start": float(validation_epochs.times[start]),
        "window_stop": float(validation_epochs.times[stop - 1]),
    }
    row = {
        **common,
        "accuracy": accuracy_score(validation_labels, predictions),
        "log_loss": log_loss(validation_labels, probabilities, labels=classes),
        "brier": brier_score_multiclass(probabilities, validation_labels),
        "ece": expected_calibration_error(probabilities, validation_labels),
        "n_train": n_train,
        "n_test": len(validation_labels),
        "n_classes": len(classes),
        "class_names": "|".join(map(str, class_names)),
    }
    row.update(tuning_metadata)
    if subject is not None:
        row = {"subject": subject, **row}
    rows.append(row)

    if calibration_out_path is not None:
        for bin_row in reliability_bins(probabilities, validation_labels, n_bins=calibration_bins):
            calibration_row = {**common, **bin_row}
            calibration_row.update(tuning_metadata)
            if subject is not None:
                calibration_row = {"subject": subject, **calibration_row}
            calibration_rows.append(calibration_row)

    if observation_out_path is not None:
        session_values = validation_metadata["session"].to_numpy() if "session" in validation_metadata.columns else None
        for position, sample_index in enumerate(validation_original_indices):
            true_label = int(validation_labels[position])
            predicted_label = int(predictions[position])
            observation = {
                **common,
                "split_id": split_id,
                "seed": 13,
                "backend": "sklearn",
                "sample_index": int(sample_index),
                "sequence_id": int(sample_index),
                "session": "" if session_values is None else session_values[position],
                "true_label": true_label,
                "true_class": str(class_names[true_label]),
                "predicted_label": predicted_label,
                "predicted_class": str(class_names[predicted_label]),
                "probability_true_class": float(probabilities[position, true_label]),
                "confidence": float(probabilities[position].max()),
                "is_correct": bool(predicted_label == true_label),
                "calibration_fold": "",
                "preprocessing_hash": preprocessing_hash,
                "model_hash": model_hash,
            }
            observation.update(tuning_metadata)
            for class_index, class_name in enumerate(class_names):
                observation[f"class_{class_index}"] = str(class_name)
                observation[f"prob_class_{class_index}"] = float(probabilities[position, class_index])
            if subject is not None:
                observation = {"subject": subject, **observation}
            observation_rows.append(observation)


def run_time_transfer_decode(
    train_epochs_path: Path,
    validation_epochs_path: Path,
    label_column: str,
    out_path: Path,
    *,
    train_metadata_csv: Path | None = None,
    validation_metadata_csv: Path | None = None,
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
    tuning_c_grid: tuple[float, ...] | str | None = None,
    calibration_out_path: Path | None = None,
    calibration_bins: int = 10,
    observation_out_path: Path | None = None,
    subject: str | None = None,
    train_subject: str | None = None,
    validation_subject: str | None = None,
    transfer_label: str = "train-to-validation",
    temporal_train_window: tuple[float, float] | None = None,
) -> pd.DataFrame:
    """Train on one epochs file and evaluate on another across time windows.

    This is the NeuRepTrace-native equivalent of PyMEGDec-style main-to-cue or
    cue-to-main stimulus transfer: no validation samples are used for fitting,
    while metrics, calibration bins, and probability observations are emitted for
    every validation trial/time window.
    """

    train_epochs, _train_metadata, train_labels, validation_epochs, validation_metadata, validation_labels, validation_original_indices, encoder = _load_transfer_inputs(
        train_epochs_path=train_epochs_path,
        validation_epochs_path=validation_epochs_path,
        train_metadata_csv=train_metadata_csv,
        validation_metadata_csv=validation_metadata_csv,
        label_column=label_column,
        picks=picks,
        tmin=tmin,
        tmax=tmax,
    )
    decoder_name = normalize_decoder_name(decoder)
    emission_modes = list(EMISSION_MODE_CHOICES) if emission_mode == "both" else [normalize_emission_mode(emission_mode)]
    feature_preprocessor_name = normalize_feature_preprocessor(feature_preprocessor)
    normalization_name = normalize_epoch_normalization(normalization)
    baseline_window_value = _normalize_baseline_window(baseline_window)
    if feature_preprocessor_name == "none" and pca_components is not None:
        raise ValueError("pca_components can only be set when feature_preprocessor is 'pca', 'pca_whiten', or 'anova_select'.")
    if feature_preprocessor_name == "anova_select":
        pca_components_value = normalize_anova_select_percentile(pca_components)
    elif feature_preprocessor_name != "none":
        pca_components_value = normalize_pca_components(pca_components)
    else:
        pca_components_value = None
    tuning_scoring = normalize_tuning_scoring(tuning_scoring)
    tuning_c_grid_values = parse_c_grid(tuning_c_grid)
    normalized_temporal_train_window = _normalize_temporal_train_window(temporal_train_window)

    train_data = _apply_epoch_normalization(train_epochs.get_data(copy=False), train_epochs.times, normalization_name, baseline_window=baseline_window_value)
    validation_data = _apply_epoch_normalization(validation_epochs.get_data(copy=False), validation_epochs.times, normalization_name, baseline_window=baseline_window_value)
    windows = time_windows(train_epochs.times, window_ms=window_ms, step_ms=step_ms)
    selected_train_windows = _select_temporal_train_windows(windows, normalized_temporal_train_window)
    temporal_mode = "same_time_transfer" if selected_train_windows is None else "train_window_transfer_ensemble"
    classes = np.arange(len(encoder.classes_))
    split_id = f"transfer:{transfer_label}"
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
            "train_epochs": str(train_epochs_path),
            "validation_epochs": str(validation_epochs_path),
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

    rows: list[dict] = []
    calibration_rows: list[dict] = []
    observation_rows: list[dict] = []

    if selected_train_windows is None:
        for time_window in windows:
            start, stop, center = time_window
            train_features = _features_for_window(train_data, time_window)
            validation_features = _features_for_window(validation_data, time_window)
            for current_emission_mode in emission_modes:
                model = make_decoder(
                    decoder_name,
                    max_iter=max_iter,
                    emission_mode=current_emission_mode,
                    feature_preprocessor=feature_preprocessor_name,
                    pca_components=pca_components_value,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                )
                model.fit(train_features, train_labels)
                probabilities = _align_probability_columns(
                    predict_emission_probabilities(model, validation_features, emission_mode=current_emission_mode),
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
                    validation_labels=validation_labels,
                    validation_original_indices=validation_original_indices,
                    validation_metadata=validation_metadata,
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
                    validation_epochs=validation_epochs,
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
                    subject=subject,
                    train_subject=train_subject,
                    validation_subject=validation_subject,
                    transfer_label=transfer_label,
                    tuning_metadata=tuning_metadata,
                )
    else:
        train_feature_cache = {time_window: _features_for_window(train_data, time_window) for time_window in windows}
        validation_feature_cache = {time_window: _features_for_window(validation_data, time_window) for time_window in windows}
        train_time, train_window_start, train_window_stop = _train_window_summary(train_epochs, selected_train_windows)
        train_window_centers = [window[2] for window in selected_train_windows]
        for current_emission_mode in emission_modes:
            fitted_models = []
            probability_sums = {time_window: np.zeros((len(validation_labels), len(classes)), dtype=float) for time_window in windows}
            for train_window in selected_train_windows:
                model = make_decoder(
                    decoder_name,
                    max_iter=max_iter,
                    emission_mode=current_emission_mode,
                    feature_preprocessor=feature_preprocessor_name,
                    pca_components=pca_components_value,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                )
                model.fit(train_feature_cache[train_window], train_labels)
                fitted_models.append(model)
                for validation_window in windows:
                    probability_sums[validation_window] += _align_probability_columns(
                        predict_emission_probabilities(model, validation_feature_cache[validation_window], emission_mode=current_emission_mode),
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
            for validation_window in windows:
                probabilities = _probability_average(probability_sums[validation_window], len(selected_train_windows))
                _append_transfer_outputs(
                    rows=rows,
                    calibration_rows=calibration_rows,
                    observation_rows=observation_rows,
                    probabilities=probabilities,
                    validation_labels=validation_labels,
                    validation_original_indices=validation_original_indices,
                    validation_metadata=validation_metadata,
                    classes=classes,
                    class_names=encoder.classes_,
                    n_train=len(train_labels),
                    decoder_name=decoder_name,
                    emission_mode=current_emission_mode,
                    feature_preprocessor_name=feature_preprocessor_name,
                    pca_components_value=pca_components_value,
                    normalization_name=normalization_name,
                    baseline_window=baseline_window_value,
                    time_window=validation_window,
                    validation_epochs=validation_epochs,
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
                    train_subject=train_subject,
                    validation_subject=validation_subject,
                    transfer_label=transfer_label,
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
    parser = argparse.ArgumentParser(description="Train on one MNE Epochs file and evaluate on another across time windows.")
    parser.add_argument("--train-epochs", type=Path, required=True)
    parser.add_argument("--validation-epochs", type=Path, required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-metadata-csv", type=Path)
    parser.add_argument("--validation-metadata-csv", type=Path)
    parser.add_argument("--picks", default="data")
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--step-ms", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--decoder", choices=DECODER_CLI_CHOICES, default="logistic")
    parser.add_argument("--emission-mode", choices=EMISSION_RUN_CHOICES, default="calibrated")
    parser.add_argument("--feature-preprocessor", choices=FEATURE_PREPROCESSOR_RUN_CHOICES, default="none")
    parser.add_argument("--pca-components")
    parser.add_argument("--normalization", choices=EPOCH_NORMALIZATION_RUN_CHOICES, default="none")
    parser.add_argument("--baseline-window", nargs=2, type=float, metavar=("START", "STOP"), default=DEFAULT_BASELINE_WINDOW)
    parser.add_argument("--tune-hyperparameters", action="store_true")
    parser.add_argument("--tuning-cv-splits", type=int, default=3)
    parser.add_argument("--tuning-scoring", choices=TUNING_SCORING_CHOICES, default="accuracy")
    parser.add_argument("--selection-metric", choices=RESULT_SELECTION_METRIC_CHOICES, default="accuracy")
    parser.add_argument("--tuning-c-grid", default=",".join(str(value) for value in parse_c_grid(None)))
    parser.add_argument("--calibration-out", type=Path)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--observations-out", type=Path)
    parser.add_argument("--subject")
    parser.add_argument("--train-subject")
    parser.add_argument("--validation-subject")
    parser.add_argument("--transfer-label", default="train-to-validation")
    parser.add_argument("--temporal-train-window", nargs=2, type=float, metavar=("START", "STOP"))
    args = parser.parse_args()

    results = run_time_transfer_decode(
        train_epochs_path=args.train_epochs,
        validation_epochs_path=args.validation_epochs,
        train_metadata_csv=args.train_metadata_csv,
        validation_metadata_csv=args.validation_metadata_csv,
        label_column=args.label_column,
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
        subject=args.subject,
        train_subject=args.train_subject,
        validation_subject=args.validation_subject,
        transfer_label=args.transfer_label,
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
        print(f"Best transfer {emission_mode_name} mean {args.selection_metric} ({direction}): {best_value:.3f} at {best_time:.3f}s")


if __name__ == "__main__":
    main()

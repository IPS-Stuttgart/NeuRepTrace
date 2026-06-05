from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder

from neureptrace import mne_time_decode as _base
from neureptrace.decoding import (
    EMISSION_MODE_CHOICES,
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
from neureptrace.observations import ProbabilityObservationTable, stable_hash


FoldNormalizationParams = dict[str, np.ndarray]


def _fit_fold_epoch_normalization(
    data: np.ndarray,
    times: np.ndarray,
    normalization: str,
    *,
    baseline_window: tuple[float, float],
    train_idx: Sequence[int] | np.ndarray,
) -> FoldNormalizationParams:
    """Fit epoch-normalization statistics from train-fold trials only."""

    normalization = _base.normalize_epoch_normalization(normalization)
    if normalization in {"none", "subject_trial_z"}:
        return {}

    train_idx = np.asarray(train_idx, dtype=int)
    if train_idx.size == 0:
        raise ValueError("Cannot fit fold-local epoch normalization without training trials.")
    train_data = np.asarray(data, dtype=float)[train_idx]

    if normalization == "subject_z":
        mean, std = _base._channel_mean_std(train_data)
        return {"mean": mean, "std": std}

    mask = _base._baseline_time_mask(times, baseline_window)
    baseline = train_data[:, :, mask]
    baseline_mean, baseline_std = _base._channel_mean_std(baseline)
    if normalization == "subject_baseline_z":
        return {"mean": baseline_mean, "std": baseline_std}

    if normalization == "subject_baseline_whiten":
        return {
            "mean": baseline_mean,
            "whitening": _base._baseline_channel_whitening_matrix(train_data, times, baseline_window),
        }

    raise ValueError(f"Unsupported normalization: {normalization}")


def _apply_fold_epoch_normalization(
    data: np.ndarray,
    normalization: str,
    params: FoldNormalizationParams,
) -> np.ndarray:
    """Apply fold-local epoch-normalization parameters to train and test trials."""

    data = np.asarray(data, dtype=float)
    normalization = _base.normalize_epoch_normalization(normalization)
    if normalization == "none":
        return data

    if normalization == "subject_trial_z":
        mean = data.mean(axis=(1, 2), keepdims=True)
        std = _base._nonzero_std(data.std(axis=(1, 2), keepdims=True))
        return (data - mean) / std

    if normalization in {"subject_z", "subject_baseline_z"}:
        return (data - params["mean"]) / params["std"]

    if normalization == "subject_baseline_whiten":
        centered = data - params["mean"]
        whitened = np.einsum("ntc,dc->ntd", np.transpose(centered, (0, 2, 1)), params["whitening"])
        return np.transpose(whitened, (0, 2, 1))

    raise ValueError(f"Unsupported normalization: {normalization}")


def _normalize_epoch_data_for_fold(
    data: np.ndarray,
    times: np.ndarray,
    normalization: str,
    *,
    baseline_window: tuple[float, float],
    train_idx: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """Normalize all epochs with statistics fitted only on ``train_idx``."""

    params = _fit_fold_epoch_normalization(
        data,
        times,
        normalization,
        baseline_window=baseline_window,
        train_idx=train_idx,
    )
    return _apply_fold_epoch_normalization(data, normalization, params)


def _fit_foldlocal_source_time_selector(
    *,
    raw_data: np.ndarray,
    times: np.ndarray,
    normalization: str,
    baseline_window: tuple[float, float],
    candidate_windows: Sequence[_base.TimeWindow],
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
    """Learn response-time weights with inner source-only fold-local decoding."""

    inner_splits = list(make_tuning_cross_validator(train_labels, train_groups, tuning_cv_splits))
    if not inner_splits:
        raise ValueError("Source-time selection needs at least one inner validation split.")

    train_idx = np.asarray(train_idx, dtype=int)
    validation_probabilities: list[list[np.ndarray]] = [[] for _ in candidate_windows]
    validation_labels: list[np.ndarray] = []
    for inner_train_local, inner_valid_local in inner_splits:
        inner_train_global = train_idx[inner_train_local]
        inner_valid_global = train_idx[inner_valid_local]
        inner_data = _normalize_epoch_data_for_fold(
            raw_data,
            times,
            normalization,
            baseline_window=baseline_window,
            train_idx=inner_train_global,
        )
        validation_labels.append(train_labels[inner_valid_local])
        for window_index, window in enumerate(candidate_windows):
            features = _base._features_for_window(inner_data, window)
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
            model.fit(features[inner_train_global], train_labels[inner_train_local])
            probabilities = _base._align_probability_columns(
                predict_emission_probabilities(
                    model,
                    features[inner_valid_global],
                    emission_mode=emission_mode,
                ),
                model=model,
                classes=classes,
            )
            probabilities = _base._apply_class_prior_correction(
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
        for weights_candidate in _base._nonnegative_weight_candidates_with_uniform(len(candidate_windows)):
            probabilities = _base._combine_probability_logits(probability_cube, weights_candidate)
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
        weights, source_score = _base._fit_classwise_nonnegative_time_weights(probability_cube, labels)
        bias = None
        objective = None
    elif mode == "source_oof_logit_stacker":
        weights, bias, source_score, objective = _base._fit_logit_stacker_time_weights_and_bias(probability_cube, labels)
    else:
        raise ValueError(f"Unknown source-time-selection mode '{mode}'.")

    time_mass = weights if weights.ndim == 1 else weights.sum(axis=0)
    metadata = {
        "source_time_selection": mode,
        "source_time_selection_candidate_times": "|".join(f"{window[2]:.12g}" for window in candidate_windows),
        "source_time_selection_time_scores": "|".join(f"{score:.12g}" for score in time_scores),
        "source_time_selection_weights": _base._format_source_time_weights(weights),
        "source_time_selection_selected_time": float(candidate_windows[int(np.argmax(time_mass))][2]),
        "source_time_selection_inner_score": float(source_score),
        "source_time_selection_weight_type": "classwise" if weights.ndim == 2 else ("stacker" if bias is not None else "global"),
        "source_time_selection_normalization_scope": "inner_train_fold",
    }
    if bias is not None:
        metadata.update(
            {
                "source_time_selection_stacker_type": _base.SOURCE_TIME_STACKER_TYPE,
                "source_time_selection_stacker_regularization": _base.SOURCE_TIME_STACKER_REGULARIZATION,
                "source_time_selection_class_bias": _base._format_source_time_bias(bias),
                "source_time_selection_inner_objective": float(objective) if objective is not None else "",
            }
        )
    return weights, bias, metadata


def run_time_resolved_decode(
    epochs_path: Path,
    label_column: str,
    out_path: Path,
    *,
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
    baseline_window: tuple[float, float] | None = _base.DEFAULT_BASELINE_WINDOW,
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
    time_decode_backend: str = "sklearn",
    class_prior_correction: str = "none",
    source_calibration: str = "none",
    source_time_selection: str = "none",
    source_time_selection_times: Sequence[float] | str | None = None,
    source_time_selection_output_time: float = 0.184,
    label_shuffle_control: bool = False,
    label_shuffle_seed: int = 13,
) -> pd.DataFrame:
    """Run time-resolved decoding with train-fold-local epoch normalization.

    Subject-level normalization statistics are fitted separately inside each
    outer cross-validation train fold and then applied to that fold's train and
    held-out trials. This prevents transductive leakage from held-out trials
    into ``subject_z``, ``subject_baseline_z``, and ``subject_baseline_whiten``.
    """

    epochs, metadata = _base._load_epochs_and_metadata(
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
    normalization_name = _base.normalize_epoch_normalization(normalization)
    time_decode_backend = _base.normalize_time_decode_backend(time_decode_backend)
    if time_decode_backend != "sklearn":
        raise ValueError("Fold-local normalization currently supports only the sklearn time-decode backend.")
    label_shuffle_control = bool(label_shuffle_control)
    label_shuffle_seed = int(label_shuffle_seed)
    baseline_window_value = _base._normalize_baseline_window(baseline_window)
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
    normalized_decode_window = _base._normalize_decode_window(decode_window)
    normalized_temporal_train_window = _base._normalize_temporal_train_window(temporal_train_window)
    temporal_train_mode_name = _base._normalize_temporal_train_mode(temporal_train_mode)
    class_prior_correction_name = _base.normalize_class_prior_correction(class_prior_correction)
    source_calibration_name = _base.normalize_source_calibration(source_calibration)
    source_time_selection_name = _base.normalize_source_time_selection(source_time_selection)
    source_time_selection_time_values = _base._parse_float_sequence(
        source_time_selection_times,
        default=_base.DEFAULT_SOURCE_TIME_SELECTION_TIMES,
    )
    source_time_selection_output_time = float(source_time_selection_output_time)
    outer_test_groups_value = _base._normalize_outer_test_groups(outer_test_groups)
    if source_time_selection_name != "none" and normalized_temporal_train_window is not None:
        raise ValueError("source_time_selection currently supports same-time decoding only.")
    if source_time_selection_name != "none" and source_calibration_name != "none":
        raise ValueError("source_time_selection should be run with source_calibration='none'.")

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
            "normalization_scope": "train_fold",
            "baseline_window": baseline_window_value,
            "decode_window": normalized_decode_window,
            "temporal_train_window": normalized_temporal_train_window,
            "temporal_train_mode": None if normalized_temporal_train_window is None else temporal_train_mode_name,
            "class_prior_correction": class_prior_correction_name,
            "source_calibration": source_calibration_name,
            "source_time_selection": source_time_selection_name,
            "source_time_selection_times": source_time_selection_time_values,
            "outer_test_groups": outer_test_groups_value,
        }
    )
    default_model_hash = _base._model_hash(
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
        class_prior_correction=class_prior_correction_name,
        source_calibration=source_calibration_name,
        source_time_selection=source_time_selection_name,
        source_time_selection_times=source_time_selection_time_values,
        label_shuffle_control=label_shuffle_control,
        label_shuffle_seed=label_shuffle_seed,
    )

    raw_data = epochs.get_data(copy=False)
    classes = np.arange(len(encoder.classes_))
    rows: list[dict] = []
    calibration_rows: list[dict] = []
    observation_rows: list[dict] = []
    all_windows = time_windows(epochs.times, window_ms=window_ms, step_ms=step_ms)
    windows = _base._select_decode_windows(all_windows, normalized_decode_window)
    selected_train_windows = _base._select_temporal_train_windows(all_windows, normalized_temporal_train_window)
    splits = _base._filter_splits_for_outer_test_groups(
        list(enumerate(make_cross_validator(labels, groups, n_splits))),
        groups,
        outer_test_groups_value,
    )

    if source_time_selection_name != "none":
        candidate_windows = _base._nearest_candidate_windows(windows, source_time_selection_time_values)
        selection_decoder_name = f"{decoder_name}_{_base.SOURCE_TIME_SELECTION_OUTPUT_DECODER_SUFFIX[source_time_selection_name]}"
        candidate_centers = [window[2] for window in candidate_windows]
        for fold, (train_idx, test_idx) in splits:
            fold_data = _normalize_epoch_data_for_fold(
                raw_data,
                epochs.times,
                normalization_name,
                baseline_window=baseline_window_value,
                train_idx=train_idx,
            )
            feature_cache = {time_window: _base._features_for_window(fold_data, time_window) for time_window in candidate_windows}
            test_labels = labels[test_idx]
            train_labels = _base._fold_training_labels(
                labels,
                train_idx,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
                context=(split_id, fold, "foldlocal", source_time_selection_name, *candidate_centers),
            )
            train_groups = None if groups is None else groups[train_idx]
            for current_emission_mode in emission_modes:
                weights, bias, selection_metadata = _fit_foldlocal_source_time_selector(
                    raw_data=raw_data,
                    times=epochs.times,
                    normalization=normalization_name,
                    baseline_window=baseline_window_value,
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
                selected_indices = _base._source_time_active_indices(weights)
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
                    probabilities = _base._align_probability_columns(
                        predict_emission_probabilities(
                            model,
                            features[test_idx],
                            emission_mode=current_emission_mode,
                        ),
                        model=model,
                        classes=classes,
                    )
                    probabilities = _base._apply_class_prior_correction(
                        probabilities,
                        train_labels,
                        classes,
                        class_prior_correction_name,
                    )
                    test_probability_parts.append(probabilities)
                    if not final_tuning_metadata:
                        final_tuning_metadata = _base._tuning_metadata(
                            model,
                            tune_hyperparameters=tune_hyperparameters,
                            tuning_cv_splits=tuning_cv_splits,
                            tuning_scoring=tuning_scoring,
                            tuning_c_grid=tuning_c_grid_values,
                        )
                if not test_probability_parts:
                    raise ValueError("Source-time selection produced no final test probabilities.")
                probability_cube = np.stack(test_probability_parts, axis=1)
                probabilities = _base._combine_source_time_probabilities(
                    probability_cube,
                    weights,
                    selected_indices,
                    bias=bias,
                )
                tuning_metadata = {**final_tuning_metadata, **selection_metadata}
                selected_windows = [candidate_windows[index] for index in selected_indices]
                synthetic_window = (
                    min(window[0] for window in selected_windows),
                    max(window[1] for window in selected_windows),
                    source_time_selection_output_time,
                )
                current_model_hash = _base._model_hash(
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
                    class_prior_correction=class_prior_correction_name,
                    source_calibration=source_calibration_name,
                    source_time_selection=source_time_selection_name,
                    source_time_selection_times=source_time_selection_time_values,
                    label_shuffle_control=label_shuffle_control,
                    label_shuffle_seed=label_shuffle_seed,
                )
                _base._append_decoded_outputs(
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
                    class_prior_correction=class_prior_correction_name,
                    source_calibration_metadata=_base.source_calibration_metadata(_base.SourceProbabilityCalibrator(mode="none")),
                    label_shuffle_control=label_shuffle_control,
                    label_shuffle_seed=label_shuffle_seed,
                    outer_test_groups=outer_test_groups_value,
                )
    elif selected_train_windows is None:
        for fold, (train_idx, test_idx) in splits:
            fold_data = _normalize_epoch_data_for_fold(
                raw_data,
                epochs.times,
                normalization_name,
                baseline_window=baseline_window_value,
                train_idx=train_idx,
            )
            test_labels = labels[test_idx]
            train_labels = _base._fold_training_labels(
                labels,
                train_idx,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
                context=(split_id, fold, "foldlocal_same_time"),
            )
            for time_window in windows:
                features = _base._features_for_window(fold_data, time_window)
                start, stop, center = time_window
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
                    model.fit(features[train_idx], train_labels)

                    probabilities = _base._align_probability_columns(
                        predict_emission_probabilities(
                            model,
                            features[test_idx],
                            emission_mode=current_emission_mode,
                        ),
                        model=model,
                        classes=classes,
                    )
                    probabilities = _base._apply_class_prior_correction(
                        probabilities,
                        train_labels,
                        classes,
                        class_prior_correction_name,
                    )
                    calibrator = _base.fit_inner_source_probability_calibrator(
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
                    probabilities = _base.apply_source_probability_calibration(probabilities, calibrator)
                    source_metadata = _base.source_calibration_metadata(calibrator)
                    tuning_metadata = _base._tuning_metadata(
                        model,
                        tune_hyperparameters=tune_hyperparameters,
                        tuning_cv_splits=tuning_cv_splits,
                        tuning_scoring=tuning_scoring,
                        tuning_c_grid=tuning_c_grid_values,
                    )
                    current_model_hash = _base._model_hash(
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
                        class_prior_correction=class_prior_correction_name,
                        source_calibration=source_calibration_name,
                        label_shuffle_control=label_shuffle_control,
                        label_shuffle_seed=label_shuffle_seed,
                    )
                    _base._append_decoded_outputs(
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
                        class_prior_correction=class_prior_correction_name,
                        source_calibration_metadata=source_metadata,
                        label_shuffle_control=label_shuffle_control,
                        label_shuffle_seed=label_shuffle_seed,
                        outer_test_groups=outer_test_groups_value,
                    )
    elif temporal_train_mode_name == "pooled":
        train_time, train_window_start, train_window_stop = _base._train_window_summary(epochs, selected_train_windows)
        train_window_centers = [window[2] for window in selected_train_windows]
        model_windows = list(dict.fromkeys([*windows, *selected_train_windows]))
        for fold, (train_idx, test_idx) in splits:
            fold_data = _normalize_epoch_data_for_fold(
                raw_data,
                epochs.times,
                normalization_name,
                baseline_window=baseline_window_value,
                train_idx=train_idx,
            )
            feature_cache = {time_window: _base._features_for_window(fold_data, time_window) for time_window in model_windows}
            test_labels = labels[test_idx]
            train_labels = _base._fold_training_labels(
                labels,
                train_idx,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
                context=(split_id, fold, "foldlocal_pooled", *train_window_centers),
            )
            fold_labels = labels.copy()
            fold_labels[train_idx] = train_labels
            pooled_train_features, pooled_train_labels, pooled_train_groups = _base._pooled_temporal_training_set(
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
                tuning_metadata = _base._tuning_metadata(
                    model,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                )
                current_model_hash = _base._model_hash(
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
                    class_prior_correction=class_prior_correction_name,
                    label_shuffle_control=label_shuffle_control,
                    label_shuffle_seed=label_shuffle_seed,
                )
                for test_window in windows:
                    probabilities = _base._align_probability_columns(
                        predict_emission_probabilities(
                            model,
                            feature_cache[test_window][test_idx],
                            emission_mode=current_emission_mode,
                        ),
                        model=model,
                        classes=classes,
                    )
                    probabilities = _base._apply_class_prior_correction(
                        probabilities,
                        train_labels,
                        classes,
                        class_prior_correction_name,
                    )
                    if source_calibration_name != "none":
                        raise ValueError("source_calibration currently supports same-time decoding only.")
                    _base._append_decoded_outputs(
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
                        source_calibration_metadata=_base.source_calibration_metadata(_base.SourceProbabilityCalibrator(mode="none")),
                        label_shuffle_control=label_shuffle_control,
                        label_shuffle_seed=label_shuffle_seed,
                        outer_test_groups=outer_test_groups_value,
                    )
    else:
        train_time, train_window_start, train_window_stop = _base._train_window_summary(epochs, selected_train_windows)
        train_window_centers = [window[2] for window in selected_train_windows]
        model_windows = list(dict.fromkeys([*windows, *selected_train_windows]))
        for fold, (train_idx, test_idx) in splits:
            fold_data = _normalize_epoch_data_for_fold(
                raw_data,
                epochs.times,
                normalization_name,
                baseline_window=baseline_window_value,
                train_idx=train_idx,
            )
            feature_cache = {time_window: _base._features_for_window(fold_data, time_window) for time_window in model_windows}
            test_labels = labels[test_idx]
            train_labels = _base._fold_training_labels(
                labels,
                train_idx,
                label_shuffle_control=label_shuffle_control,
                label_shuffle_seed=label_shuffle_seed,
                context=(split_id, fold, "foldlocal_train_window_ensemble", *train_window_centers),
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
                        probability_sums[test_window] += _base._align_probability_columns(
                            predict_emission_probabilities(
                                model,
                                feature_cache[test_window][test_idx],
                                emission_mode=current_emission_mode,
                            ),
                            model=model,
                            classes=classes,
                        )

                tuning_metadata = _base._tuning_metadata(
                    fitted_models,
                    tune_hyperparameters=tune_hyperparameters,
                    tuning_cv_splits=tuning_cv_splits,
                    tuning_scoring=tuning_scoring,
                    tuning_c_grid=tuning_c_grid_values,
                )
                current_model_hash = _base._model_hash(
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
                    class_prior_correction=class_prior_correction_name,
                    label_shuffle_control=label_shuffle_control,
                    label_shuffle_seed=label_shuffle_seed,
                )
                for test_window in windows:
                    probabilities = _base._probability_average(probability_sums[test_window], len(selected_train_windows))
                    probabilities = _base._apply_class_prior_correction(
                        probabilities,
                        train_labels,
                        classes,
                        class_prior_correction_name,
                    )
                    if source_calibration_name != "none":
                        raise ValueError("source_calibration currently supports same-time decoding only.")
                    _base._append_decoded_outputs(
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
                        class_prior_correction=class_prior_correction_name,
                        source_calibration_metadata=_base.source_calibration_metadata(_base.SourceProbabilityCalibrator(mode="none")),
                        label_shuffle_control=label_shuffle_control,
                        label_shuffle_seed=label_shuffle_seed,
                        outer_test_groups=outer_test_groups_value,
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
    """Run the existing MNE time-decode CLI with fold-local normalization."""

    _base.run_time_resolved_decode = run_time_resolved_decode
    _base.main()


if __name__ == "__main__":
    main()

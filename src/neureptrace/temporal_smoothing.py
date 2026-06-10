from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from neureptrace.metrics import brier_score_multiclass, expected_calibration_error
from neureptrace.observations import stable_hash
from neureptrace.temporal_model import (
    _class_names,
    _expand_paths,
    _filter_time_window,
    _forward_backward,
    _model_group_columns,
    _normalize_probabilities,
    _sequences_from_frame,
    fit_sticky_switching_model,
    probability_columns,
    read_probability_observations,
    sequence_key_columns,
    validate_unique_sequence_times,
)

DEFAULT_FIT_WINDOW = (0.1, 0.8)
DEFAULT_POSTSTIMULUS_APPLY_WINDOW = (0.04, 0.30)
DEFAULT_EMISSION_SUFFIX = "temporal_posterior"
SMOOTHING_MODE_CHOICES = ("forward_backward", "forward_only", "poststimulus_forward_only")
SMOOTHING_METHODS = {
    "forward_backward": "sticky_forward_backward",
    "forward_only": "sticky_forward_only",
    "poststimulus_forward_only": "sticky_poststimulus_forward_only",
}
SMOOTHING_GROUP_COLUMNS = (
    "decoder",
    "emission_mode",
    "feature_preprocessor",
    "pca_components",
    "tuned_hyperparameters",
    "tuning_cv_splits",
    "tuning_scoring",
    "tuning_c_grid",
    "temporal_mode",
    "temporal_train_window_start",
    "temporal_train_window_stop",
)
METRIC_GROUP_COLUMNS = (
    "subject",
    "fold",
    *SMOOTHING_GROUP_COLUMNS,
    "train_time",
    "test_time",
    "time",
    "window_start",
    "window_stop",
)
METRIC_PROVENANCE_COLUMNS = (
    "split_id",
    "seed",
    "backend",
    "preprocessing_hash",
    "model_hash",
    "label_shuffle_control",
    "label_shuffle_seed",
    "class_prior_correction",
    "source_calibration",
    "source_time_selection",
    "source_time_selection_candidate_times",
    "source_time_selection_selected_time",
    "alignment_method",
    "alignment_anchor_mode",
    "alignment_anchor_column",
    "alignment_repetition_cap",
    "alignment_components",
    "alignment_times",
    "alignment_window_mode",
    "alignment_same_decode_window",
    "alignment_target_projection",
    "alignment_strict_source_only",
    "alignment_uses_unlabeled_target_data",
    "alignment_uses_class_labels",
    "alignment_target_calibrated",
    "alignment_target_calibration_per_anchor",
    "alignment_target_calibration_seed",
    "alignment_oracle_target_calibrated",
    "alignment_debug_upper_bound",
    "alignment_valid_for_benchmark",
    "alignment_protocol",
    "base_emission_mode",
    "best_params",
    "best_score",
    "best_scores",
    "temporal_smoothing_method",
    "temporal_smoothing_stay_probability",
    "temporal_smoothing_fit_window_start",
    "temporal_smoothing_fit_window_stop",
    "temporal_smoothing_apply_window_start",
    "temporal_smoothing_apply_window_stop",
)


def _iter_groups(frame: pd.DataFrame, columns: list[str]) -> Iterable[tuple[object, pd.DataFrame]]:
    if columns:
        yield from frame.groupby(columns, sort=True, dropna=False)
    else:
        yield (), frame


def _group_values(columns: list[str], key: object) -> dict[str, object]:
    if not columns:
        return {}
    key_values = key if isinstance(key, tuple) else (key,)
    return dict(zip(columns, key_values, strict=True))


def _smoothing_group_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in SMOOTHING_GROUP_COLUMNS if column in frame.columns]


def _smoothed_emission_mode(base_mode: object, suffix: str) -> str:
    base = "" if pd.isna(base_mode) else str(base_mode).strip()
    if not base:
        return suffix
    if base.endswith(f"_{suffix}"):
        return base
    return f"{base}_{suffix}"


def _numeric_labels(frame: pd.DataFrame, n_classes: int) -> np.ndarray:
    if "true_label" not in frame.columns:
        raise ValueError("Temporal smoothing metrics require a true_label column.")
    labels = pd.to_numeric(frame["true_label"], errors="coerce")
    if labels.isna().any():
        raise ValueError("true_label must be numeric and non-missing.")
    label_values = labels.to_numpy(dtype=float)
    rounded = np.rint(label_values)
    if not bool(np.isclose(label_values, rounded, rtol=0.0, atol=1.0e-12).all()):
        raise ValueError("true_label values must be integer-valued.")
    labels_array = rounded.astype(int)
    if bool(((labels_array < 0) | (labels_array >= n_classes)).any()):
        raise ValueError("true_label values must index prob_class_* columns.")
    return labels_array


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    if len(labels) == 0:
        return float("nan")
    effective_k = min(int(k), probabilities.shape[1])
    top_columns = np.argsort(probabilities, axis=1)[:, ::-1][:, :effective_k]
    return float(np.mean(np.any(top_columns == labels[:, None], axis=1)))


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True)), axis=axis)


def _log_transition(n_states: int, stay_probability: float) -> np.ndarray:
    if n_states < 2:
        raise ValueError("Temporal smoothing requires at least two states.")
    if not 0.0 < stay_probability < 1.0:
        raise ValueError("stay_probability must be between 0 and 1.")
    switch_probability = (1.0 - stay_probability) / (n_states - 1)
    transition = np.full((n_states, n_states), switch_probability)
    np.fill_diagonal(transition, stay_probability)
    return np.log(transition)


def _forward_filter(probabilities: np.ndarray, stay_probability: float) -> np.ndarray:
    probabilities = _normalize_probabilities(probabilities)
    n_states = probabilities.shape[1]
    log_emissions = np.log(probabilities)
    log_transition = _log_transition(n_states, stay_probability)

    log_alpha = np.empty_like(log_emissions)
    log_alpha[0] = np.full(n_states, -np.log(n_states)) + log_emissions[0]
    log_alpha[0] = log_alpha[0] - _logsumexp(log_alpha[0])
    for time_index in range(1, len(probabilities)):
        log_alpha[time_index] = log_emissions[time_index] + _logsumexp(
            log_alpha[time_index - 1][:, None] + log_transition,
            axis=0,
        )
        log_alpha[time_index] = log_alpha[time_index] - _logsumexp(log_alpha[time_index])

    posterior = np.exp(log_alpha)
    return posterior / posterior.sum(axis=1, keepdims=True)


def _smooth_sequence_posteriors(
    sequence_frame: pd.DataFrame,
    probabilities: np.ndarray,
    *,
    stay_probability: float,
    mode: str,
    apply_window: tuple[float, float] | None,
) -> np.ndarray:
    if mode == "forward_backward":
        return _forward_backward(probabilities, stay_probability)
    if mode == "forward_only":
        return _forward_filter(probabilities, stay_probability)
    if mode != "poststimulus_forward_only":
        raise ValueError(f"Unknown temporal smoothing mode '{mode}'.")

    if apply_window is None:
        apply_window = DEFAULT_POSTSTIMULUS_APPLY_WINDOW
    times = pd.to_numeric(sequence_frame["time"], errors="coerce").to_numpy(dtype=float)
    mask = (times >= float(apply_window[0])) & (times <= float(apply_window[1]))
    posterior = probabilities.copy()
    if int(mask.sum()) >= 2:
        posterior[mask] = _forward_filter(probabilities[mask], stay_probability)
    elif int(mask.sum()) == 1:
        posterior[mask] = _normalize_probabilities(probabilities[mask])
    return _normalize_probabilities(posterior)


def _with_posterior_columns(
    sequence_frame: pd.DataFrame,
    posterior: np.ndarray,
    *,
    prob_columns: list[str],
    class_names: list[str],
    stay_probability: float,
    fit_window: tuple[float, float] | None,
    apply_window: tuple[float, float] | None,
    emission_suffix: str,
    smoothing_method: str,
) -> pd.DataFrame:
    smoothed = sequence_frame.copy()
    posterior = _normalize_probabilities(posterior)
    predictions = posterior.argmax(axis=1)

    smoothed.loc[:, prob_columns] = posterior
    if "emission_mode" in smoothed.columns:
        smoothed["base_emission_mode"] = smoothed["emission_mode"].astype(str)
        smoothed["emission_mode"] = smoothed["emission_mode"].map(lambda value: _smoothed_emission_mode(value, emission_suffix))
    else:
        smoothed["base_emission_mode"] = ""
        smoothed["emission_mode"] = emission_suffix

    smoothed["predicted_label"] = predictions.astype(int)
    smoothed["predicted_class"] = [class_names[int(index)] for index in predictions]
    smoothed["confidence"] = posterior.max(axis=1)
    if "true_label" in smoothed.columns:
        labels = _numeric_labels(smoothed, posterior.shape[1])
        smoothed["probability_true_class"] = posterior[np.arange(len(smoothed)), labels]
        smoothed["is_correct"] = predictions == labels

    smoothed["temporal_smoothing_method"] = smoothing_method
    smoothed["temporal_smoothing_stay_probability"] = float(stay_probability)
    if fit_window is not None:
        smoothed["temporal_smoothing_fit_window_start"] = float(fit_window[0])
        smoothed["temporal_smoothing_fit_window_stop"] = float(fit_window[1])
    else:
        smoothed["temporal_smoothing_fit_window_start"] = np.nan
        smoothed["temporal_smoothing_fit_window_stop"] = np.nan
    if apply_window is not None:
        smoothed["temporal_smoothing_apply_window_start"] = float(apply_window[0])
        smoothed["temporal_smoothing_apply_window_stop"] = float(apply_window[1])
    else:
        smoothed["temporal_smoothing_apply_window_start"] = np.nan
        smoothed["temporal_smoothing_apply_window_stop"] = np.nan

    if "model_hash" in smoothed.columns:
        smoothed["base_model_hash"] = smoothed["model_hash"].astype(str)
        smoothed["model_hash"] = smoothed["base_model_hash"].map(
            lambda base_hash: stable_hash(
                {
                    "base_model_hash": base_hash,
                    "temporal_smoothing_method": smoothing_method,
                    "temporal_smoothing_stay_probability": float(stay_probability),
                    "fit_window": None if fit_window is None else [float(fit_window[0]), float(fit_window[1])],
                    "apply_window": None if apply_window is None else [float(apply_window[0]), float(apply_window[1])],
                }
            )
        )
    return smoothed


def metrics_from_probability_observations(observations: pd.DataFrame, *, ece_bins: int = 10) -> pd.DataFrame:
    """Compute fold/time decoding metrics directly from probability observations.

    This is useful after temporal smoothing because the updated posterior probabilities are the result object;
    averaging old per-fold CSV metrics would silently discard the temporal posterior.
    """

    prob_columns = probability_columns(observations)
    n_classes = len(prob_columns)
    labels = _numeric_labels(observations, n_classes)
    working = observations.copy()
    working["true_label"] = labels
    for column in prob_columns:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    if working[prob_columns].isna().any().any():
        raise ValueError("prob_class_* columns must be numeric and non-missing.")

    group_columns = [column for column in METRIC_GROUP_COLUMNS if column in working.columns]
    if "time" not in group_columns:
        raise ValueError("Probability observations must contain a time column.")

    rows: list[dict[str, object]] = []
    for key, group in _iter_groups(working, group_columns):
        row = _group_values(group_columns, key)
        probabilities = _normalize_probabilities(group[prob_columns].to_numpy(dtype=float))
        group_labels = group["true_label"].to_numpy(dtype=int)
        predictions = probabilities.argmax(axis=1)
        class_names = _class_names(group, prob_columns)
        row.update(
            {
                "accuracy": accuracy_score(group_labels, predictions),
                "balanced_accuracy": balanced_accuracy_score(group_labels, predictions),
                "top2_accuracy": _top_k_accuracy(probabilities, group_labels, k=2),
                "top3_accuracy": _top_k_accuracy(probabilities, group_labels, k=3),
                "log_loss": log_loss(group_labels, probabilities, labels=np.arange(n_classes)),
                "brier": brier_score_multiclass(probabilities, group_labels),
                "ece": expected_calibration_error(probabilities, group_labels, n_bins=ece_bins),
                "n_test": int(len(group)),
                "n_classes": int(n_classes),
                "class_names": "|".join(map(str, class_names)),
            }
        )
        for optional_column in METRIC_PROVENANCE_COLUMNS:
            if optional_column not in group.columns:
                continue
            values = group[optional_column].dropna().astype(str).unique()
            if len(values) == 1:
                row[optional_column] = values[0]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def smooth_probability_observations(
    observation_csvs: list[Path],
    *,
    fit_window: tuple[float, float] | None = DEFAULT_FIT_WINDOW,
    stay_grid_size: int = 200,
    mode: str = "forward_backward",
    apply_window: tuple[float, float] | None = None,
    emission_suffix: str = DEFAULT_EMISSION_SUFFIX,
    ece_bins: int = 10,
    out_observations: Path | None = None,
    out_metrics: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replace decoder probabilities by sticky-switching temporal posteriors.

    The sticky transition probability is fit without labels from held-out probability observations within
    ``fit_window`` for each decoder/emission group, then applied to every complete sequence in that group.
    ``forward_backward`` uses all times in a sequence, ``forward_only`` only uses observations up to each
    time, and ``poststimulus_forward_only`` applies forward filtering only inside ``apply_window`` while
    leaving probabilities outside that window unchanged.
    The returned observation table preserves the NeuRepTrace probability-observation schema but changes
    ``prob_class_*`` to temporally smoothed posterior probabilities and updates prediction columns.
    """

    if mode not in SMOOTHING_MODE_CHOICES:
        raise ValueError(f"Unknown temporal smoothing mode '{mode}'. Available modes: {', '.join(SMOOTHING_MODE_CHOICES)}.")
    if mode == "poststimulus_forward_only" and apply_window is None:
        apply_window = DEFAULT_POSTSTIMULUS_APPLY_WINDOW
    elif mode != "poststimulus_forward_only":
        apply_window = None
    smoothing_method = SMOOTHING_METHODS[mode]

    observations = read_probability_observations(observation_csvs).copy()
    observations["__input_order"] = np.arange(len(observations))
    prob_columns = probability_columns(observations)
    group_columns = _smoothing_group_columns(observations) or _model_group_columns(observations)
    smoothed_frames: list[pd.DataFrame] = []

    for _, decoder_frame in _iter_groups(observations, group_columns):
        fit_frame = _filter_time_window(decoder_frame, fit_window) if fit_window is not None else decoder_frame.copy()
        fit_sequences = _sequences_from_frame(fit_frame, prob_columns)
        fit = fit_sticky_switching_model(fit_sequences, stay_grid_size=stay_grid_size)
        stay_probability = float(fit["best_stay_probability"])
        class_names = _class_names(decoder_frame, prob_columns)
        key_columns = sequence_key_columns(decoder_frame)
        validate_unique_sequence_times(decoder_frame, key_columns)

        for _, sequence_frame in decoder_frame.sort_values([*key_columns, "time"]).groupby(key_columns, sort=True, dropna=False):
            probabilities = _normalize_probabilities(sequence_frame[prob_columns].to_numpy(dtype=float))
            if len(probabilities) < 2:
                continue
            posterior = _smooth_sequence_posteriors(
                sequence_frame,
                probabilities,
                stay_probability=stay_probability,
                mode=mode,
                apply_window=apply_window,
            )
            smoothed_frames.append(
                _with_posterior_columns(
                    sequence_frame,
                    posterior,
                    prob_columns=prob_columns,
                    class_names=class_names,
                    stay_probability=stay_probability,
                    fit_window=fit_window,
                    apply_window=apply_window,
                    emission_suffix=emission_suffix,
                    smoothing_method=smoothing_method,
                )
            )

    if not smoothed_frames:
        raise ValueError("Need at least one sequence with two or more time points for temporal smoothing.")

    smoothed = pd.concat(smoothed_frames, ignore_index=True).sort_values("__input_order").drop(columns=["__input_order"]).reset_index(drop=True)
    metrics = metrics_from_probability_observations(smoothed, ece_bins=ece_bins)

    if out_observations is not None:
        out_observations.parent.mkdir(parents=True, exist_ok=True)
        smoothed.to_csv(out_observations, index=False)
    if out_metrics is not None:
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out_metrics, index=False)
    return smoothed, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply sticky temporal-model forward-backward smoothing to NeuRepTrace probability observation CSVs."
    )
    parser.add_argument("observation_csv", nargs="+", help="Observation CSVs or glob patterns emitted by --observations-out/--observation-dir.")
    parser.add_argument("--out-observations", type=Path, required=True, help="CSV path for temporally smoothed probability observations.")
    parser.add_argument("--out-metrics", type=Path, help="Optional fold/time metric CSV computed from smoothed posteriors.")
    parser.add_argument("--fit-window", nargs=2, type=float, default=DEFAULT_FIT_WINDOW, metavar=("START", "STOP"))
    parser.add_argument("--use-full-sequence-fit", action="store_true", help="Fit the sticky transition on every available time bin instead of --fit-window.")
    parser.add_argument("--stay-grid-size", type=int, default=200)
    parser.add_argument("--mode", choices=SMOOTHING_MODE_CHOICES, default="forward_backward")
    parser.add_argument(
        "--apply-window",
        nargs=2,
        type=float,
        default=DEFAULT_POSTSTIMULUS_APPLY_WINDOW,
        metavar=("START", "STOP"),
        help="Application window for poststimulus_forward_only mode.",
    )
    parser.add_argument("--emission-suffix", default=DEFAULT_EMISSION_SUFFIX)
    parser.add_argument("--ece-bins", type=int, default=10)
    args = parser.parse_args()

    fit_window = None if args.use_full_sequence_fit else tuple(args.fit_window)
    paths = _expand_paths(args.observation_csv)
    smoothed, metrics = smooth_probability_observations(
        paths,
        fit_window=fit_window,
        stay_grid_size=args.stay_grid_size,
        mode=args.mode,
        apply_window=tuple(args.apply_window),
        emission_suffix=args.emission_suffix,
        ece_bins=args.ece_bins,
        out_observations=args.out_observations,
        out_metrics=args.out_metrics,
    )
    print(f"Wrote smoothed probability observations: {args.out_observations}")
    if args.out_metrics is not None:
        print(f"Wrote smoothed metrics: {args.out_metrics}")
    print(f"Smoothed {len(smoothed)} observation row(s) into {len(metrics)} metric row(s).")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from neureptrace.observations import stable_hash
from neureptrace.temporal_model import probability_columns, read_probability_observations
from neureptrace.temporal_smoothing import metrics_from_probability_observations

DEFAULT_RESPONSE_TIMES = (0.088, 0.136, 0.184, 0.232)
ENSEMBLE_MODE_CHOICES = ("uniform", "source_oof_nonnegative")
COMBINE_CHOICES = ("log_probability_mean", "probability_mean")
OUTPUT_DECODER = "poststimulus_response_window_logit_ensemble"
OUTPUT_EMISSION_MODE = "response_window_logit_ensemble"
EPSILON = 1e-12
TARGET_GROUP_COLUMNS = ("subject", "group", "outer_test_groups", "session", "fold")


def _normalize_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    probabilities = np.clip(probabilities, EPSILON, None)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(shifted, -50.0, 50.0))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _time_label(value: float) -> str:
    return f"{float(value):.12g}"


def _nearest_times(available_times: np.ndarray, requested_times: tuple[float, ...]) -> tuple[float, ...]:
    if available_times.size == 0:
        raise ValueError("Observation table does not contain any time values.")
    selected = []
    for requested in requested_times:
        selected.append(float(available_times[np.argmin(np.abs(available_times - requested))]))
    return tuple(selected)


def _has_nonempty_values(series: pd.Series) -> bool:
    if series.dropna().empty:
        return False
    return series.dropna().astype(str).str.strip().ne("").any()


def _subject_key_column(frame: pd.DataFrame) -> str | None:
    for column in TARGET_GROUP_COLUMNS:
        if column in frame.columns and _has_nonempty_values(frame[column]):
            return column
    return None


def _sequence_key_columns(frame: pd.DataFrame) -> list[str]:
    keys = []
    subject_column = _subject_key_column(frame)
    if subject_column is not None:
        keys.append(subject_column)
    if "fold" in frame.columns and "fold" not in keys:
        keys.append("fold")
    if "sample_index" in frame.columns:
        keys.append("sample_index")
    elif "sequence_id" in frame.columns:
        keys.append("sequence_id")
    else:
        raise ValueError("Observation table needs sample_index or sequence_id columns.")
    return keys


def _target_subject_values(base: pd.DataFrame, key_columns: list[str]) -> np.ndarray:
    subject_column = _subject_key_column(base)
    if subject_column is not None:
        return base[subject_column].astype(str).to_numpy()
    for column in TARGET_GROUP_COLUMNS:
        if column in key_columns:
            return base.index.get_level_values(column).astype(str).to_numpy()
    return np.full(len(base), "", dtype=object)


def _candidate_weights(n_times: int, step: float) -> np.ndarray:
    if n_times < 1:
        raise ValueError("Need at least one response time.")
    if n_times == 1:
        return np.ones((1, 1), dtype=float)
    levels = np.arange(0.0, 1.0 + step / 2.0, step)
    weights = []
    for candidate in product(levels, repeat=n_times):
        candidate = np.asarray(candidate, dtype=float)
        total = float(candidate.sum())
        if abs(total - 1.0) <= step / 2.0 and total > 0.0:
            weights.append(candidate / total)
    if not weights:
        raise ValueError("Weight grid is empty; use a larger --weight-grid-step.")
    return np.unique(np.vstack(weights).round(12), axis=0)


def _combine_logits(probability_cube: np.ndarray, weights: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probability_cube, EPSILON, 1.0))
    weighted_logits = np.tensordot(logits, weights, axes=([1], [0]))
    return _softmax(weighted_logits)


def _combine_probabilities(probability_cube: np.ndarray, weights: np.ndarray, *, combine: str) -> np.ndarray:
    if combine == "log_probability_mean":
        return _combine_logits(probability_cube, weights)
    if combine == "probability_mean":
        probabilities = np.tensordot(probability_cube, weights, axes=([1], [0]))
        return _normalize_rows(probabilities)
    raise ValueError(f"Unknown response-window combine mode '{combine}'.")


def _score_weights(probability_cube: np.ndarray, labels: np.ndarray, weights: np.ndarray, *, combine: str) -> float:
    probabilities = _combine_probabilities(probability_cube, weights, combine=combine)
    return float(balanced_accuracy_score(labels, probabilities.argmax(axis=1)))


def _learn_weights(
    probability_cube: np.ndarray,
    labels: np.ndarray,
    candidates: np.ndarray,
    *,
    combine: str,
) -> tuple[np.ndarray, float]:
    best_weights = candidates[0]
    best_score = -np.inf
    for weights in candidates:
        score = _score_weights(probability_cube, labels, weights, combine=combine)
        if score > best_score:
            best_weights = weights
            best_score = score
    return best_weights, float(best_score)


def _response_window_rows(
    observations: pd.DataFrame,
    *,
    requested_times: tuple[float, ...],
    mode: str,
    combine: str,
    weight_grid_step: float,
    output_time: float | None,
) -> pd.DataFrame:
    if mode not in ENSEMBLE_MODE_CHOICES:
        raise ValueError(f"Unknown response-window ensemble mode '{mode}'.")
    if combine not in COMBINE_CHOICES:
        raise ValueError(f"Unknown response-window combine mode '{combine}'.")
    prob_columns = list(probability_columns(observations))
    times = _nearest_times(pd.to_numeric(observations["time"], errors="coerce").dropna().unique(), requested_times)
    selected = observations.loc[observations["time"].astype(float).isin(times)].copy()
    key_columns = _sequence_key_columns(selected)
    metadata_columns = [
        column
        for column in selected.columns
        if column not in {*prob_columns, "time", "test_time", "window_start", "window_stop", "train_time"}
    ]
    time_labels = [_time_label(time) for time in times]
    candidates = _candidate_weights(len(times), weight_grid_step)
    subject_column = _subject_key_column(selected)
    target_subjects = selected[subject_column].dropna().astype(str).unique().tolist() if subject_column else [""]
    output_rows: list[pd.DataFrame] = []

    wide_probabilities = {}
    for time in times:
        time_frame = selected.loc[selected["time"].astype(float) == float(time)].copy()
        time_frame = time_frame.sort_values(key_columns).drop_duplicates(key_columns, keep="first")
        wide_probabilities[time] = time_frame.set_index(key_columns)[prob_columns]
    common_index = None
    for frame in wide_probabilities.values():
        common_index = frame.index if common_index is None else common_index.intersection(frame.index)
    if common_index is None or len(common_index) == 0:
        raise ValueError("No observations contain all requested response-window times.")

    base = selected.sort_values(key_columns).drop_duplicates(key_columns, keep="first").set_index(key_columns).loc[common_index]
    labels = pd.to_numeric(base["true_label"], errors="raise").to_numpy(dtype=int)
    probability_cube = np.stack(
        [_normalize_rows(wide_probabilities[time].loc[common_index].to_numpy(dtype=float)) for time in times],
        axis=1,
    )
    subjects = _target_subject_values(base, key_columns)

    for target_subject in target_subjects:
        target_mask = subjects == str(target_subject)
        if not np.any(target_mask):
            continue
        if mode == "uniform":
            weights = np.full(len(times), 1.0 / len(times), dtype=float)
            source_score = np.nan
        else:
            source_mask = ~target_mask
            if not np.any(source_mask):
                weights = np.full(len(times), 1.0 / len(times), dtype=float)
                source_score = np.nan
            else:
                weights, source_score = _learn_weights(
                    probability_cube[source_mask],
                    labels[source_mask],
                    candidates,
                    combine=combine,
                )

        probabilities = _combine_probabilities(probability_cube[target_mask], weights, combine=combine)
        target_base = base.iloc[np.flatnonzero(target_mask)].reset_index()
        target_base = target_base[[column for column in metadata_columns if column in target_base.columns]].copy()
        predictions = probabilities.argmax(axis=1)
        for class_index, column in enumerate(prob_columns):
            target_base[column] = probabilities[:, class_index]
        class_names = []
        for class_index in range(len(prob_columns)):
            class_column = f"class_{class_index}"
            if class_column in target_base.columns and target_base[class_column].notna().any():
                class_names.append(str(target_base[class_column].dropna().iloc[0]))
            else:
                class_names.append(str(class_index))
        true_labels = pd.to_numeric(target_base["true_label"], errors="raise").to_numpy(dtype=int)
        target_base["predicted_label"] = predictions.astype(int)
        target_base["predicted_class"] = [class_names[int(label)] for label in predictions]
        target_base["probability_true_class"] = probabilities[np.arange(len(probabilities)), true_labels]
        target_base["confidence"] = probabilities.max(axis=1)
        target_base["is_correct"] = predictions == true_labels
        target_base["decoder"] = OUTPUT_DECODER
        target_base["emission_mode"] = f"{OUTPUT_EMISSION_MODE}_{mode}"
        target_base["time"] = float(np.mean(times) if output_time is None else output_time)
        target_base["test_time"] = target_base["time"]
        target_base["train_time"] = target_base["time"]
        target_base["window_start"] = float(min(times))
        target_base["window_stop"] = float(max(times))
        target_base["response_window_mode"] = mode
        target_base["response_window_requested_times"] = "|".join(_time_label(time) for time in requested_times)
        target_base["response_window_actual_times"] = "|".join(time_labels)
        target_base["response_window_weights"] = "|".join(f"{float(weight):.12g}" for weight in weights)
        target_base["response_window_combine"] = combine
        target_base["response_window_source_score"] = "" if not np.isfinite(source_score) else float(source_score)
        target_base["model_hash"] = stable_hash(
            {
                "decoder": OUTPUT_DECODER,
                "mode": mode,
                "combine": combine,
                "requested_times": requested_times,
                "actual_times": times,
                "weights": tuple(float(weight) for weight in weights),
                "target_subject": target_subject,
            }
        )
        output_rows.append(target_base)

    if not output_rows:
        raise ValueError("No response-window ensemble rows were produced.")
    return pd.concat(output_rows, ignore_index=True)


def run_response_window_ensemble(
    observation_csvs: list[Path],
    *,
    response_times: tuple[float, ...] = DEFAULT_RESPONSE_TIMES,
    mode: str = "uniform",
    combine: str = "log_probability_mean",
    weight_grid_step: float = 0.1,
    output_time: float | None = 0.184,
    ece_bins: int = 10,
    out_observations: Path | None = None,
    out_metrics: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    observations = read_probability_observations(observation_csvs).copy()
    ensembled = _response_window_rows(
        observations,
        requested_times=tuple(float(time) for time in response_times),
        mode=mode,
        combine=combine,
        weight_grid_step=float(weight_grid_step),
        output_time=output_time,
    )
    metrics = metrics_from_probability_observations(ensembled, ece_bins=ece_bins)
    if out_observations is not None:
        out_observations.parent.mkdir(parents=True, exist_ok=True)
        ensembled.to_csv(out_observations, index=False)
    if out_metrics is not None:
        out_metrics.parent.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out_metrics, index=False)
    return ensembled, metrics


def _parse_times(text: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in text.replace(";", ",").split(",") if part.strip())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a post-stimulus response-window logit ensemble from probability observations.")
    parser.add_argument("observation_csv", nargs="+", type=Path)
    parser.add_argument("--mode", choices=ENSEMBLE_MODE_CHOICES, default="uniform")
    parser.add_argument("--combine", choices=COMBINE_CHOICES, default="log_probability_mean")
    parser.add_argument("--response-times", default=",".join(str(time) for time in DEFAULT_RESPONSE_TIMES))
    parser.add_argument("--output-time", type=float, default=0.184)
    parser.add_argument("--weight-grid-step", type=float, default=0.1)
    parser.add_argument("--ece-bins", type=int, default=10)
    parser.add_argument("--out-observations", type=Path, required=True)
    parser.add_argument("--out-metrics", type=Path, required=True)
    args = parser.parse_args(argv)

    _, metrics = run_response_window_ensemble(
        args.observation_csv,
        response_times=_parse_times(args.response_times),
        mode=args.mode,
        combine=args.combine,
        weight_grid_step=args.weight_grid_step,
        output_time=args.output_time,
        ece_bins=args.ece_bins,
        out_observations=args.out_observations,
        out_metrics=args.out_metrics,
    )
    print(f"Wrote response-window observations: {args.out_observations}")
    print(f"Wrote response-window metrics: {args.out_metrics}")
    if not metrics.empty:
        print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()

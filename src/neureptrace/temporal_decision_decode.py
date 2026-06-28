"""Grouped temporal-decision decoder for source-only M/EEG benchmarks."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import LabelEncoder

from neureptrace.dataset_config import apply_overrides, effective_config, load_config, load_epoch_dataset_from_config
from neureptrace.decode_from_config import _bool_value, _resolve_output, _section, _window_ms, _write_provenance_sidecars
from neureptrace.decoding import make_decoder, normalize_decoder_name, normalize_emission_mode, normalize_feature_preprocessor, normalize_pca_components, predict_emission_probabilities, time_windows
from neureptrace.io.dataset import EpochDataset
from neureptrace.metrics import brier_score_multiclass, expected_calibration_error
from neureptrace.mne_time_decode import (
    DEFAULT_BASELINE_WINDOW,
    _align_probability_columns,
    _features_for_window,
    _normalize_baseline_window,
    _normalize_positive_float,
    _normalize_positive_int,
    _top_k_accuracy,
    normalize_epoch_normalization,
)
from neureptrace.mne_time_decode_foldlocal import _normalize_epoch_data_for_fold
from neureptrace.observations import ProbabilityObservationTable, stable_hash

ENSEMBLE_ALIASES = {"logistic_svm_ensemble", "logistic-svm-ensemble", "logistic_linear_svm_ensemble", "logistic-linear-svm-ensemble"}
DEFAULT_ENSEMBLE_DECODERS = ("multinomial-logistic", "linear_svm")
DEFAULT_PROBABILITY_TOLERANCE = 1.0e-3
TimeWindow = tuple[int, int, float]


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()] if "," in value else [value]
    return list(value) if isinstance(value, (list, tuple)) else [value]


def _pair(value: Any, *, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        value = [part.strip() for comma_part in text.split(",") for part in comma_part.split() if part.strip()]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numbers.")
    start = _finite_float(value[0], name=name)
    stop = _finite_float(value[1], name=name)
    if stop < start:
        raise ValueError(f"{name} stop must be greater than or equal to start.")
    return start, stop


def _baseline_window(value: Any, *, name: str = "baseline_window") -> tuple[float, float]:
    """Normalize a baseline window accepted from Python, YAML, or ``--set`` overrides."""

    if value is None:
        return _normalize_baseline_window(None)
    if isinstance(value, str):
        parsed = _pair(value, name=name)
        if parsed is None:
            return _normalize_baseline_window(None)
        return parsed
    return _normalize_baseline_window(value)


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite.")
    parsed = float(value)
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite.")
    return parsed


def _normalize_min_probability(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("min_probability must lie in (0, 1).")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0 or parsed >= 1.0:
        raise ValueError("min_probability must lie in (0, 1).")
    return parsed


def _normalize_aggregation(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"log", "log_mean", "geometric_mean"}:
        return "log_mean"
    if normalized in {"mean", "arithmetic_mean", "probability", "probability_mean"}:
        return "mean"
    raise ValueError("aggregation must be one of: log_mean, geometric_mean, mean, arithmetic_mean.")


def _decoders(value: Sequence[str] | str | None) -> tuple[str, ...]:
    expanded: list[str] = []
    for name in _list(value) or ["logistic"]:
        if str(name).strip().lower().replace("_", "-") in {alias.replace("_", "-") for alias in ENSEMBLE_ALIASES}:
            expanded.extend(DEFAULT_ENSEMBLE_DECODERS)
        else:
            expanded.append(normalize_decoder_name(str(name)))
    return tuple(dict.fromkeys(expanded))


def _windows(times: np.ndarray, *, window_ms: float, step_ms: float, tmin: Any, tmax: Any, decision_window: tuple[float, float] | None) -> list[TimeWindow]:
    window_ms = _normalize_positive_float(window_ms, name="window_ms")
    step_ms = _normalize_positive_float(step_ms, name="step_ms")
    lower = -np.inf if tmin is None else _finite_float(tmin, name="tmin")
    upper = np.inf if tmax is None else _finite_float(tmax, name="tmax")
    if upper < lower:
        raise ValueError("tmax must be greater than or equal to tmin.")
    windows = [window for window in time_windows(times, window_ms=window_ms, step_ms=step_ms) if lower <= window[2] <= upper]
    if decision_window is not None:
        eps = 1.0e-12
        windows = [
            window
            for window in windows
            if float(times[window[1] - 1]) >= decision_window[0] - eps
            and float(times[window[0]]) <= decision_window[1] + eps
        ]
    if not windows:
        raise ValueError("No temporal decision windows are available after filtering.")
    return windows


def _combine(probabilities: Sequence[np.ndarray], *, mode: str, min_probability: float) -> np.ndarray:
    if not probabilities:
        raise ValueError("At least one probability matrix is required for temporal decision aggregation.")
    min_probability = _normalize_min_probability(min_probability)
    aggregation = _normalize_aggregation(mode)
    stack = np.stack(probabilities, axis=0)
    if stack.ndim != 3:
        raise ValueError("Temporal decision probabilities must have shape (n_sources, n_samples, n_classes).")
    if not np.isfinite(stack).all():
        raise ValueError("Temporal decision probabilities must be finite.")
    if bool((stack < 0.0).any()):
        raise ValueError("Temporal decision probabilities must be non-negative.")
    if bool((stack > 1.0).any()):
        raise ValueError("Temporal decision probabilities must not exceed 1.0.")
    row_sums = stack.sum(axis=2)
    bad_rows = np.flatnonzero(np.abs(row_sums - 1.0) > DEFAULT_PROBABILITY_TOLERANCE)
    if len(bad_rows):
        examples = [float(row_sums.ravel()[index]) for index in bad_rows[:5]]
        raise ValueError(
            "Temporal decision probability rows must sum to 1.0 within tolerance "
            f"{DEFAULT_PROBABILITY_TOLERANCE:g}; example row sums: {examples}"
        )
    if aggregation == "log_mean":
        scores = np.mean(np.log(np.clip(stack, min_probability, 1.0)), axis=0)
        scores -= scores.max(axis=1, keepdims=True)
        combined = np.exp(np.clip(scores, -745.0, 0.0))
    else:
        combined = np.mean(stack, axis=0)
    row_sums = combined.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Combined probabilities must have positive row sums.")
    return combined / row_sums


def _components(feature_preprocessor: str, pca_components: Any) -> int | float | None:
    if feature_preprocessor == "none":
        if pca_components is not None:
            raise ValueError("pca_components requires pca, pca_whiten, or anova_select preprocessing.")
        return None
    if feature_preprocessor == "anova_select":
        from neureptrace.decoding import normalize_anova_select_percentile
        return normalize_anova_select_percentile(pca_components)
    return normalize_pca_components(pca_components)


def _observation_rows(probabilities: np.ndarray, labels: np.ndarray, original_indices: np.ndarray, class_names: np.ndarray, *, fold: int, heldout_group: str, split_id: str, decoder: str, emission_mode: str, preprocessing_hash: str, model_hash: str) -> list[dict[str, Any]]:
    predictions = probabilities.argmax(axis=1)
    rows = []
    for row_index, original_index in enumerate(original_indices):
        true_label = int(labels[row_index])
        predicted_label = int(predictions[row_index])
        row: dict[str, Any] = {
            "fold": int(fold), "heldout_group": heldout_group, "decoder": decoder, "emission_mode": emission_mode,
            "temporal_mode": "temporal_decision_ensemble", "sample_index": int(original_index), "sequence_id": int(original_index),
            "true_label": true_label, "true_class": str(class_names[true_label]), "predicted_label": predicted_label,
            "predicted_class": str(class_names[predicted_label]), "probability_true_class": float(probabilities[row_index, true_label]),
            "confidence": float(probabilities[row_index].max()), "is_correct": bool(predicted_label == true_label),
            "backend": "sklearn", "split_id": split_id, "seed": 13, "calibration_fold": "",
            "preprocessing_hash": preprocessing_hash, "model_hash": model_hash,
        }
        for class_index, class_name in enumerate(class_names):
            row[f"class_{class_index}"] = str(class_name)
            row[f"prob_class_{class_index}"] = float(probabilities[row_index, class_index])
        rows.append(row)
    return rows


def run_temporal_decision_decode_dataset(dataset: EpochDataset, *, label_column: str, out_path: Path, group_column: str | None = None, decoders: Sequence[str] | str | None = None, tmin: float | None = None, tmax: float | None = None, window_ms: float = 100.0, step_ms: float = 25.0, test_window: tuple[float, float] | None = None, max_iter: int = 1000, emission_mode: str = "calibrated", feature_preprocessor: str = "none", pca_components: int | float | str | None = None, normalization: str = "none", baseline_window: Any = DEFAULT_BASELINE_WINDOW, aggregation: str = "log_mean", min_probability: float = 1e-12, observation_out_path: Path | None = None, calibration_bins: int = 10) -> pd.DataFrame:
    """Evaluate one temporally aggregated decision per trial with LOSO splits."""
    if group_column is None:
        raise ValueError("temporal decision decoding requires group_column for leave-one-group-out evaluation.")
    if label_column not in dataset.metadata.columns or group_column not in dataset.metadata.columns:
        raise ValueError("Configured label_column/group_column is missing from dataset metadata.")

    keep = pd.notna(dataset.metadata[label_column].to_numpy()) & pd.notna(dataset.metadata[group_column].to_numpy())
    if not bool(np.any(keep)):
        raise ValueError(
            "temporal decision decoding found no rows with non-missing "
            f"{label_column!r} and {group_column!r} values."
        )
    data = np.asarray(dataset.data[keep], dtype=float)
    metadata = dataset.metadata.loc[keep].reset_index(drop=True)
    original_indices = np.arange(len(dataset.metadata))[keep]
    label_values = metadata[label_column].to_numpy()
    groups = metadata[group_column].to_numpy()
    encoder = LabelEncoder()
    labels = encoder.fit_transform(label_values)
    class_names = encoder.classes_
    if len(class_names) < 2:
        raise ValueError("temporal decision decoding requires at least two classes after dropping missing label/group rows.")
    if len(pd.unique(groups)) < 2:
        raise ValueError("temporal decision decoding requires at least two groups after dropping missing label/group rows.")
    classes = np.arange(len(class_names))

    decoder_names = _decoders(decoders)
    decoder_label = "+".join(decoder_names)
    emission_mode = normalize_emission_mode(emission_mode)
    feature_preprocessor_name = normalize_feature_preprocessor(feature_preprocessor)
    pca_components_value = _components(feature_preprocessor_name, pca_components)
    normalization_name = normalize_epoch_normalization(normalization)
    baseline_window_value = _baseline_window(baseline_window)
    max_iter = _normalize_positive_int(max_iter, name="max_iter")
    calibration_bins = _normalize_positive_int(calibration_bins, name="calibration_bins")
    aggregation = _normalize_aggregation(aggregation)
    min_probability = _normalize_min_probability(min_probability)
    test_window = _pair(test_window, name="test_window")
    decision_windows = _windows(dataset.times, window_ms=window_ms, step_ms=step_ms, tmin=tmin, tmax=tmax, decision_window=test_window)
    split_id = "leave-one-group-out"
    preprocessing_hash = stable_hash({"dataset": dataset.name, "label_column": label_column, "group_column": group_column, "window_ms": window_ms, "step_ms": step_ms, "test_window": test_window, "feature_preprocessor": feature_preprocessor_name, "pca_components": pca_components_value, "normalization": normalization_name, "baseline_window": baseline_window_value})

    rows: list[dict[str, Any]] = []
    obs_rows: list[dict[str, Any]] = []
    for fold, (train_idx, test_idx) in enumerate(LeaveOneGroupOut().split(np.zeros(len(labels)), labels, groups)):
        if set(labels[test_idx]) - set(labels[train_idx]):
            raise ValueError(f"Fold {fold} holds out a class that is absent from training.")
        heldout_group = "|".join(sorted(map(str, np.unique(groups[test_idx]))))
        fold_data = _normalize_epoch_data_for_fold(data, dataset.times, normalization_name, baseline_window=baseline_window_value, train_idx=train_idx)
        feature_cache = {window: _features_for_window(fold_data, window) for window in decision_windows}
        probability_stack: list[np.ndarray] = []
        for window in decision_windows:
            for decoder_name in decoder_names:
                model = make_decoder(decoder_name, max_iter=max_iter, emission_mode=emission_mode, feature_preprocessor=feature_preprocessor_name, pca_components=pca_components_value)
                model.fit(feature_cache[window][train_idx], labels[train_idx])
                probability_stack.append(_align_probability_columns(predict_emission_probabilities(model, feature_cache[window][test_idx], emission_mode=emission_mode), model=model, classes=classes))
        probabilities = _combine(probability_stack, mode=aggregation, min_probability=min_probability)
        predictions = probabilities.argmax(axis=1)
        model_hash = stable_hash({"decoder": decoder_label, "emission_mode": emission_mode, "fold": int(fold), "heldout_group": heldout_group, "decision_centers": [window[2] for window in decision_windows], "aggregation": aggregation})
        row: dict[str, Any] = {
            "fold": int(fold), "heldout_group": heldout_group, "split_id": split_id, "decoder": decoder_label,
            "source_decoders": "|".join(decoder_names), "emission_mode": emission_mode, "feature_preprocessor": feature_preprocessor_name,
            "pca_components": "" if pca_components_value is None else pca_components_value, "normalization": normalization_name,
            "baseline_window_start": baseline_window_value[0], "baseline_window_stop": baseline_window_value[1],
            "temporal_mode": "temporal_decision_ensemble", "temporal_aggregation": aggregation,
            "test_window_start": float(min(dataset.times[window[0]] for window in decision_windows)),
            "test_window_stop": float(max(dataset.times[window[1] - 1] for window in decision_windows)),
            "n_test_windows": len(decision_windows), "n_source_decoders": len(decoder_names),
            "accuracy": accuracy_score(labels[test_idx], predictions), "balanced_accuracy": balanced_accuracy_score(labels[test_idx], predictions),
            "top2_accuracy": _top_k_accuracy(probabilities, labels[test_idx], k=2), "top3_accuracy": _top_k_accuracy(probabilities, labels[test_idx], k=3),
            "log_loss": log_loss(labels[test_idx], probabilities, labels=classes), "brier": brier_score_multiclass(probabilities, labels[test_idx]),
            "ece": expected_calibration_error(probabilities, labels[test_idx], n_bins=calibration_bins), "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)), "n_classes": int(len(classes)), "class_names": "|".join(map(str, class_names)),
            "preprocessing_hash": preprocessing_hash, "model_hash": model_hash,
        }
        rows.append(row)
        if observation_out_path is not None:
            obs_rows.extend(_observation_rows(probabilities, labels[test_idx], original_indices[test_idx], class_names, fold=fold, heldout_group=heldout_group, split_id=split_id, decoder=decoder_label, emission_mode=emission_mode, preprocessing_hash=preprocessing_hash, model_hash=model_hash))

    results = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    if observation_out_path is not None:
        observation_out_path.parent.mkdir(parents=True, exist_ok=True)
        ProbabilityObservationTable(pd.DataFrame(obs_rows)).standardized(defaults={"backend": "sklearn", "split_id": split_id}).to_csv(observation_out_path)
    return results


def _outputs(config: Mapping[str, Any], *, config_dir: Path) -> tuple[Path, Path | None]:
    summary = _resolve_output(config, config_dir=config_dir, key="summary_csv", default="results/{dataset}_temporal_decision_summary.csv")
    observations = _resolve_output(config, config_dir=config_dir, key="observations_csv")
    if summary is None:
        raise ValueError("outputs.summary_csv is required.")
    return summary, observations


def _decoder_request(config: Mapping[str, Any]) -> Sequence[str] | str | None:
    temporal = _section(config, "temporal_decision")
    decoding = _section(config, "decoding") or _section(config, "workflow")
    return temporal.get("decoders", temporal.get("classifier", decoding.get("decoders", decoding.get("classifier", decoding.get("decoder", "logistic")))))


def run_temporal_decision_decode_from_config(config_path: str | Path, *, overrides: Sequence[str] | None = None, write_provenance: bool | None = None) -> pd.DataFrame:
    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    if write_provenance is not None:
        config.setdefault("outputs", {})["provenance"] = _bool_value(write_provenance, name="write_provenance")
    dataset = load_epoch_dataset_from_config(config, base_dir=config_path.parent, check_files=True)
    preprocessing = _section(config, "preprocessing")
    decoding = _section(config, "decoding") or _section(config, "workflow")
    temporal = _section(config, "temporal_decision")
    summary_out, observations_out = _outputs(config, config_dir=config_path.parent)
    baseline_window = preprocessing.get("baseline_window", decoding.get("baseline_window", DEFAULT_BASELINE_WINDOW))
    results = run_temporal_decision_decode_dataset(
        dataset,
        label_column=str(temporal.get("label_column", decoding.get("label_column"))),
        group_column=temporal.get("group_column", decoding.get("group_column")),
        out_path=summary_out,
        decoders=_decoder_request(config),
        tmin=preprocessing.get("tmin"),
        tmax=preprocessing.get("tmax"),
        window_ms=_window_ms(preprocessing, key_ms="window_ms", key_seconds="window_size", default=100.0),
        step_ms=_window_ms(preprocessing, key_ms="step_ms", key_seconds="window_step", default=25.0),
        test_window=_pair(temporal.get("test_window", temporal.get("decision_window")), name="temporal_decision.test_window"),
        max_iter=_normalize_positive_int(temporal.get("max_iter", decoding.get("max_iter", 1000)), name="temporal_decision.max_iter"),
        emission_mode=temporal.get("emission_mode", decoding.get("emission_mode", "calibrated")),
        feature_preprocessor=temporal.get("feature_preprocessor", decoding.get("feature_preprocessor", preprocessing.get("feature_preprocessor", "none"))),
        pca_components=temporal.get("pca_components", decoding.get("pca_components", preprocessing.get("pca_components"))),
        normalization=temporal.get("normalization", preprocessing.get("normalization", "none")),
        baseline_window=baseline_window,
        aggregation=temporal.get("aggregation", "log_mean"),
        min_probability=_normalize_min_probability(temporal.get("min_probability", 1e-12)),
        observation_out_path=observations_out,
        calibration_bins=_normalize_positive_int(
            temporal.get("calibration_bins", decoding.get("calibration_bins", 10)),
            name="temporal_decision.calibration_bins",
        ),
    )
    _write_provenance_sidecars(config, config_path=config_path, config_dir=config_path.parent, output_paths=[path for path in (summary_out, observations_out) if path is not None])
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run grouped temporal decision decoding from a NeuRepTrace dataset config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--print-effective-config", action="store_true")
    parser.add_argument("--no-provenance", action="store_true")
    args = parser.parse_args(argv)
    config = apply_overrides(load_config(args.config), args.overrides)
    if args.print_effective_config:
        print(json.dumps(effective_config(config, base_dir=args.config.parent), indent=2, sort_keys=True, default=str))
        return 0
    results = run_temporal_decision_decode_from_config(args.config, overrides=args.overrides, write_provenance=not args.no_provenance)
    print(f"Wrote {len(results)} temporal decision rows from {args.config}")
    if not results.empty:
        print(f"Mean balanced_accuracy: {results['balanced_accuracy'].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

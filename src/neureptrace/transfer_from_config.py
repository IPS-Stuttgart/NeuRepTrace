"""Run configured train/test transfer decoding from a dataset config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder

from neureptrace.dataset_config import apply_overrides, effective_config, load_config, load_epoch_dataset_from_config
from neureptrace.decode_from_config import _resolve_output, _section, _window_ms, _write_provenance_sidecars
from neureptrace.decoding import (
    make_decoder,
    normalize_decoder_name,
    normalize_emission_mode,
    normalize_feature_preprocessor,
    normalize_pca_components,
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
)
from neureptrace.observations import ProbabilityObservationTable, stable_hash


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _filter_mask(metadata: pd.DataFrame, filter_spec: Mapping[str, Any] | None, *, name: str) -> np.ndarray:
    if not filter_spec:
        raise ValueError(f"transfer.{name}_filter is required.")
    mask = np.ones(len(metadata), dtype=bool)
    for column_name, expected in filter_spec.items():
        if column_name not in metadata.columns:
            raise ValueError(f"transfer.{name}_filter references unknown metadata column '{column_name}'.")
        mask &= metadata[column_name].isin(_as_list(expected)).to_numpy()
    if not np.any(mask):
        raise ValueError(f"transfer.{name}_filter selected no trials: {dict(filter_spec)}")
    return mask


def _transfer_section(config: Mapping[str, Any]) -> dict[str, Any]:
    transfer = config.get("transfer") or config.get("workflow") or {}
    if not isinstance(transfer, dict):
        raise ValueError("Config section 'transfer' must be a mapping.")
    return dict(transfer)


def _selected_config(config: Mapping[str, Any]) -> dict[str, Any]:
    preprocessing = _section(config, "preprocessing")
    transfer = _transfer_section(config)
    return {"preprocessing": preprocessing, "transfer": transfer}


def _transfer_output_paths(config: Mapping[str, Any], *, config_dir: Path) -> tuple[Path, Path | None]:
    summary = _resolve_output(config, config_dir=config_dir, key="summary_csv", default="results/{dataset}_transfer_summary.csv")
    observations = _resolve_output(config, config_dir=config_dir, key="observations_csv")
    if summary is None:
        raise ValueError("transfer-from-config requires outputs.summary_csv.")
    return summary, observations


def _observation_rows(
    *,
    metadata: pd.DataFrame,
    test_indices: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
    class_names: np.ndarray,
    window,
    decoder_name: str,
    emission_mode: str,
    preprocessing_hash: str,
    model_hash: str,
    train_filter: Mapping[str, Any],
    test_filter: Mapping[str, Any],
    window_start: float,
    window_stop: float,
) -> list[dict[str, Any]]:
    start, stop, center = window
    predictions = probabilities.argmax(axis=1)
    rows = []
    for local_position, sample_index in enumerate(test_indices):
        true_label = int(labels[sample_index])
        predicted_label = int(predictions[local_position])
        row = {
            "fold": 0,
            "decoder": decoder_name,
            "emission_mode": emission_mode,
            "temporal_mode": "configured_transfer",
            "train_filter": json.dumps(train_filter, sort_keys=True, default=str),
            "test_filter": json.dumps(test_filter, sort_keys=True, default=str),
            "time": center,
            "test_time": center,
            "window_start": window_start,
            "window_stop": window_stop,
            "sample_index": int(sample_index),
            "sequence_id": int(sample_index),
            "session": metadata.iloc[sample_index].get("session", ""),
            "true_label": true_label,
            "true_class": str(class_names[true_label]),
            "predicted_label": predicted_label,
            "predicted_class": str(class_names[predicted_label]),
            "probability_true_class": float(probabilities[local_position, true_label]),
            "confidence": float(probabilities[local_position].max()),
            "is_correct": bool(predicted_label == true_label),
            "backend": "sklearn",
            "split_id": "configured-transfer",
            "seed": 13,
            "calibration_fold": "",
            "preprocessing_hash": preprocessing_hash,
            "model_hash": model_hash,
        }
        for class_index, class_name in enumerate(class_names):
            row[f"class_{class_index}"] = str(class_name)
            row[f"prob_class_{class_index}"] = float(probabilities[local_position, class_index])
        rows.append(row)
    return rows


def run_transfer_from_config(
    config_path: str | Path,
    *,
    overrides: Sequence[str] | None = None,
    write_provenance: bool | None = None,
) -> pd.DataFrame:
    """Train on one configured subset and evaluate on another."""

    config_path = Path(config_path)
    config = apply_overrides(load_config(config_path), overrides)
    if write_provenance is not None:
        config.setdefault("outputs", {})["provenance"] = bool(write_provenance)
    dataset = load_epoch_dataset_from_config(config, base_dir=config_path.parent, check_files=True)
    preprocessing = _section(config, "preprocessing")
    transfer = _transfer_section(config)

    label_column = transfer.get("label_column") or _section(config, "decoding").get("label_column")
    if not label_column:
        raise ValueError("transfer-from-config requires transfer.label_column or decoding.label_column.")
    if label_column not in dataset.metadata.columns:
        raise ValueError(f"Label column '{label_column}' not found in metadata.")

    train_filter = transfer.get("train_filter") or {"split": "main"}
    test_filter = transfer.get("test_filter") or {"split": "cue"}
    train_mask = _filter_mask(dataset.metadata, train_filter, name="train")
    test_mask = _filter_mask(dataset.metadata, test_filter, name="test")
    if np.any(train_mask & test_mask):
        raise ValueError("transfer train_filter and test_filter overlap; use disjoint train/test subsets.")

    encoder = LabelEncoder()
    labels = encoder.fit_transform(dataset.metadata[label_column].to_numpy())
    classes = np.arange(len(encoder.classes_))
    decoder_name = normalize_decoder_name(transfer.get("decoder", transfer.get("classifier", _section(config, "decoding").get("classifier", "logistic"))))
    emission_mode = normalize_emission_mode(transfer.get("emission_mode", _section(config, "decoding").get("emission_mode", "calibrated")))
    feature_preprocessor = normalize_feature_preprocessor(
        transfer.get("feature_preprocessor", preprocessing.get("feature_preprocessor", "none"))
    )
    pca_components = None if feature_preprocessor == "none" else normalize_pca_components(transfer.get("pca_components", preprocessing.get("pca_components")))
    max_iter = int(transfer.get("max_iter", _section(config, "decoding").get("max_iter", 1000)))
    baseline_window = _normalize_baseline_window(preprocessing.get("baseline_window", DEFAULT_BASELINE_WINDOW))
    data = _apply_epoch_normalization(
        dataset.data,
        dataset.times,
        preprocessing.get("normalization", "none"),
        baseline_window=baseline_window,
    )
    windows = time_windows(
        dataset.times,
        window_ms=_window_ms(preprocessing, key_ms="window_ms", key_seconds="window_size", default=20.0),
        step_ms=_window_ms(preprocessing, key_ms="step_ms", key_seconds="window_step", default=10.0),
    )
    if preprocessing.get("tmin") is not None or preprocessing.get("tmax") is not None:
        tmin = -np.inf if preprocessing.get("tmin") is None else float(preprocessing.get("tmin"))
        tmax = np.inf if preprocessing.get("tmax") is None else float(preprocessing.get("tmax"))
        windows = [window for window in windows if tmin <= window[2] <= tmax]
    if not windows:
        raise ValueError("No transfer time windows are available after preprocessing time selection.")

    train_indices = np.flatnonzero(train_mask)
    test_indices = np.flatnonzero(test_mask)
    rows: list[dict[str, Any]] = []
    observation_rows: list[dict[str, Any]] = []
    preprocessing_hash = stable_hash({"config": _selected_config(config)})
    for window in windows:
        features = _features_for_window(data, window)
        model = make_decoder(
            decoder_name,
            max_iter=max_iter,
            emission_mode=emission_mode,
            feature_preprocessor=feature_preprocessor,
            pca_components=pca_components,
        )
        model.fit(features[train_indices], labels[train_indices])
        probabilities = _align_probability_columns(
            predict_emission_probabilities(model, features[test_indices], emission_mode=emission_mode),
            model=model,
            classes=classes,
        )
        predictions = probabilities.argmax(axis=1)
        start, stop, center = window
        model_hash = stable_hash(
            {
                "decoder": decoder_name,
                "emission_mode": emission_mode,
                "feature_preprocessor": feature_preprocessor,
                "pca_components": pca_components,
                "window": center,
                "train_filter": train_filter,
                "test_filter": test_filter,
            }
        )
        rows.append(
            {
                "decoder": decoder_name,
                "emission_mode": emission_mode,
                "feature_preprocessor": feature_preprocessor,
                "pca_components": "" if pca_components is None else pca_components,
                "time": center,
                "window_start": float(dataset.times[start]),
                "window_stop": float(dataset.times[stop - 1]),
                "accuracy": accuracy_score(labels[test_indices], predictions),
                "log_loss": log_loss(labels[test_indices], probabilities, labels=classes),
                "brier": brier_score_multiclass(probabilities, labels[test_indices]),
                "ece": expected_calibration_error(probabilities, labels[test_indices]),
                "n_train": int(len(train_indices)),
                "n_test": int(len(test_indices)),
                "n_classes": int(len(classes)),
                "class_names": "|".join(map(str, encoder.classes_)),
                "train_filter": json.dumps(train_filter, sort_keys=True, default=str),
                "test_filter": json.dumps(test_filter, sort_keys=True, default=str),
                "preprocessing_hash": preprocessing_hash,
                "model_hash": model_hash,
            }
        )
        observation_rows.extend(
            _observation_rows(
                metadata=dataset.metadata,
                test_indices=test_indices,
                probabilities=probabilities,
                labels=labels,
                class_names=encoder.classes_,
                window=window,
                decoder_name=decoder_name,
                emission_mode=emission_mode,
                preprocessing_hash=preprocessing_hash,
                model_hash=model_hash,
                train_filter=train_filter,
                test_filter=test_filter,
                window_start=float(dataset.times[start]),
                window_stop=float(dataset.times[stop - 1]),
            )
        )

    summary_out, observations_out = _transfer_output_paths(config, config_dir=config_path.parent)
    results = pd.DataFrame(rows)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(summary_out, index=False)
    if observations_out is not None:
        observations_out.parent.mkdir(parents=True, exist_ok=True)
        ProbabilityObservationTable(pd.DataFrame(observation_rows)).standardized().to_csv(observations_out)
    _write_provenance_sidecars(
        config,
        config_path=config_path,
        config_dir=config_path.parent,
        output_paths=[path for path in (summary_out, observations_out) if path is not None],
    )
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run NeuRepTrace train/test transfer decoding from a JSON/YAML config.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a dotted config key, e.g. --set transfer.classifier=lda.")
    parser.add_argument("--print-effective-config", action="store_true", help="Print the effective config and exit without decoding.")
    parser.add_argument("--no-provenance", action="store_true", help="Do not write .provenance.json sidecars next to output CSVs.")
    args = parser.parse_args(argv)

    config = apply_overrides(load_config(args.config), args.overrides)
    if args.print_effective_config:
        print(json.dumps(effective_config(config, base_dir=args.config.parent), indent=2, sort_keys=True, default=str))
        return 0
    results = run_transfer_from_config(args.config, overrides=args.overrides, write_provenance=not args.no_provenance)
    print(f"Wrote {len(results)} transfer rows from {args.config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

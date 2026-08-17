"""Matched evaluation on Julia's Katja sliding-window cache.

The supplied cache defines a different endpoint from NeuRepTrace's response-
conditioned four-event benchmark. Here every 500 ms window receives one of six
finger labels (0=rest, 1..5=finger). Splits are made by complete target trials,
so overlapping windows from one trial can never straddle calibration and test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss

from neureptrace.katja_online_protocol import build_trial_split_manifest


JULIA_SUBJECTS = ("s05", "s06", "s08", "s09", "s10", "s11", "s15", "s16", "s17", "s18")
DEFAULT_K_VALUES = (1, 3, 5, 10, 15, 20)
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_METHODS = ("source_only", "adapter_only", "progressive_full")
ALL_CLASSES = np.arange(6, dtype=int)


@dataclass(frozen=True)
class TrialSplit:
    k: int
    calibration_trials: np.ndarray
    evaluation_trials: np.ndarray
    calibration_rows: np.ndarray
    evaluation_rows: np.ndarray
    reserved_trials: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    reserved_rows: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))


def _positive_int(value: int, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return parsed


def _utc_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _parse_csv(raw: str, *, cast=str) -> tuple:
    values = tuple(cast(value.strip()) for value in str(raw).split(",") if value.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value")
    return values


def _stable_seed(seed: int, *context: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(int(seed)).encode("utf-8"))
    for value in context:
        digest.update(b"\0")
        digest.update(str(value).encode("utf-8"))
    return int.from_bytes(digest.digest(), "little") % (2**32)


def relabel_minimum_overlap(
    raw_finger_ids: np.ndarray,
    press_overlap_fraction: np.ndarray,
    minimum_overlap: float,
) -> np.ndarray:
    """Apply Julia's training relabel without changing already-rest labels."""

    labels = np.asarray(raw_finger_ids, dtype=np.int64).reshape(-1).copy()
    overlap = np.asarray(press_overlap_fraction, dtype=float).reshape(-1)
    if labels.shape != overlap.shape:
        raise ValueError("finger_ids and press_overlap_fraction must have identical lengths")
    tau = float(minimum_overlap)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("minimum_overlap must be in [0, 1]")
    labels[overlap < tau] = 0
    return labels


def select_nested_trial_splits(
    sequence_ids: np.ndarray,
    trial_ids: np.ndarray,
    *,
    k_values: tuple[int, ...],
    seed: int,
    context: object = "target",
    split_mode: str = "nested_rest",
) -> dict[int, TrialSplit]:
    """Select nested k trials per sequence with changing or fixed evaluation rows."""

    sequence_ids = np.asarray(sequence_ids).reshape(-1)
    trial_ids = np.asarray(trial_ids).reshape(-1)
    if sequence_ids.shape != trial_ids.shape or sequence_ids.size == 0:
        raise ValueError("sequence_ids and trial_ids must be nonempty vectors of equal length")
    trial_table = pd.DataFrame(
        {
            "subject": str(context),
            "trial_id": trial_ids,
            "sequence_id": sequence_ids,
        }
    ).drop_duplicates()
    manifest, _ = build_trial_split_manifest(
        trial_table,
        calibration_counts=k_values,
        seeds=(int(seed),),
        mode=split_mode,
    )
    results: dict[int, TrialSplit] = {}
    for k, split_frame in manifest.groupby("k", sort=True):
        calibration_trials = split_frame.loc[split_frame["split_role"] == "calibration", "trial_id"].to_numpy()
        evaluation_trials = split_frame.loc[split_frame["split_role"] == "evaluation", "trial_id"].to_numpy()
        reserved_trials = split_frame.loc[split_frame["split_role"] == "reserved_unused", "trial_id"].to_numpy()
        calibration_rows = np.flatnonzero(np.isin(trial_ids, calibration_trials))
        evaluation_rows = np.flatnonzero(np.isin(trial_ids, evaluation_trials))
        reserved_rows = np.flatnonzero(np.isin(trial_ids, reserved_trials))
        if np.intersect1d(calibration_rows, evaluation_rows).size:
            raise AssertionError("Calibration and evaluation rows overlap")
        results[int(k)] = TrialSplit(
            k=int(k),
            calibration_trials=calibration_trials,
            evaluation_trials=evaluation_trials,
            calibration_rows=calibration_rows,
            evaluation_rows=evaluation_rows,
            reserved_trials=reserved_trials,
            reserved_rows=reserved_rows,
        )
    previous: set[Any] = set()
    for k in sorted(results):
        current = set(results[k].calibration_trials.tolist())
        if not previous.issubset(current):
            raise AssertionError("Calibration trials are not nested across k")
        previous = current
    if split_mode == "fixed_max_complement":
        reference = next(iter(results.values())).evaluation_rows
        if any(not np.array_equal(split.evaluation_rows, reference) for split in results.values()):
            raise AssertionError("fixed_max_complement did not produce identical evaluation rows")
    return results


def _load_npz_metadata(cache_path: Path) -> dict[str, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as loaded:
        required = {
            "meg_windows",
            "finger_ids",
            "press_overlap_fraction",
            "sequence_id",
            "trial_id",
        }
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"Window cache is missing keys: {missing}")
        subject_key = "subject_indices" if "subject_indices" in loaded.files else "subject_id"
        if subject_key not in loaded.files:
            raise ValueError("Window cache is missing subject_indices/subject_id")
        result = {
            "finger_ids": np.asarray(loaded["finger_ids"], dtype=np.int64),
            "press_overlap_fraction": np.asarray(loaded["press_overlap_fraction"], dtype=np.float32),
            "sequence_id": np.asarray(loaded["sequence_id"], dtype=np.int64),
            "trial_id": np.asarray(loaded["trial_id"], dtype=np.int64),
            "subject_indices": np.asarray(loaded[subject_key], dtype=np.int64),
        }
        for optional in ("press_order", "press_ratios", "sequence_id_global", "trial_id_global"):
            if optional in loaded.files:
                result[optional] = np.asarray(loaded[optional])
    with zipfile.ZipFile(cache_path) as archive:
        member_name = next((name for name in archive.namelist() if name.rstrip("/").endswith("meg_windows.npy")), None)
        if member_name is None:
            raise ValueError("Window cache ZIP has no meg_windows.npy member")
        with archive.open(member_name) as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _, _ = np.lib.format.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, _ = np.lib.format.read_array_header_2_0(stream)
            else:
                shape, _, _ = np.lib.format._read_array_header(stream, version)  # type: ignore[attr-defined]
    shape = tuple(int(value) for value in shape)
    lengths = {name: int(values.shape[0]) for name, values in result.items()}
    if len(set(lengths.values())) != 1 or next(iter(lengths.values())) != shape[0]:
        raise ValueError(f"Cache arrays do not share the MEG row count: MEG={shape[0]}, labels={lengths}")
    result["meg_shape"] = np.asarray(shape, dtype=np.int64)
    return result


def prepare_dct_feature_cache(
    cache_path: str | Path,
    output_path: str | Path,
    *,
    temporal_coefficients: int = 4,
    batch_size: int = 2048,
    force: bool = False,
) -> Path:
    """Create a row-aligned DCT feature matrix from the supplied windows."""

    cache_path = Path(cache_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    expected_source_size = cache_path.stat().st_size
    if output_path.exists() and metadata_path.exists() and not force:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if int(existing.get("source_size_bytes", -1)) == expected_source_size and int(existing.get("temporal_coefficients", -1)) == int(temporal_coefficients):
            return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = _load_npz_metadata(cache_path)
    n_rows, n_times, n_channels = metadata["meg_shape"].tolist()
    coefficients = min(_positive_int(temporal_coefficients, "temporal_coefficients"), int(n_times))
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    feature_matrix = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.float32,
        shape=(int(n_rows), int(coefficients * n_channels)),
    )
    with np.load(cache_path, allow_pickle=False) as loaded:
        windows = np.asarray(loaded["meg_windows"], dtype=np.float32)
        for start in range(0, int(n_rows), int(batch_size)):
            stop = min(int(n_rows), start + int(batch_size))
            transformed = dct(windows[start:stop], type=2, axis=1, norm="ortho")[:, :coefficients, :]
            feature_matrix[start:stop] = transformed.reshape(stop - start, -1).astype(np.float32, copy=False)
            if start == 0 or stop == int(n_rows) or (start // int(batch_size)) % 25 == 0:
                print(f"DCT feature cache: {stop}/{n_rows} windows", flush=True)
        del windows
    feature_matrix.flush()
    del feature_matrix
    os.replace(temporary, output_path)
    metadata_path.write_text(
        json.dumps(
            {
                "created_at": _utc_timestamp(),
                "feature_shape": [int(n_rows), int(coefficients * n_channels)],
                "feature_kind": "temporal_dct",
                "source_cache": str(cache_path),
                "source_size_bytes": expected_source_size,
                "temporal_coefficients": int(coefficients),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return output_path


def prepare_raw_window_memmap(cache_path: str | Path, output_path: str | Path) -> Path:
    """Extract the large ``meg_windows.npy`` member once for true memory mapping."""

    cache_path = Path(cache_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        values = np.load(output_path, mmap_mode="r")
        if values.ndim != 3:
            raise ValueError(f"Existing raw-window cache has invalid shape {values.shape}")
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    with zipfile.ZipFile(cache_path) as archive:
        member_name = next((name for name in archive.namelist() if name.rstrip("/").endswith("meg_windows.npy")), None)
        if member_name is None:
            raise ValueError("Window cache ZIP has no meg_windows.npy member")
        with archive.open(member_name) as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
            destination.flush()
            os.fsync(destination.fileno())
    values = np.load(temporary, mmap_mode="r")
    if values.ndim != 3:
        raise ValueError(f"Extracted raw-window cache has invalid shape {values.shape}")
    del values
    os.replace(temporary, output_path)
    return output_path


def prepare_subject_sensor_moments(
    window_store: np.ndarray,
    subject_indices: np.ndarray,
    output_path: str | Path,
    *,
    batch_size: int = 2048,
) -> dict[str, np.ndarray]:
    """Cache per-subject sensor moments for leakage-free fold normalization."""

    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        with np.load(output_path, allow_pickle=False) as loaded:
            return {name: np.asarray(loaded[name]) for name in loaded.files}
    subjects = np.unique(subject_indices)
    subject_to_row = {int(subject): position for position, subject in enumerate(subjects.tolist())}
    sums = np.zeros((subjects.size, window_store.shape[2]), dtype=np.float64)
    squared_sums = np.zeros_like(sums)
    counts = np.zeros(subjects.size, dtype=np.int64)
    for start in range(0, window_store.shape[0], int(batch_size)):
        stop = min(window_store.shape[0], start + int(batch_size))
        values = np.asarray(window_store[start:stop], dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Raw window cache contains nonfinite values in rows {start}:{stop}")
        batch_subjects = subject_indices[start:stop]
        for subject in np.unique(batch_subjects).tolist():
            selected = values[batch_subjects == subject]
            row = subject_to_row[int(subject)]
            sums[row] += selected.sum(axis=(0, 1), dtype=np.float64)
            squared_sums[row] += np.square(selected, dtype=np.float64).sum(axis=(0, 1), dtype=np.float64)
            counts[row] += int(selected.shape[0] * selected.shape[1])
        if start == 0 or stop == window_store.shape[0] or (start // int(batch_size)) % 25 == 0:
            print(f"Sensor moments: {stop}/{window_store.shape[0]} windows", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, subjects=subjects, sums=sums, squared_sums=squared_sums, counts=counts)
    return {"subjects": subjects, "sums": sums, "squared_sums": squared_sums, "counts": counts}


def source_sensor_normalization(moments: dict[str, np.ndarray], source_subjects: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = {int(subject): position for position, subject in enumerate(moments["subjects"].tolist())}
    rows = np.asarray([positions[int(subject)] for subject in source_subjects], dtype=int)
    total_count = int(np.sum(moments["counts"][rows]))
    total_sum = np.sum(moments["sums"][rows], axis=0)
    total_squared_sum = np.sum(moments["squared_sums"][rows], axis=0)
    mean = total_sum / total_count
    variance = np.maximum(total_squared_sum / total_count - np.square(mean), 1e-12)
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _balanced_source_sample(
    subject_indices: np.ndarray,
    source_subjects: np.ndarray,
    *,
    max_rows: int,
    seed: int,
) -> np.ndarray:
    per_subject = max(1, int(np.ceil(max_rows / source_subjects.size)))
    parts: list[np.ndarray] = []
    for subject in source_subjects.tolist():
        rows = np.flatnonzero(subject_indices == subject)
        rng = np.random.default_rng(_stable_seed(seed, "pca", subject))
        parts.append(rng.choice(rows, size=min(per_subject, rows.size), replace=False))
    result = np.concatenate(parts)
    if result.size > max_rows:
        rng = np.random.default_rng(_stable_seed(seed, "pca", "truncate"))
        result = rng.choice(result, size=max_rows, replace=False)
    return np.sort(result.astype(int, copy=False))


def prepare_fold_features(
    raw_features: np.ndarray,
    subject_indices: np.ndarray,
    *,
    source_subjects: np.ndarray,
    target_subject: int,
    pca_components: int,
    pca_fit_max_windows: int,
    seed: int,
    cache_path: Path | None = None,
) -> np.ndarray:
    """Fit source-only standardization/PCA and transform all rows for one LOSO fold."""

    if cache_path is not None and cache_path.exists():
        return np.load(cache_path, mmap_mode="r")
    source_subjects = np.asarray(source_subjects, dtype=int).reshape(-1)
    if source_subjects.size == 0 or int(target_subject) in source_subjects:
        raise ValueError("source_subjects must be nonempty and exclude target_subject")
    source_rows = np.flatnonzero(np.isin(subject_indices, source_subjects))
    source_values = np.asarray(raw_features[source_rows], dtype=np.float32)
    mean = source_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = source_values.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    fit_rows = _balanced_source_sample(
        subject_indices,
        source_subjects,
        max_rows=min(int(pca_fit_max_windows), source_rows.size),
        seed=int(seed),
    )
    fit_values = (np.asarray(raw_features[fit_rows], dtype=np.float32) - mean) / std
    n_components = min(int(pca_components), fit_values.shape[0] - 1, fit_values.shape[1])
    if n_components <= 0:
        raise ValueError("Not enough source rows to fit PCA")
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=int(seed))
    pca.fit(fit_values)
    del fit_values, source_values

    if cache_path is None:
        transformed = np.empty((raw_features.shape[0], n_components), dtype=np.float32)
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".part")
        transformed = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(raw_features.shape[0], n_components),
        )
    for start in range(0, raw_features.shape[0], 4096):
        stop = min(raw_features.shape[0], start + 4096)
        normalized = (np.asarray(raw_features[start:stop], dtype=np.float32) - mean) / std
        transformed[start:stop] = pca.transform(normalized).astype(np.float32, copy=False)
    if cache_path is not None:
        transformed.flush()
        del transformed
        os.replace(temporary, cache_path)
        cache_path.with_suffix(cache_path.suffix + ".json").write_text(
            json.dumps(
                {
                    "created_at": _utc_timestamp(),
                    "n_components": int(n_components),
                    "pca_fit_max_windows": int(pca_fit_max_windows),
                    "pca_fit_rows": int(fit_rows.size),
                    "preprocessing_fit_subjects": [int(value) for value in source_subjects.tolist()],
                    "target_subject_excluded": int(target_subject),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return np.load(cache_path, mmap_mode="r")
    return transformed


def _trial_macro_accuracy(y_true: np.ndarray, y_pred: np.ndarray, trial_ids: np.ndarray) -> float:
    values = [float(np.mean(y_pred[trial_ids == trial] == y_true[trial_ids == trial])) for trial in np.unique(trial_ids)]
    return float(np.mean(values))


def _metric_row(
    *,
    method: str,
    target: str,
    target_index: int,
    seed: int,
    split: TrialSplit,
    probabilities: np.ndarray,
    raw_labels: np.ndarray,
    training_labels: np.ndarray,
    target_trial_ids: np.ndarray,
    n_source_windows: int,
    n_source_subjects: int,
    adaptation_stages: str,
    predicted_labels: np.ndarray | None = None,
) -> dict[str, Any]:
    evaluation_rows = split.evaluation_rows
    y_true = raw_labels[evaluation_rows]
    y_train_definition = training_labels[evaluation_rows]
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (evaluation_rows.size, ALL_CLASSES.size):
        raise ValueError("probabilities must contain one six-class row per evaluation window")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Every probability row must have positive mass")
    probabilities = probabilities / row_sums
    if predicted_labels is None:
        predicted = ALL_CLASSES[np.argmax(probabilities, axis=1)]
    else:
        predicted = np.asarray(predicted_labels, dtype=np.int64).reshape(-1)
        if predicted.shape != y_true.shape or np.any((predicted < 0) | (predicted > 5)):
            raise ValueError("predicted_labels must contain one class 0..5 per evaluation row")
    press_mask = y_true > 0
    binary_true = press_mask.astype(int)
    binary_pred = (predicted > 0).astype(int)
    confusion = confusion_matrix(y_true, predicted, labels=ALL_CLASSES)
    n_target_trials = int(np.unique(target_trial_ids).size)
    return {
        "method": method,
        "target": target,
        "target_index": int(target_index),
        "seed": int(seed),
        "k_trials_per_sequence": int(split.k),
        "n_source_subjects": int(n_source_subjects),
        "n_source_windows": int(n_source_windows),
        "n_calibration_trials": int(split.calibration_trials.size),
        "n_calibration_windows": int(split.calibration_rows.size),
        "n_evaluation_trials": int(split.evaluation_trials.size),
        "n_evaluation_windows": int(evaluation_rows.size),
        "calibration_share_target_trials": float(split.calibration_trials.size / n_target_trials),
        "accuracy_raw_labels": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy_raw_labels": float(balanced_accuracy_score(y_true, predicted)),
        "accuracy_tau_labels": float(accuracy_score(y_train_definition, predicted)),
        "press_only_finger_accuracy": float(accuracy_score(y_true[press_mask], predicted[press_mask])) if np.any(press_mask) else float("nan"),
        "rest_recall": float(np.mean(predicted[y_true == 0] == 0)) if np.any(y_true == 0) else float("nan"),
        "press_detection_accuracy": float(accuracy_score(binary_true, binary_pred)),
        "trial_macro_accuracy_raw_labels": _trial_macro_accuracy(
            y_true,
            predicted,
            target_trial_ids[evaluation_rows],
        ),
        "log_loss_raw_labels": float(log_loss(y_true, probabilities, labels=ALL_CLASSES)),
        "majority_class_accuracy": float(np.max(np.bincount(y_true, minlength=ALL_CLASSES.size)) / y_true.size),
        "confusion_matrix": json.dumps(confusion.tolist(), separators=(",", ":")),
        "adaptation_stages": adaptation_stages,
        "calibration_evaluation_disjoint": bool(np.intersect1d(split.calibration_rows, split.evaluation_rows).size == 0),
    }


def _append_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if path.exists():
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        missing = sorted(set(columns) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(columns))
        if missing or extra:
            raise ValueError(
                f"Cannot append a row with a different CSV schema to {path}: "
                f"missing={missing}, extra={extra}"
            )
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def _write_status(path: Path, **payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps({**payload, "updated_at": _utc_timestamp()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summarize_results(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_columns = [
        "accuracy_raw_labels",
        "balanced_accuracy_raw_labels",
        "accuracy_tau_labels",
        "press_only_finger_accuracy",
        "rest_recall",
        "press_detection_accuracy",
        "trial_macro_accuracy_raw_labels",
        "log_loss_raw_labels",
        "majority_class_accuracy",
    ]
    metric_columns = [column for column in metric_columns if column in rows.columns]
    subject = rows.groupby(["method", "k_trials_per_sequence", "target"], as_index=False)[metric_columns].mean()
    subject_rows: list[dict[str, Any]] = []
    for (method, k), frame in subject.groupby(["method", "k_trials_per_sequence"], sort=True):
        row: dict[str, Any] = {"method": method, "k_trials_per_sequence": int(k), "n_subjects": int(frame.shape[0])}
        for metric in metric_columns:
            values = frame[metric].to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(np.nanmean(values))
            row[f"sem_{metric}"] = float(np.nanstd(values, ddof=1) / np.sqrt(np.sum(np.isfinite(values)))) if np.sum(np.isfinite(values)) > 1 else float("nan")
        subject_rows.append(row)
    subject_summary = pd.DataFrame(subject_rows)

    fold_rows: list[dict[str, Any]] = []
    for (method, k), frame in rows.groupby(["method", "k_trials_per_sequence"], sort=True):
        row = {"method": method, "k_trials_per_sequence": int(k), "n_subject_seed_folds": int(frame.shape[0])}
        for metric in metric_columns:
            values = frame[metric].to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(np.nanmean(values))
            row[f"sd_{metric}"] = float(np.nanstd(values, ddof=1)) if np.sum(np.isfinite(values)) > 1 else float("nan")
        fold_rows.append(row)
    julia_style = pd.DataFrame(fold_rows)
    return subject, subject_summary, julia_style


def paired_common_cohort_statistics(subject_rows: pd.DataFrame) -> pd.DataFrame:
    """Compute predeclared paired contrasts after averaging seeds per subject."""

    from scipy.stats import t as student_t
    from scipy.stats import ttest_1samp, wilcoxon

    required = {"method", "k_trials_per_sequence", "target", "accuracy_raw_labels"}
    missing = sorted(required - set(subject_rows.columns))
    if missing:
        raise ValueError(f"subject_rows is missing columns: {missing}")
    k_values = sorted(int(value) for value in subject_rows["k_trials_per_sequence"].unique())
    if not k_values:
        raise ValueError("subject_rows contains no k values")
    minimum_k, maximum_k = k_values[0], k_values[-1]
    contrasts = (
        ("progressive dose response", "progressive_full", maximum_k, "progressive_full", minimum_k),
        ("adapter versus source at maximum k", "adapter_only", maximum_k, "source_only", maximum_k),
        ("progressive versus source at maximum k", "progressive_full", maximum_k, "source_only", maximum_k),
        ("progressive versus adapter at maximum k", "progressive_full", maximum_k, "adapter_only", maximum_k),
    )
    indexed = subject_rows.set_index(["method", "k_trials_per_sequence", "target"])["accuracy_raw_labels"]
    output_columns = [
        "contrast",
        "method_a",
        "k_a",
        "method_b",
        "k_b",
        "n_subjects",
        "mean_delta_accuracy",
        "sem_delta_accuracy",
        "ci95_low_accuracy",
        "ci95_high_accuracy",
        "paired_t_p",
        "wilcoxon_p",
    ]
    available = set(indexed.index.droplevel("target").unique().tolist())
    rows: list[dict[str, Any]] = []
    for label, method_a, k_a, method_b, k_b in contrasts:
        if (method_a, k_a) not in available or (method_b, k_b) not in available:
            continue
        first = indexed.xs((method_a, k_a), level=("method", "k_trials_per_sequence"))
        second = indexed.xs((method_b, k_b), level=("method", "k_trials_per_sequence"))
        common_targets = first.index.intersection(second.index)
        differences = (first.loc[common_targets] - second.loc[common_targets]).to_numpy(dtype=float)
        n_subjects = int(differences.size)
        if n_subjects < 2:
            raise ValueError(f"Contrast {label!r} needs at least two paired subjects")
        mean_delta = float(np.mean(differences))
        sem_delta = float(np.std(differences, ddof=1) / np.sqrt(n_subjects))
        critical = float(student_t.ppf(0.975, df=n_subjects - 1))
        try:
            wilcoxon_p = float(wilcoxon(differences).pvalue)
        except ValueError:
            wilcoxon_p = float("nan")
        rows.append(
            {
                "contrast": label,
                "method_a": method_a,
                "k_a": int(k_a),
                "method_b": method_b,
                "k_b": int(k_b),
                "n_subjects": n_subjects,
                "mean_delta_accuracy": mean_delta,
                "sem_delta_accuracy": sem_delta,
                "ci95_low_accuracy": mean_delta - critical * sem_delta,
                "ci95_high_accuracy": mean_delta + critical * sem_delta,
                "paired_t_p": float(ttest_1samp(differences, 0.0).pvalue),
                "wilcoxon_p": wilcoxon_p,
            }
        )
    return pd.DataFrame(rows, columns=output_columns)


def _write_comparison_scope(
    output_path: Path,
    *,
    source_model_per_seed: bool,
    common_targets: tuple[str, ...],
) -> None:
    payload = {
        "claim_level": "task_data_and_split_convention_matched_not_model_identical",
        "exact_or_directly_verified": {
            "supplied_window_cache": True,
            "window_length_ms": 500,
            "sampling_rate_hz": 100,
            "stride_ms": 40,
            "classes": "0=rest, 1-5=fingers",
            "training_minimum_overlap_tau": 0.2,
            "evaluation_labels": "raw cache finger_ids",
            "target_participants": list(JULIA_SUBJECTS),
            "calibration_unit": "complete trial",
            "calibration_count_definition": "k trials per each of four local sequences",
            "calibration_evaluation_trial_disjoint": True,
            "seeds": list(DEFAULT_SEEDS),
            "source_model_retrained_per_seed": bool(source_model_per_seed),
            "common_curve_participants": list(common_targets),
            "primary_uncertainty": "average seeds within subject, then SEM across subjects",
            "collaborator_aggregation_also_reported": "mean and SD across subject-by-seed folds",
        },
        "approximated_because_reference_was_not_supplied": {
            "split_rng_implementation": "deterministic nested per-sequence permutations",
            "model_architecture": "NeuRepTrace temporal multitask model with pre-encoder low-rank adapter",
            "optimization_and_hyperparameters": "NeuRepTrace configuration",
        },
        "unavailable_for_formal_method_comparison": {
            "collaborator_fold_predictions": True,
            "collaborator_per_k_summary": True,
            "collaborator_exact_model_code": True,
            "collaborator_exact_split_function": True,
        },
        "interpretation": (
            "Use this result as an independent model on the same endpoint and split convention. "
            "Do not present the difference from Julia's reported range as a controlled "
            "architecture benchmark or significance test."
        ),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _plot_summary(
    summary: pd.DataFrame,
    output_path: Path,
    *,
    cohort_note: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "source_only": "Source only",
        "adapter_only": "Low-rank adapter + head",
        "progressive_full": "Progressive full fine-tune",
    }
    colors = {"source_only": "#1f1f1f", "adapter_only": "#2a7f62", "progressive_full": "#c24e3a"}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhspan(
        62.5,
        64.5,
        color="#4f6d9b",
        alpha=0.14,
        label="Julia reported range (not per-k)",
    )
    for method, frame in summary.groupby("method", sort=False):
        frame = frame.sort_values("k_trials_per_sequence")
        ax.errorbar(
            frame["k_trials_per_sequence"],
            100.0 * frame["mean_accuracy_raw_labels"],
            yerr=100.0 * frame["sem_accuracy_raw_labels"],
            marker="o" if method != "source_only" else "D",
            linestyle="-" if method != "source_only" else "--",
            color=colors.get(method),
            capsize=3,
            label=labels.get(method, method),
        )
    majority = summary.groupby("k_trials_per_sequence", as_index=False)["mean_majority_class_accuracy"].mean().sort_values("k_trials_per_sequence")
    ax.plot(
        majority["k_trials_per_sequence"],
        100.0 * majority["mean_majority_class_accuracy"],
        color="#777777",
        linestyle=":",
        linewidth=1.3,
        label="Evaluation-set majority baseline",
    )
    ax.axhline(100.0 / 6.0, color="#777777", linestyle=":", linewidth=1.2, label="Uniform chance (16.7%)")
    ax.set_xscale("log")
    ax.set_xticks(sorted(summary["k_trials_per_sequence"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("Labeled target trials per sequence (k)")
    ax.set_ylabel("Finger-window accuracy (%)")
    title = "Katja sliding-window task, matched split convention"
    if cohort_note:
        title = f"{title}\n{cohort_note}"
    ax.set_title(title)
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _write_comparison_report(
    summary: pd.DataFrame,
    julia_style: pd.DataFrame,
    output_path: Path,
    *,
    common_summary: pd.DataFrame,
    common_targets: tuple[str, ...],
    paired_statistics: pd.DataFrame,
    source_model_per_seed: bool,
) -> None:
    collaborator_low = 0.625
    collaborator_high = 0.645
    table = summary.merge(
        julia_style,
        on=["method", "k_trials_per_sequence"],
        how="left",
        suffixes=("", "_fold"),
    )
    labels = {
        "source_only": "Source only",
        "adapter_only": "Low-rank adapter + heads",
        "progressive_full": "Progressive full fine-tune",
    }
    lines = [
        "# Katja online-window comparison",
        "",
        "## Comparison boundary",
        "",
        "This run matches the supplied 500 ms, 100 Hz sliding-window cache, the six-class "
        "finger/rest endpoint, complete-trial target calibration, `k` trials per sequence, "
        "training relabel threshold `tau=0.2`, raw-cache scoring labels, ten named target "
        "participants, and seeds 0-4.",
        "",
        f"Source pretraining was {'repeated independently for every seed' if source_model_per_seed else 'fixed across seeds'}. "
        "Target calibration and evaluation use disjoint complete trials.",
        "",
        "It does **not** reproduce Julia's unpublished architecture: that code and the exact "
        "reference split function were not supplied. NeuRepTrace uses a pre-encoder low-rank "
        "sensor adapter, temporal convolutional backbone, and progressive fine-tuning. The "
        "comparison is therefore task- and data-matched, but not model-identical.",
        "",
        f"Julia reported approximately {100.0 * collaborator_low:.1f}-"
        f"{100.0 * collaborator_high:.1f}% finger accuracy depending on random seed. "
        "That range is contextual rather than a per-k uncertainty interval, so it is not used "
        "as a formal significance test.",
        "",
        "## Results",
        "",
        "| Method | k per sequence | Subjects | Folds | Mean accuracy | SEM across subjects | Mean over folds | SD over folds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table.sort_values(["method", "k_trials_per_sequence"]).itertuples(index=False):
        sem = getattr(row, "sem_accuracy_raw_labels")
        fold_mean = getattr(row, "mean_accuracy_raw_labels_fold")
        fold_sd = getattr(row, "sd_accuracy_raw_labels")
        lines.append(
            f"| {labels.get(row.method, row.method)} | {int(row.k_trials_per_sequence)} | "
            f"{int(row.n_subjects)} | {int(row.n_subject_seed_folds)} | "
            f"{100.0 * row.mean_accuracy_raw_labels:.2f}% | "
            f"{100.0 * sem:.2f}% | {100.0 * fold_mean:.2f}% | {100.0 * fold_sd:.2f}% |"
        )
    best = table.loc[table["mean_accuracy_raw_labels"].idxmax()]
    delta_low = 100.0 * (float(best["mean_accuracy_raw_labels"]) - collaborator_low)
    relation = "above" if delta_low >= 0.0 else "below"
    preferred_method = "progressive_full" if "progressive_full" in set(common_summary["method"]) else str(common_summary.iloc[0]["method"])
    common_curve = common_summary[common_summary["method"] == preferred_method].sort_values("k_trials_per_sequence")
    common_first = common_curve.iloc[0]
    common_last = common_curve.iloc[-1]
    maximum_subject_count = int(table["n_subjects"].max())
    full_cohort_rows = table[(table["method"] == preferred_method) & (table["n_subjects"] == maximum_subject_count)]
    full_cohort_last = full_cohort_rows.sort_values("k_trials_per_sequence").iloc[-1]
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"The strongest NeuRepTrace row is `{best['method']}` at "
            f"`k={int(best['k_trials_per_sequence'])}`, with "
            f"{100.0 * float(best['mean_accuracy_raw_labels']):.2f}% mean accuracy. "
            f"This is {abs(delta_low):.2f} percentage points {relation} the lower edge of "
            "Julia's reported range.",
            "",
            f"The largest-k point retaining all {maximum_subject_count} target participants is "
            f"`{preferred_method}` at k={int(full_cohort_last['k_trials_per_sequence'])}: "
            f"{100.0 * float(full_cohort_last['mean_accuracy_raw_labels']):.2f}% with "
            f"{100.0 * float(full_cohort_last['sem_accuracy_raw_labels']):.2f}% SEM across subjects "
            f"and {100.0 * float(full_cohort_last['sd_accuracy_raw_labels']):.2f}% SD over "
            f"{int(full_cohort_last['n_subject_seed_folds'])} subject-by-seed folds.",
            "",
            f"On the common cohort, `{preferred_method}` rises from "
            f"{100.0 * float(common_first['mean_accuracy_raw_labels']):.2f}% at "
            f"k={int(common_first['k_trials_per_sequence'])} to "
            f"{100.0 * float(common_last['mean_accuracy_raw_labels']):.2f}% at "
            f"k={int(common_last['k_trials_per_sequence'])}, a gain of "
            f"{100.0 * (float(common_last['mean_accuracy_raw_labels']) - float(common_first['mean_accuracy_raw_labels'])):.2f} percentage points.",
            "",
        ]
    )
    paired_lookup = paired_statistics.set_index("contrast")
    required_contrasts = {
        "progressive dose response",
        "progressive versus source at maximum k",
        "progressive versus adapter at maximum k",
    }
    if required_contrasts.issubset(paired_lookup.index):
        dose = paired_lookup.loc["progressive dose response"]
        progressive_source = paired_lookup.loc["progressive versus source at maximum k"]
        progressive_adapter = paired_lookup.loc["progressive versus adapter at maximum k"]
        lines.extend(
            [
                f"The paired common-cohort dose-response estimate is {100.0 * float(dose['mean_delta_accuracy']):.2f} "
                f"percentage points (95% CI {100.0 * float(dose['ci95_low_accuracy']):.2f} to "
                f"{100.0 * float(dose['ci95_high_accuracy']):.2f}; n={int(dose['n_subjects'])}). "
                f"At maximum k, progressive adaptation exceeds source-only by "
                f"{100.0 * float(progressive_source['mean_delta_accuracy']):.2f} points "
                f"(95% CI {100.0 * float(progressive_source['ci95_low_accuracy']):.2f} to "
                f"{100.0 * float(progressive_source['ci95_high_accuracy']):.2f}).",
                "",
                f"Progressive full fine-tuning exceeds the adapter-only model by only "
                f"{100.0 * float(progressive_adapter['mean_delta_accuracy']):.2f} point at maximum k "
                f"(95% CI {100.0 * float(progressive_adapter['ci95_low_accuracy']):.2f} to "
                f"{100.0 * float(progressive_adapter['ci95_high_accuracy']):.2f}). This is modest "
                "relative to the gain from labeled target calibration itself.",
                "",
            ]
        )
    lines.extend(
        [
            "Rows with fewer than ten subjects reflect strict k-per-sequence feasibility; "
            "the runner records those omissions in `skipped_folds.csv` rather than silently "
            "capping k or sampling with replacement.",
            "",
            f"The figure uses the same {len(common_targets)} participants at every k "
            f"({', '.join(common_targets)}). The table above remains available-case so the "
            "k<=10 rows also retain s06. Common-cohort values are in "
            "`summary_common_subject_sem.csv`.",
            "",
            "For the primary uncertainty estimate, seeds are averaged within each target "
            "participant before computing SEM across participants. The final two columns also "
            "reproduce Julia's stated aggregation convention: mean and SD over subject-by-seed "
            "folds (50 where every participant supports k; otherwise the exact fold count is shown).",
            "",
            "The match boundary and unavailable reference components are recorded in `comparison_scope.json`; paired estimates are in `paired_common_cohort_statistics.csv`.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _common_k_cohort(rows: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, ...]]:
    required_keys = set(rows[["method", "k_trials_per_sequence", "seed"]].drop_duplicates().itertuples(index=False, name=None))
    targets = tuple(
        sorted(
            target
            for target, frame in rows.groupby("target")
            if set(frame[["method", "k_trials_per_sequence", "seed"]].drop_duplicates().itertuples(index=False, name=None)) == required_keys
        )
    )
    if not targets:
        raise RuntimeError("No target participant has complete coverage across k, methods, and seeds")
    return rows[rows["target"].isin(targets)].copy(), targets


def _write_result_validation(rows: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    key_columns = ["target", "seed", "k_trials_per_sequence", "method"]
    checks: dict[str, bool] = {
        "unique_fold_keys": not rows.duplicated(key_columns).any(),
        "known_methods_only": set(rows["method"]).issubset(DEFAULT_METHODS),
        "known_targets_only": set(rows["target"]).issubset(JULIA_SUBJECTS),
    }
    if "calibration_evaluation_disjoint" in rows:
        disjoint = rows["calibration_evaluation_disjoint"]
        if disjoint.dtype != bool:
            disjoint = disjoint.astype(str).str.lower().eq("true")
        checks["calibration_evaluation_disjoint"] = bool(disjoint.all())
    full_target_set = set(rows["target"]) == set(JULIA_SUBJECTS)
    if full_target_set and {
        "n_calibration_trials",
        "k_trials_per_sequence",
    }.issubset(rows.columns):
        checks["four_sequences_times_k_calibration_trials"] = bool((rows["n_calibration_trials"].astype(int) == 4 * rows["k_trials_per_sequence"].astype(int)).all())
    if "n_source_subjects" in rows:
        checks["nine_source_subjects"] = bool((rows["n_source_subjects"].astype(int) == 9).all())
    bounded_metrics = [
        "accuracy_raw_labels",
        "balanced_accuracy_raw_labels",
        "accuracy_tau_labels",
        "press_only_finger_accuracy",
        "rest_recall",
        "press_detection_accuracy",
        "trial_macro_accuracy_raw_labels",
        "majority_class_accuracy",
    ]
    for metric in bounded_metrics:
        if metric in rows:
            values = rows[metric].to_numpy(dtype=float)
            checks[f"{metric}_in_unit_interval"] = bool(np.all(np.isfinite(values)) and np.all((0.0 <= values) & (values <= 1.0)))
    if full_target_set and set(rows["method"]) == set(DEFAULT_METHODS):
        actual = rows.groupby(["method", "k_trials_per_sequence"]).size().to_dict()
        expected = {(method, k): 50 if k <= 10 else 45 for method in DEFAULT_METHODS for k in DEFAULT_K_VALUES}
        checks["expected_full_fold_counts"] = actual == expected
        checks["expected_full_row_count_870"] = int(rows.shape[0]) == 870
    payload = {
        "all_required_checks_pass": bool(all(checks.values())),
        "checks": checks,
        "n_rows": int(rows.shape[0]),
        "n_targets": int(rows["target"].nunique()),
        "n_methods": int(rows["method"].nunique()),
        "created_at": _utc_timestamp(),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def aggregate_benchmark_shards(
    shard_dirs: tuple[str | Path, ...],
    output_dir: str | Path,
) -> Path:
    """Merge disjoint target shards and rebuild all aggregate artifacts."""

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    skipped_frames: list[pd.DataFrame] = []
    shard_provenance: list[dict[str, Any]] = []
    resolved_shards: list[Path] = []
    for raw_dir in shard_dirs:
        shard = Path(raw_dir).expanduser().resolve()
        resolved_shards.append(shard)
        result_path = shard / "fold_results.csv"
        if not result_path.exists():
            result_path = shard / "fold_results.partial.csv"
        if not result_path.exists():
            raise FileNotFoundError(f"Shard has no fold results: {shard}")
        frames.append(pd.read_csv(result_path))
        skipped_path = shard / "skipped_folds.csv"
        if skipped_path.exists():
            skipped_frames.append(pd.read_csv(skipped_path))
        provenance_path = shard / "provenance.json"
        if provenance_path.exists():
            shard_provenance.append(json.loads(provenance_path.read_text(encoding="utf-8")))

    rows = pd.concat(frames, ignore_index=True)
    key_columns = ["target", "seed", "k_trials_per_sequence", "method"]
    duplicates = rows.duplicated(key_columns, keep=False)
    if duplicates.any():
        duplicate_keys = rows.loc[duplicates, key_columns].drop_duplicates().to_dict("records")
        raise ValueError(f"Shard outputs overlap on fold keys: {duplicate_keys[:10]}")
    rows = rows.sort_values(["method", "k_trials_per_sequence", "target", "seed"]).reset_index(drop=True)
    rows.to_csv(output_path / "fold_results.csv", index=False)
    rows.to_csv(output_path / "fold_results.partial.csv", index=False)
    subject, summary, julia_style = summarize_results(rows)
    common_rows, common_targets = _common_k_cohort(rows)
    common_subject, common_summary, common_julia_style = summarize_results(common_rows)
    paired_statistics = paired_common_cohort_statistics(common_subject)
    source_model_per_seed = bool(shard_provenance) and all(provenance.get("source_model_per_seed") is True for provenance in shard_provenance)
    subject.to_csv(output_path / "subject_seed_averages.csv", index=False)
    summary.to_csv(output_path / "summary_subject_sem.csv", index=False)
    julia_style.to_csv(output_path / "summary_julia_fold_sd.csv", index=False)
    julia_style.to_csv(output_path / "summary_julia_50fold_sd.csv", index=False)
    common_subject.to_csv(output_path / "subject_seed_averages_common.csv", index=False)
    common_summary.to_csv(output_path / "summary_common_subject_sem.csv", index=False)
    common_julia_style.to_csv(output_path / "summary_common_fold_sd.csv", index=False)
    paired_statistics.to_csv(output_path / "paired_common_cohort_statistics.csv", index=False)
    _write_comparison_scope(
        output_path / "comparison_scope.json",
        source_model_per_seed=source_model_per_seed,
        common_targets=common_targets,
    )
    _plot_summary(
        common_summary,
        output_path / "katja_julia_window_comparison.png",
        cohort_note=f"common {len(common_targets)}-participant cohort",
    )
    _write_comparison_report(
        summary,
        julia_style,
        output_path / "comparison_to_julia.md",
        common_summary=common_summary,
        common_targets=common_targets,
        paired_statistics=paired_statistics,
        source_model_per_seed=source_model_per_seed,
    )
    validation = _write_result_validation(rows, output_path / "validation.json")
    if not validation["all_required_checks_pass"]:
        raise RuntimeError(f"Aggregated result validation failed: {validation['checks']}")
    if skipped_frames:
        skipped = pd.concat(skipped_frames, ignore_index=True).drop_duplicates()
        skipped.to_csv(output_path / "skipped_folds.csv", index=False)
    provenance = {
        "aggregation_only": True,
        "source_shards": [str(path) for path in resolved_shards],
        "shard_provenance": shard_provenance,
        "aggregation_primary": "seeds averaged within subject, SEM across subjects",
        "aggregation_collaborator": "mean and SD over subject-by-seed folds",
        "source_model_per_seed": source_model_per_seed,
        "created_at": _utc_timestamp(),
    }
    (output_path / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_status(
        output_path / "status.json",
        state="complete",
        n_rows=int(rows.shape[0]),
        n_targets=int(rows["target"].nunique()),
        n_subject_seed_folds=int(rows[["target", "seed"]].drop_duplicates().shape[0]),
    )
    print(summary.to_string(index=False), flush=True)
    return output_path


def run_benchmark(args: argparse.Namespace) -> Path:
    from neureptrace.decoding.progressive_temporal_window_finetune import (
        TorchProgressiveTemporalWindowClassifier,
    )
    from neureptrace.decoding.progressive_window_finetune import (
        TorchProgressiveWindowClassifier,
    )

    cache_path = Path(args.cache).expanduser().resolve()
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "status.json"
    partial_path = output_dir / "fold_results.partial.csv"
    metadata = _load_npz_metadata(cache_path)
    raw_labels = metadata["finger_ids"]
    training_labels = relabel_minimum_overlap(raw_labels, metadata["press_overlap_fraction"], args.minimum_overlap)
    subject_indices = metadata["subject_indices"]
    unique_subjects = np.unique(subject_indices)
    if unique_subjects.size != len(JULIA_SUBJECTS) or not np.array_equal(unique_subjects, np.arange(len(JULIA_SUBJECTS))):
        raise ValueError(f"Expected subject indices 0..9, found {unique_subjects.tolist()}")
    subjects = _parse_csv(args.subjects)
    targets = subjects if args.targets is None else _parse_csv(args.targets)
    unknown = sorted(set(subjects) - set(JULIA_SUBJECTS) | (set(targets) - set(subjects)))
    if unknown:
        raise ValueError(f"Unknown or out-of-pool subjects: {unknown}")
    k_values = _parse_csv(args.k_values, cast=int)
    seeds = _parse_csv(args.seeds, cast=int)
    methods = _parse_csv(args.methods)
    if not set(methods).issubset(DEFAULT_METHODS):
        raise ValueError(f"Methods must be selected from {DEFAULT_METHODS}")

    raw_features = None
    window_store = None
    sensor_moments = None
    order_training_labels = None
    if args.feature_mode == "dct":
        feature_path = Path(args.feature_cache) if args.feature_cache else output_dir / f"dct{args.temporal_coefficients}_features.npy"
        prepare_dct_feature_cache(
            cache_path,
            feature_path,
            temporal_coefficients=args.temporal_coefficients,
            batch_size=args.feature_batch_size,
        )
        raw_features = np.load(feature_path, mmap_mode="r")
    else:
        raw_window_path = Path(args.raw_window_cache) if args.raw_window_cache else cache_path.parent / "derived" / "meg_windows_raw.npy"
        prepare_raw_window_memmap(cache_path, raw_window_path)
        window_store = np.load(raw_window_path, mmap_mode="r")
        moments_path = raw_window_path.with_suffix(raw_window_path.suffix + ".subject_moments.npz")
        sensor_moments = prepare_subject_sensor_moments(
            window_store,
            subject_indices,
            moments_path,
            batch_size=args.feature_batch_size,
        )
        if "press_order" not in metadata:
            raise ValueError("The temporal multi-task model requires press_order labels")
        order_training_labels = relabel_minimum_overlap(
            metadata["press_order"],
            metadata["press_overlap_fraction"],
            args.minimum_overlap,
        )
    completed: set[tuple[str, int, int, str]] = set()
    if args.resume and partial_path.exists():
        existing = pd.read_csv(partial_path)
        completed = set(
            zip(
                existing["target"].astype(str),
                existing["seed"].astype(int),
                existing["k_trials_per_sequence"].astype(int),
                existing["method"].astype(str),
                strict=True,
            )
        )
    elif partial_path.exists():
        partial_path.unlink()

    model_kwargs = {
        "hidden_units": args.hidden_units,
        "adapter_rank": args.adapter_rank,
        "source_epochs": args.source_epochs,
        "adapter_steps": args.adapter_steps,
        "last_block_steps": args.last_block_steps,
        "full_finetune_steps": args.full_finetune_steps,
        "batch_size": args.batch_size,
        "device": args.device,
    }
    if args.feature_mode == "dct":
        model_kwargs["num_layers"] = args.num_layers
    else:
        model_kwargs.update(
            {
                "num_blocks": args.num_blocks,
                "sequence_loss_weight": args.sequence_loss_weight,
                "order_loss_weight": args.order_loss_weight,
                "overlap_loss_weight": args.overlap_loss_weight,
            }
        )
    feasibility: dict[str, dict[str, Any]] = {}
    skipped_path = output_dir / "skipped_folds.csv"
    if not args.resume and skipped_path.exists():
        skipped_path.unlink()
    skipped_keys: set[tuple[str, int, int, str]] = set()
    if args.resume and skipped_path.exists():
        existing_skips = pd.read_csv(skipped_path)
        skipped_keys = set(
            zip(
                existing_skips["target"].astype(str),
                existing_skips["seed"].astype(int),
                existing_skips["k_trials_per_sequence"].astype(int),
                existing_skips["method"].astype(str),
                strict=True,
            )
        )
    fold_counter = 0
    for target in targets:
        target_index = JULIA_SUBJECTS.index(target)
        target_rows_global = np.flatnonzero(subject_indices == target_index)
        source_subject_indices = np.asarray([JULIA_SUBJECTS.index(subject) for subject in subjects if subject != target], dtype=int)
        source_rows_global = np.flatnonzero(np.isin(subject_indices, source_subject_indices))
        _write_status(status_path, state="preprocessing", target=target, target_index=target_index)
        fold_features = None
        source_x = None
        target_x = None
        sensor_mean = None
        sensor_std = None
        if args.feature_mode == "dct":
            assert raw_features is not None
            fold_feature_path = output_dir / "preprocessed" / f"target_{target}_pca{args.pca_components}.npy"
            fold_features = prepare_fold_features(
                raw_features,
                subject_indices,
                source_subjects=source_subject_indices,
                target_subject=target_index,
                pca_components=args.pca_components,
                pca_fit_max_windows=args.pca_fit_max_windows,
                seed=args.preprocessing_seed,
                cache_path=fold_feature_path,
            )
            source_x = np.asarray(fold_features[source_rows_global], dtype=np.float32)
            target_x = np.asarray(fold_features[target_rows_global], dtype=np.float32)
        else:
            assert sensor_moments is not None
            sensor_mean, sensor_std = source_sensor_normalization(sensor_moments, source_subject_indices)
        source_y = training_labels[source_rows_global]
        target_raw_y = raw_labels[target_rows_global]
        target_training_y = training_labels[target_rows_global]
        target_sequence = metadata["sequence_id"][target_rows_global]
        target_trials = metadata["trial_id"][target_rows_global]
        trial_registry = pd.DataFrame({"trial": target_trials, "sequence": target_sequence}).drop_duplicates()
        per_sequence_trials = trial_registry.groupby("sequence")["trial"].nunique()
        feasible_k = tuple(int(k) for k in k_values if np.all(per_sequence_trials.to_numpy() > int(k)))
        infeasible_k = tuple(int(k) for k in k_values if int(k) not in feasible_k)
        feasibility[target] = {
            "trials_per_sequence": {str(key): int(value) for key, value in per_sequence_trials.to_dict().items()},
            "feasible_k": list(feasible_k),
            "infeasible_k": list(infeasible_k),
        }
        for k in infeasible_k:
            for seed in seeds:
                for method in methods:
                    skip_row = {
                        "target": target,
                        "seed": int(seed),
                        "k_trials_per_sequence": int(k),
                        "method": method,
                        "skip_reason": "at least one sequence has <=k trials, so strict k-per-sequence calibration would leave no sequence-matched evaluation trial",
                        "trials_per_sequence": json.dumps(feasibility[target]["trials_per_sequence"], separators=(",", ":")),
                    }
                    skip_key = (target, int(seed), int(k), method)
                    if skip_key not in skipped_keys:
                        _append_row(skipped_path, skip_row)
                        skipped_keys.add(skip_key)
        if not feasible_k:
            continue

        cached_source_model = None
        cached_source_fit_seconds = None
        for seed in seeds:
            if all((target, int(seed), int(k), method) in completed for k in feasible_k for method in methods):
                continue
            fold_counter += 1
            if args.max_folds is not None and fold_counter > args.max_folds:
                break
            _write_status(status_path, state="source_fit", target=target, seed=int(seed), fold=fold_counter)
            if args.feature_mode == "temporal" and cached_source_model is not None:
                source_model = cached_source_model
                source_fit_seconds = float(cached_source_fit_seconds)
            else:
                started = time.monotonic()
                if args.feature_mode == "dct":
                    assert source_x is not None
                    source_model = TorchProgressiveWindowClassifier(
                        **model_kwargs,
                        random_state=int(seed) if args.source_model_per_seed else int(args.source_model_seed),
                    )
                    source_model.fit_source(source_x, source_y)
                else:
                    assert window_store is not None and sensor_mean is not None and sensor_std is not None
                    assert order_training_labels is not None
                    source_model = TorchProgressiveTemporalWindowClassifier(
                        **model_kwargs,
                        random_state=int(seed) if args.source_model_per_seed else int(args.source_model_seed),
                    )
                    source_model.fit_source(
                        window_store,
                        source_indices=source_rows_global,
                        source_domains=subject_indices,
                        finger_labels=training_labels,
                        sequence_labels=metadata["sequence_id"],
                        order_labels=order_training_labels,
                        overlap_targets=metadata["press_overlap_fraction"],
                        sensor_mean=sensor_mean,
                        sensor_std=sensor_std,
                    )
                source_fit_seconds = float(time.monotonic() - started)
                if args.feature_mode == "temporal" and not args.source_model_per_seed:
                    cached_source_model = source_model
                    cached_source_fit_seconds = source_fit_seconds
            splits = select_nested_trial_splits(
                target_sequence,
                target_trials,
                k_values=feasible_k,
                seed=int(seed),
                context=target,
            )
            for k in feasible_k:
                split = splits[int(k)]
                for method in methods:
                    key = (target, int(seed), int(k), method)
                    if key in completed:
                        continue
                    _write_status(status_path, state="method_fit", target=target, seed=int(seed), k=int(k), method=method)
                    if method == "source_only":
                        model = source_model
                        stages = "none"
                    else:
                        if args.feature_mode == "dct":
                            assert target_x is not None
                            model = source_model.clone_source()
                            model.adapt_target(
                                target_x[split.calibration_rows],
                                target_training_y[split.calibration_rows],
                                n_calibration_trials=int(split.calibration_trials.size),
                                mode=method,
                            )
                        else:
                            model = source_model.clone_source(random_state=int(seed))
                            model.adapt_target_indices(
                                target_rows_global[split.calibration_rows],
                                n_calibration_trials=int(split.calibration_trials.size),
                                mode=method,
                            )
                        stages = ",".join(item["stage"] for item in model.adaptation_history_)
                    if args.feature_mode == "dct":
                        assert target_x is not None
                        probabilities = model.predict_proba(target_x[split.evaluation_rows])
                    else:
                        probabilities = model.predict_proba_indices(target_rows_global[split.evaluation_rows])
                    row = _metric_row(
                        method=method,
                        target=target,
                        target_index=target_index,
                        seed=int(seed),
                        split=split,
                        probabilities=probabilities,
                        raw_labels=target_raw_y,
                        training_labels=target_training_y,
                        target_trial_ids=target_trials,
                        n_source_windows=source_rows_global.size,
                        n_source_subjects=source_subject_indices.size,
                        adaptation_stages=stages,
                    )
                    row["minimum_overlap_training_tau"] = float(args.minimum_overlap)
                    row["evaluation_label_definition"] = "raw_cache_finger_ids"
                    row["feature_kind"] = (
                        f"temporal_dct{args.temporal_coefficients}_source_pca{args.pca_components}" if args.feature_mode == "dct" else "raw_500ms_temporal_multitask"
                    )
                    row["source_fit_seconds"] = source_fit_seconds
                    if args.feature_mode == "temporal":
                        source_metadata = source_model.metadata()
                        row["best_source_epoch"] = source_metadata["best_source_epoch"]
                        row["source_validation_loss"] = source_metadata["best_source_validation_loss"]
                        row["source_validation_mode"] = source_metadata["source_validation_mode"]
                        row["source_validation_domain"] = source_metadata["source_validation_domain"]
                    _append_row(partial_path, row)
                    completed.add(key)
                    print(
                        f"{target} seed={seed} k={k} {method}: accuracy={row['accuracy_raw_labels']:.4f}",
                        flush=True,
                    )
            if args.feature_mode == "dct" or args.source_model_per_seed:
                del source_model
        if args.max_folds is not None and fold_counter >= args.max_folds:
            break
        del source_x, target_x, fold_features, cached_source_model

    if not partial_path.exists():
        raise RuntimeError("No benchmark rows were produced")
    rows = pd.read_csv(partial_path).drop_duplicates(["target", "seed", "k_trials_per_sequence", "method"], keep="last")
    rows = rows.sort_values(["method", "k_trials_per_sequence", "target", "seed"]).reset_index(drop=True)
    rows.to_csv(output_dir / "fold_results.csv", index=False)
    subject, summary, julia_style = summarize_results(rows)
    common_rows, common_targets = _common_k_cohort(rows)
    common_subject, common_summary, common_julia_style = summarize_results(common_rows)
    paired_statistics = paired_common_cohort_statistics(common_subject)
    subject.to_csv(output_dir / "subject_seed_averages.csv", index=False)
    summary.to_csv(output_dir / "summary_subject_sem.csv", index=False)
    julia_style.to_csv(output_dir / "summary_julia_fold_sd.csv", index=False)
    julia_style.to_csv(output_dir / "summary_julia_50fold_sd.csv", index=False)
    common_subject.to_csv(output_dir / "subject_seed_averages_common.csv", index=False)
    common_summary.to_csv(output_dir / "summary_common_subject_sem.csv", index=False)
    common_julia_style.to_csv(output_dir / "summary_common_fold_sd.csv", index=False)
    paired_statistics.to_csv(output_dir / "paired_common_cohort_statistics.csv", index=False)
    _write_comparison_scope(
        output_dir / "comparison_scope.json",
        source_model_per_seed=bool(args.source_model_per_seed),
        common_targets=common_targets,
    )
    _plot_summary(
        common_summary,
        output_dir / "katja_julia_window_comparison.png",
        cohort_note=f"common {len(common_targets)}-participant cohort",
    )
    _write_comparison_report(
        summary,
        julia_style,
        output_dir / "comparison_to_julia.md",
        common_summary=common_summary,
        common_targets=common_targets,
        paired_statistics=paired_statistics,
        source_model_per_seed=bool(args.source_model_per_seed),
    )
    validation = _write_result_validation(rows, output_dir / "validation.json")
    if not validation["all_required_checks_pass"]:
        raise RuntimeError(f"Result validation failed: {validation['checks']}")
    provenance = {
        "cache": str(cache_path),
        "cache_size_bytes": cache_path.stat().st_size,
        "subjects": list(subjects),
        "targets": list(targets),
        "subject_index_mapping": {str(index): subject for index, subject in enumerate(JULIA_SUBJECTS)},
        "k_values": list(k_values),
        "seeds": list(seeds),
        "methods": list(methods),
        "minimum_overlap_training_tau": float(args.minimum_overlap),
        "evaluation_labels": "raw finger_ids, following supplied sequence-average diagnostic",
        "window_definition": "500 ms at 100 Hz; stride 40 ms; go cue to 6 s",
        "split_unit": "complete target trial, stratified by within-subject sequence_id",
        "k_definition": "complete target trials per sequence",
        "calibration_subsets_nested": True,
        "split_implementation": "neureptrace.katja_online_protocol.build_trial_split_manifest(mode=nested_rest)",
        "evaluation_definition": "complement of each k calibration set, matching collaborator description",
        "pca_fit_uses_target": False,
        "feature_mode": args.feature_mode,
        "source_model_per_seed": bool(args.source_model_per_seed),
        "source_model_seed": int(args.source_model_seed),
        "preprocessing_seed": int(args.preprocessing_seed),
        "model_kwargs": model_kwargs,
        "aggregation_primary": "seeds averaged within subject, SEM across subjects",
        "aggregation_collaborator": "mean and SD over subject-by-seed folds",
        "collaborator_reported_accuracy_range": [0.625, 0.645],
        "collaborator_architecture_available": False,
        "k_feasibility_by_target": feasibility,
        "created_at": _utc_timestamp(),
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_status(
        status_path,
        state="complete",
        n_rows=int(rows.shape[0]),
        n_targets=int(rows["target"].nunique()),
        n_subject_seed_folds=int(rows[["target", "seed"]].drop_duplicates().shape[0]),
    )
    print(summary.to_string(index=False), flush=True)
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", help="Julia's meg_windows_0to6_100hz_stride4.npz")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--aggregate-shards",
        help="Comma-separated completed shard directories; aggregate them without fitting.",
    )
    parser.add_argument("--feature-mode", choices=("temporal", "dct"), default="temporal")
    parser.add_argument("--feature-cache", help="Reusable row-aligned DCT .npy file")
    parser.add_argument("--raw-window-cache", help="Extracted meg_windows.npy used by temporal mode")
    parser.add_argument("--subjects", default=",".join(JULIA_SUBJECTS))
    parser.add_argument("--targets")
    parser.add_argument("--k-values", default=",".join(str(value) for value in DEFAULT_K_VALUES))
    parser.add_argument("--seeds", default=",".join(str(value) for value in DEFAULT_SEEDS))
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--minimum-overlap", type=float, default=0.2)
    parser.add_argument("--temporal-coefficients", type=int, default=4)
    parser.add_argument("--feature-batch-size", type=int, default=2048)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument("--pca-fit-max-windows", type=int, default=50000)
    parser.add_argument("--preprocessing-seed", type=int, default=13)
    parser.add_argument("--hidden-units", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--adapter-rank", type=int, default=8)
    parser.add_argument("--source-epochs", type=int, default=12)
    parser.add_argument("--adapter-steps", type=int, default=80)
    parser.add_argument("--last-block-steps", type=int, default=60)
    parser.add_argument("--full-finetune-steps", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--sequence-loss-weight", type=float, default=0.15)
    parser.add_argument("--order-loss-weight", type=float, default=0.30)
    parser.add_argument("--overlap-loss-weight", type=float, default=0.30)
    parser.add_argument("--source-model-seed", type=int, default=13)
    parser.add_argument(
        "--source-model-per-seed",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retrain source pretraining for each split seed (much slower).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-folds", type=int)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.aggregate_shards:
        aggregate_benchmark_shards(_parse_csv(args.aggregate_shards), args.out_dir)
        return 0
    if not args.cache:
        raise SystemExit("--cache is required unless --aggregate-shards is used")
    run_benchmark(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

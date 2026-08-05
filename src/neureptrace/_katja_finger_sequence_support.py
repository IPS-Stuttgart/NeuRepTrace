"""Run the Katja four-variable-finger calibrated sequence benchmark from a feature cache.

The cache is deliberately feature-level: raw MEG files and event reconstruction
remain dataset-specific, while all source/target splitting, fold-local scaling,
PCA, neural fitting, and reporting are performed here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEFAULT_PARTICIPANTS = (
    "s05",
    "s06",
    "s08",
    "s09",
    "s10",
    "s11",
    "s13",
    "s14",
    "s15",
    "s16",
    "s17",
    "s18",
    "s20",
    "s21",
    "s24",
    "s25",
    "s28",
)
DEFAULT_CALIBRATION_COUNTS = (1, 3, 5, 10, 15, 20)
DEFAULT_CALIBRATION_SEEDS = (13, 29, 47, 71, 101)
DEFAULT_SOURCE_SELECTION_SEED = 2026
JULIA_FULL_FINETUNE_ACCURACY = {1: 0.274, 3: 0.404, 5: 0.466, 10: 0.534, 15: 0.572, 20: 0.594}


def _parse_csv_values(value: str, *, cast=str) -> tuple:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("Comma-separated argument must contain at least one value.")
    return tuple(cast(item) for item in items)


def _as_1d(values: np.ndarray, *, name: str, n_rows: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1 or array.shape[0] != n_rows:
        raise ValueError(f"Cache field {name!r} must be one-dimensional with {n_rows} rows.")
    return array


def _as_boolean_1d(values: np.ndarray, *, name: str, n_rows: int) -> np.ndarray:
    """Validate a cache Boolean vector without applying truthiness coercion."""

    array = _as_1d(values, name=name, n_rows=n_rows)
    result = np.empty(n_rows, dtype=bool)
    for index, value in enumerate(array.tolist()):
        if isinstance(value, (bool, np.bool_)):
            result[index] = bool(value)
            continue
        if isinstance(value, (int, float, np.integer, np.floating)):
            numeric = float(value)
            if np.isfinite(numeric) and numeric in (0.0, 1.0):
                result[index] = bool(int(numeric))
                continue
        raise ValueError(
            f"Cache field {name!r} must contain only boolean or numeric 0/1 values; "
            f"row {index} contains {value!r}."
        )
    return result


def _cache_field(cache: Any, *names: str, required: bool = True):
    for name in names:
        if name in cache:
            return cache[name], name
    if required:
        raise ValueError(f"Feature cache is missing required field; tried {', '.join(names)}.")
    return None, None


def load_katja_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Load and validate the event-row cache used by the benchmark runner."""

    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(cache_path)
    with np.load(cache_path, allow_pickle=True) as cache:
        features, feature_name = _cache_field(cache, "features", "X")
        feature_array = np.asarray(features, dtype=np.float32)
        if feature_array.ndim < 2 or feature_array.shape[0] < 1:
            raise ValueError(f"Cache field {feature_name!r} must contain event rows and at least one feature dimension.")
        feature_array = feature_array.reshape(feature_array.shape[0], -1)
        if not np.all(np.isfinite(feature_array)):
            raise ValueError("Feature cache contains non-finite values.")
        n_rows = feature_array.shape[0]
        subjects, _ = _cache_field(cache, "subjects", "participant_ids", "subject_ids")
        trial_ids, _ = _cache_field(cache, "trial_ids", "trials")
        positions, _ = _cache_field(cache, "press_positions", "event_positions", "press_index")
        sequence_ids, _ = _cache_field(cache, "sequence_ids", "seq_ids", "seqID")
        labels, _ = _cache_field(cache, "labels", "finger_labels", required=False)
        finger_codes, _ = _cache_field(cache, "finger_codes", "physical_finger_codes", "button_codes", required=False)
        correct_order, _ = _cache_field(cache, "correct_order", "is_correct_order", required=False)

        if labels is None and finger_codes is None:
            raise ValueError("Feature cache needs either participant-local labels or physical finger_codes.")
        result = {
            "features": feature_array,
            "subjects": _as_1d(subjects, name="subjects", n_rows=n_rows).astype(str),
            "trial_ids": _as_1d(trial_ids, name="trial_ids", n_rows=n_rows).astype(object),
            "press_positions": _as_1d(positions, name="press_positions", n_rows=n_rows),
            "sequence_ids": _as_1d(sequence_ids, name="sequence_ids", n_rows=n_rows).astype(object),
        }
        if labels is not None:
            result["labels"] = _as_1d(labels, name="labels", n_rows=n_rows)
        if finger_codes is not None:
            result["finger_codes"] = _as_1d(finger_codes, name="finger_codes", n_rows=n_rows)
        result["correct_order"] = (
            np.ones(n_rows, dtype=bool)
            if correct_order is None
            else _as_boolean_1d(correct_order, name="correct_order", n_rows=n_rows)
        )
    return result


def derive_participant_local_finger_labels(
    subjects: np.ndarray,
    finger_codes: np.ndarray,
    *,
    included_mask: np.ndarray,
    expected_classes: int = 4,
) -> np.ndarray:
    """Map each participant's sorted variable physical codes to classes 0..K-1."""

    labels = np.full(subjects.shape[0], -1, dtype=int)
    for subject in dict.fromkeys(subjects.tolist()):
        mask = (subjects == subject) & included_mask
        codes = np.unique(finger_codes[mask])
        try:
            codes = np.sort(codes)
        except TypeError:
            codes = np.asarray(sorted(codes.tolist(), key=str), dtype=object)
        if codes.shape[0] != expected_classes:
            raise ValueError(f"Participant {subject!r} has {codes.shape[0]} variable finger codes; expected {expected_classes}.")
        for class_index, code in enumerate(codes.tolist()):
            labels[(subjects == subject) & (finger_codes == code)] = class_index
    if np.any(labels[included_mask] < 0):
        raise ValueError("Could not derive all participant-local finger labels.")
    return labels


def _stable_source_selection(
    participants: tuple[str, ...],
    *,
    target: str,
    n_sources: int,
    seed: int,
) -> tuple[str, ...]:
    """Reproduce the source subset used by the original finger comparison."""

    candidates = np.asarray(
        sorted(participant for participant in participants if participant != target),
        dtype=object,
    )
    if candidates.shape[0] < n_sources:
        raise ValueError(f"Target {target!r} has only {candidates.shape[0]} candidate sources; requested {n_sources}.")
    target_number = int("".join(character for character in target if character.isdigit()) or 0)
    rng = np.random.default_rng(int(seed) + target_number)
    selected = rng.choice(candidates, size=int(n_sources), replace=False)
    return tuple(sorted(str(value) for value in selected.tolist()))


def katja_nested_trial_calibration_indices(
    strata: np.ndarray,
    calibration_counts: tuple[int, ...],
    *,
    seed: int,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Reproduce the original sequential-RNG nested Katja trial split.

    The maximum calibration pool is drawn first inside each sorted sequence ID.
    Lower-k pools are prefixes of that maximum pool, and the complement of the
    maximum pool is the common evaluation set for every k.
    """

    values = np.asarray(strata).reshape(-1)
    requested = tuple(sorted({int(count) for count in calibration_counts if int(count) > 0}))
    if not requested:
        raise ValueError("calibration_counts must contain at least one positive value.")
    maximum = max(requested)
    rng = np.random.default_rng(int(seed))
    calibration: dict[int, list[int]] = {count: [] for count in requested}
    evaluation: list[int] = []
    reserved_all: list[int] = []
    try:
        ordered_strata = sorted(set(values.tolist()))
    except TypeError:
        ordered_strata = sorted(set(values.tolist()), key=str)
    for stratum in ordered_strata:
        candidates = np.flatnonzero(values == stratum).astype(int, copy=True)
        if candidates.size <= maximum:
            raise ValueError(
                f"Stratum {stratum!r} has {candidates.size} trials; more than {maximum} are required."
            )
        rng.shuffle(candidates)
        reserved = candidates[:maximum]
        reserved_all.extend(int(index) for index in reserved)
        evaluation.extend(int(index) for index in candidates[maximum:])
        for count in requested:
            calibration[count].extend(int(index) for index in reserved[:count])
    return (
        {count: np.asarray(sorted(indices), dtype=int) for count, indices in calibration.items()},
        np.asarray(sorted(evaluation), dtype=int),
        np.asarray(sorted(reserved_all), dtype=int),
    )


def _load_source_map(path: str | Path | None) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source-map JSON must be an object mapping target participant to source participant list.")
    result: dict[str, tuple[str, ...]] = {}
    for target, sources in payload.items():
        if not isinstance(sources, list) or not sources:
            raise ValueError(f"source-map entry for {target!r} must be a non-empty list.")
        result[str(target)] = tuple(str(source) for source in sources)
    return result


def _composite_trial_ids(subjects: np.ndarray, trial_ids: np.ndarray) -> np.ndarray:
    return np.asarray([f"{subject}\x1f{repr(trial_id)}" for subject, trial_id in zip(subjects.tolist(), trial_ids.tolist(), strict=True)], dtype=object)


def _constant_trial_values(values: np.ndarray, row_indices: np.ndarray, *, name: str) -> np.ndarray:
    result = []
    for trial_rows in row_indices:
        trial_values = values[trial_rows]
        if not np.all(trial_values == trial_values[0]):
            raise ValueError(f"{name} must be constant within each complete trial.")
        result.append(trial_values[0])
    return np.asarray(result, dtype=object)


def _fit_source_preprocessor(source_features: np.ndarray, *, pca_components: int | None):
    scaler = StandardScaler()
    source_scaled = scaler.fit_transform(source_features)
    pca = None
    if pca_components is not None:
        requested = int(pca_components)
        if requested < 1:
            raise ValueError("pca_components must be positive or omitted.")
        effective = min(requested, source_scaled.shape[0], source_scaled.shape[1])
        pca = PCA(n_components=effective, random_state=0)
        source_scaled = pca.fit_transform(source_scaled)
    return scaler, pca, np.asarray(source_scaled, dtype=np.float32)


def _transform_features(features: np.ndarray, *, scaler: StandardScaler, pca: PCA | None) -> np.ndarray:
    transformed = scaler.transform(features)
    if pca is not None:
        transformed = pca.transform(transformed)
    return np.asarray(transformed, dtype=np.float32)


def _mean_sem(values: np.ndarray) -> tuple[float, float]:
    vector = np.asarray(values, dtype=float)
    if vector.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(vector))
    sem = float(np.std(vector, ddof=1) / np.sqrt(vector.size)) if vector.size > 1 else 0.0
    return mean, sem

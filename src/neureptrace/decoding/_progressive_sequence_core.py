"""Progressive sequence-level neural fine-tuning for calibrated target subjects.

The decoder is designed for repeated-event trials such as button-press sequences.
It preserves the trial structure instead of treating each event as an unrelated
row, learns a shared source backbone, and then adapts to a labeled target subject
in progressively less constrained stages:

1. a low-rank target adapter and classifier head;
2. the adapter, head, and final sequence block;
3. the full backbone with L2-SP regularization and optional source replay.

For tasks where every complete trial contains one occurrence of every class, a
Sinkhorn loss and Hungarian decoding enforce the known one-to-one assignment.
The implementation never needs target evaluation labels during fitting.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import hashlib

import numpy as np
from scipy.optimize import linear_sum_assignment

PROGRESSIVE_SEQUENCE_PROTOCOL = "progressive_sequence_target_fine_tuning"
PROGRESSIVE_SEQUENCE_CATEGORY = "3_supervised_calibrated_target_alignment"


@dataclass(frozen=True, slots=True)
class PackedTrialEvents:
    """Complete event rows packed into a trial-by-event tensor."""

    features: np.ndarray
    labels: np.ndarray | None
    trial_ids: np.ndarray
    press_positions: np.ndarray
    row_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class NestedTrialCalibrationSplit:
    """Trial-level nested calibration split with a fixed evaluation set."""

    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray
    calibration_pool_indices: np.ndarray
    per_stratum: int
    max_per_stratum: int
    seed: int


@dataclass(frozen=True, slots=True)
class PermutationDecodingResult:
    """Soft doubly-stochastic probabilities and hard one-to-one assignments."""

    probabilities: np.ndarray
    assignments: np.ndarray
    one_hot_probabilities: np.ndarray


@dataclass(frozen=True, slots=True)
class ProgressiveSequenceCalibrationResult:
    """Fitted decoder and predictions for disjoint target evaluation trials."""

    model: "TorchProgressiveSequenceClassifier"
    probabilities: np.ndarray
    constrained_probabilities: np.ndarray
    predictions: np.ndarray
    independent_predictions: np.ndarray
    evaluation_indices: np.ndarray
    calibration_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "The progressive sequence decoder requires torch; install the "
            "NeuRepTrace torch extra, e.g. `pip install neureptrace[torch]`."
        ) from exc
    return torch


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(number) or number % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    return int(number)


def _positive_int(value: Any, name: str) -> int:
    number = _integer(value, name)
    if number < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return number


def _nonnegative_int(value: Any, name: str) -> int:
    number = _integer(value, name)
    if number < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return number


def _positive_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite value.")
    return number


def _nonnegative_float(value: Any, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative finite value.")
    return number


def _bounded_float(value: Any, name: str, *, lower: float, upper: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number < lower or number > upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return number


def _as_object_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return array.reshape(-1)


def _as_feature_tensor(values: Sequence | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (trials, events, features).")
    if min(array.shape) < 1:
        raise ValueError(f"{name} must have non-empty trial, event, and feature dimensions.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _as_label_matrix(values: Sequence | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape (trials, events).")
    if min(array.shape) < 1:
        raise ValueError(f"{name} must be non-empty.")
    return array


def _stable_seed(seed: int, context: Sequence[Hashable], *parts: Any) -> int:
    payload = repr((int(seed), tuple(str(item) for item in context), *(str(part) for part in parts))).encode("utf-8")
    return int(hashlib.blake2b(payload, digest_size=8).hexdigest(), 16) % (2**32)


def _unique_in_order(values: Iterable[Any]) -> list[Any]:
    unique: list[Any] = []
    for value in values:
        if not any(_labels_equal(value, existing) for existing in unique):
            unique.append(value)
    return unique


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
    except Exception:
        return False
    if isinstance(result, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(result)


def pack_complete_trial_events(
    features: Sequence[Sequence[float]] | np.ndarray,
    trial_ids: Sequence[Any] | np.ndarray,
    press_positions: Sequence[Any] | np.ndarray,
    *,
    labels: Sequence[Any] | np.ndarray | None = None,
    expected_events: int = 4,
    require_unique_positions: bool = True,
    require_permutation_labels: bool = False,
) -> PackedTrialEvents:
    """Pack event rows into complete, press-position-sorted trials.

    Incomplete trials are rejected rather than silently mixed into a sequence
    model. ``row_indices`` maps every packed event back to its original row.
    """

    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("features must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite.")
    trials = _as_object_vector(trial_ids, name="trial_ids")
    positions = _as_object_vector(press_positions, name="press_positions")
    if trials.shape[0] != matrix.shape[0] or positions.shape[0] != matrix.shape[0]:
        raise ValueError("features, trial_ids, and press_positions must contain the same rows.")
    label_vector = None if labels is None else _as_object_vector(labels, name="labels")
    if label_vector is not None and label_vector.shape[0] != matrix.shape[0]:
        raise ValueError("labels must contain one value per feature row.")

    n_events = _positive_int(expected_events, "expected_events")
    ordered_trial_ids = _unique_in_order(trials.tolist())
    packed_rows: list[np.ndarray] = []
    packed_positions: list[np.ndarray] = []
    for trial_id in ordered_trial_ids:
        rows = np.asarray([index for index, candidate in enumerate(trials.tolist()) if _labels_equal(candidate, trial_id)], dtype=int)
        if rows.size != n_events:
            raise ValueError(f"Trial {trial_id!r} has {rows.size} rows; expected exactly {n_events}.")
        trial_positions = positions[rows]
        try:
            order = np.argsort(trial_positions, kind="stable")
        except TypeError:
            order = np.argsort(np.asarray([str(value) for value in trial_positions], dtype=object), kind="stable")
        sorted_rows = rows[order]
        sorted_positions = trial_positions[order]
        if require_unique_positions and len(_unique_in_order(sorted_positions.tolist())) != n_events:
            raise ValueError(f"Trial {trial_id!r} contains duplicate press positions.")
        packed_rows.append(sorted_rows)
        packed_positions.append(sorted_positions)

    row_indices = np.stack(packed_rows, axis=0)
    packed_features = matrix[row_indices]
    packed_labels = None if label_vector is None else label_vector[row_indices]
    if require_permutation_labels:
        if packed_labels is None:
            raise ValueError("labels are required when require_permutation_labels=True.")
        classes = _unique_in_order(packed_labels.reshape(-1).tolist())
        if len(classes) != n_events:
            raise ValueError(f"Permutation trials require {n_events} global classes; got {len(classes)}.")
        for trial_index, trial_labels in enumerate(packed_labels):
            if len(_unique_in_order(trial_labels.tolist())) != n_events or any(
                not any(_labels_equal(label, class_label) for label in trial_labels.tolist()) for class_label in classes
            ):
                raise ValueError(f"Trial {ordered_trial_ids[trial_index]!r} is not a one-of-each-class permutation.")

    return PackedTrialEvents(
        features=packed_features,
        labels=packed_labels,
        trial_ids=np.asarray(ordered_trial_ids, dtype=object),
        press_positions=np.stack(packed_positions, axis=0),
        row_indices=row_indices,
    )


def select_nested_trial_calibration_splits(
    strata: Sequence[Any] | np.ndarray,
    calibration_counts: Sequence[int] = (1, 3, 5, 10, 15, 20),
    *,
    max_per_stratum: int | None = None,
    min_evaluation_per_stratum: int = 1,
    seed: int = 13,
    context: Sequence[Hashable] = (),
) -> dict[int, NestedTrialCalibrationSplit]:
    """Reserve a maximum calibration pool first and create nested lower-k sets.

    The evaluation set is the complement of the maximum pool and is therefore
    identical for every requested calibration count.
    """

    stratum_vector = _as_object_vector(strata, name="strata")
    if stratum_vector.size < 1:
        raise ValueError("strata must contain at least one trial.")
    requested = sorted({_positive_int(value, "calibration_count") for value in calibration_counts})
    if not requested:
        raise ValueError("calibration_counts must not be empty.")
    maximum = max(requested) if max_per_stratum is None else _positive_int(max_per_stratum, "max_per_stratum")
    if maximum < max(requested):
        raise ValueError("max_per_stratum must be at least the largest requested calibration count.")
    min_eval = _nonnegative_int(min_evaluation_per_stratum, "min_evaluation_per_stratum")
    seed_value = _nonnegative_int(seed, "seed")

    pools: dict[str, np.ndarray] = {}
    ordered_strata = _unique_in_order(stratum_vector.tolist())
    for stratum_position, stratum in enumerate(ordered_strata):
        positions = np.asarray(
            [index for index, candidate in enumerate(stratum_vector.tolist()) if _labels_equal(candidate, stratum)],
            dtype=int,
        )
        required = maximum + min_eval
        if positions.size < required:
            raise ValueError(f"Stratum {stratum!r} needs at least {required} trials; got {positions.size}.")
        rng = np.random.default_rng(_stable_seed(seed_value, context, "stratum", stratum_position, stratum))
        pools[str(stratum_position)] = rng.permutation(positions)[:maximum]

    pool_indices = np.sort(np.concatenate(list(pools.values())).astype(int, copy=False))
    evaluation_mask = np.ones(stratum_vector.shape[0], dtype=bool)
    evaluation_mask[pool_indices] = False
    evaluation_indices = np.flatnonzero(evaluation_mask)
    splits: dict[int, NestedTrialCalibrationSplit] = {}
    for count in requested:
        calibration_indices = np.sort(np.concatenate([pool[:count] for pool in pools.values()]).astype(int, copy=False))
        splits[count] = NestedTrialCalibrationSplit(
            calibration_indices=calibration_indices,
            evaluation_indices=evaluation_indices.copy(),
            calibration_pool_indices=pool_indices.copy(),
            per_stratum=count,
            max_per_stratum=maximum,
            seed=seed_value,
        )
    return splits


def _normalize_probability_tensor(probabilities: Sequence | np.ndarray) -> np.ndarray:
    array = np.asarray(probabilities, dtype=float)
    if array.ndim != 3:
        raise ValueError("probabilities must have shape (trials, events, classes).")
    if min(array.shape) < 1 or not np.all(np.isfinite(array)):
        raise ValueError("probabilities must be non-empty and finite.")
    array = np.maximum(array, 0.0)
    row_sums = array.sum(axis=2, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Each event probability row must have positive mass.")
    return array / row_sums


def sinkhorn_trial_probabilities(
    probabilities: Sequence | np.ndarray,
    *,
    temperature: float = 1.0,
    iterations: int = 30,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Project square trial probability matrices toward doubly stochastic form."""

    array = _normalize_probability_tensor(probabilities)
    if array.shape[1] != array.shape[2]:
        raise ValueError("Sinkhorn trial projection requires events == classes.")
    temp = _positive_float(temperature, "temperature")
    n_iterations = _positive_int(iterations, "iterations")
    eps = _positive_float(epsilon, "epsilon")
    log_scores = np.log(np.clip(array, eps, None)) / temp
    log_scores -= np.max(log_scores, axis=(1, 2), keepdims=True)
    scores = np.exp(log_scores)
    for _ in range(n_iterations):
        scores /= np.maximum(scores.sum(axis=2, keepdims=True), eps)
        scores /= np.maximum(scores.sum(axis=1, keepdims=True), eps)
    scores /= np.maximum(scores.sum(axis=2, keepdims=True), eps)
    return scores


def permutation_constrained_decode(
    probabilities: Sequence | np.ndarray,
    *,
    temperature: float = 1.0,
    sinkhorn_iterations: int = 30,
    epsilon: float = 1e-12,
) -> PermutationDecodingResult:
    """Decode one occurrence of every class in each complete trial."""

    soft = sinkhorn_trial_probabilities(
        probabilities,
        temperature=temperature,
        iterations=sinkhorn_iterations,
        epsilon=epsilon,
    )
    n_trials, n_events, n_classes = soft.shape
    assignments = np.empty((n_trials, n_events), dtype=int)
    one_hot = np.zeros_like(soft)
    for trial_index in range(n_trials):
        # A tiny deterministic column offset resolves exact ties without changing
        # non-tied solutions.
        costs = -np.log(np.clip(soft[trial_index], epsilon, None))
        costs = costs + np.arange(n_classes, dtype=float)[None, :] * np.finfo(float).eps
        rows, columns = linear_sum_assignment(costs)
        assignments[trial_index, rows] = columns
        one_hot[trial_index, rows, columns] = 1.0
    return PermutationDecodingResult(probabilities=soft, assignments=assignments, one_hot_probabilities=one_hot)

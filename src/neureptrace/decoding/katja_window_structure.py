"""Leakage-safe helpers for Katja sliding-window adaptation and decoding.

The functions in this module are deliberately independent of the neural model.
They define the target-label boundary, hierarchical six-class probabilities,
calibration-only finger-template learning, and constrained trial decoding.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class TargetCalibrationPartition:
    """Disjoint target calibration, reserved, and evaluation rows."""

    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray
    reserved_indices: np.ndarray
    split_seed: int

    def __post_init__(self) -> None:
        calibration = _index_vector(self.calibration_indices, "calibration_indices")
        evaluation = _index_vector(self.evaluation_indices, "evaluation_indices")
        reserved = _index_vector(self.reserved_indices, "reserved_indices")
        if np.intersect1d(calibration, evaluation).size:
            raise ValueError("Calibration and evaluation rows must be disjoint")
        if np.intersect1d(calibration, reserved).size:
            raise ValueError("Calibration and reserved rows must be disjoint")
        if np.intersect1d(evaluation, reserved).size:
            raise ValueError("Evaluation and reserved rows must be disjoint")
        object.__setattr__(self, "calibration_indices", calibration)
        object.__setattr__(self, "evaluation_indices", evaluation)
        object.__setattr__(self, "reserved_indices", reserved)


@dataclass(frozen=True, slots=True)
class StructuredTrialPrediction:
    """One constrained trial prediction and its selected calibration template."""

    labels: np.ndarray
    states: np.ndarray
    template: tuple[int, ...]
    score: float
    press_segments: tuple[tuple[int, int, int], ...]


@dataclass(frozen=True, slots=True)
class StateDurationPriors:
    """Shrunk explicit-duration priors for ordered rest/press states."""

    means: np.ndarray
    scales: np.ndarray
    minimums: np.ndarray
    maximums: np.ndarray
    source_trial_count: int
    calibration_trial_count: int
    calibration_weight: float

    def __post_init__(self) -> None:
        means = np.asarray(self.means, dtype=np.float64).reshape(-1)
        scales = np.asarray(self.scales, dtype=np.float64).reshape(-1)
        minimums = np.asarray(self.minimums, dtype=np.int64).reshape(-1)
        maximums = np.asarray(self.maximums, dtype=np.int64).reshape(-1)
        if (
            means.size == 0
            or means.size % 2 == 0
            or scales.shape != means.shape
            or minimums.shape != means.shape
            or maximums.shape != means.shape
        ):
            raise ValueError("Duration-prior arrays must have the same odd, nonzero length")
        if not np.all(np.isfinite(means)) or not np.all(np.isfinite(scales)):
            raise ValueError("Duration-prior means and scales must be finite")
        if np.any(means < 0.0) or np.any(scales <= 0.0):
            raise ValueError("Duration-prior means must be nonnegative and scales positive")
        if np.any(minimums < 0) or np.any(maximums < minimums):
            raise ValueError("Duration-prior bounds are invalid")
        if int(self.source_trial_count) <= 0 or int(self.calibration_trial_count) <= 0:
            raise ValueError("Duration priors require source and calibration trials")
        if not 0.0 <= float(self.calibration_weight) <= 1.0:
            raise ValueError("calibration_weight must be in [0, 1]")
        object.__setattr__(self, "means", means)
        object.__setattr__(self, "scales", scales)
        object.__setattr__(self, "minimums", minimums)
        object.__setattr__(self, "maximums", maximums)


def _index_vector(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    if array.size and (np.any(array < 0) or np.unique(array).size != array.size):
        raise ValueError(f"{name} must contain unique nonnegative indices")
    return np.sort(array)


def conditional_finger_targets(
    press_ratios: Sequence[Sequence[float]] | np.ndarray,
    hard_finger_labels: Sequence[int] | np.ndarray,
    *,
    epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize finger occupancy and return rows suitable for conditional loss.

    ``press_ratios`` must contain rest in column 0 and fingers 1..5 in the
    remaining columns. Rows with hard finger labels use a one-hot fallback when
    their supplied finger occupancy is zero. The returned mask never depends on
    evaluation labels when callers pass calibration/source rows only.
    """

    ratios = np.asarray(press_ratios, dtype=np.float64)
    labels = np.asarray(hard_finger_labels, dtype=np.int64).reshape(-1)
    if ratios.ndim != 2 or ratios.shape[1] != 6 or ratios.shape[0] != labels.size:
        raise ValueError("press_ratios must have shape [rows, 6] and align with labels")
    if not np.all(np.isfinite(ratios)) or np.any(ratios < 0.0):
        raise ValueError("press_ratios must be finite and nonnegative")
    if np.any((labels < 0) | (labels > 5)):
        raise ValueError("hard_finger_labels must use classes 0..5")
    finger_mass = ratios[:, 1:6].sum(axis=1)
    active = (labels > 0) | (finger_mass > float(epsilon))
    targets = np.zeros((labels.size, 5), dtype=np.float32)
    nonzero = finger_mass > float(epsilon)
    targets[nonzero] = (ratios[nonzero, 1:6] / finger_mass[nonzero, None]).astype(np.float32)
    fallback = active & ~nonzero
    if np.any(fallback):
        if np.any(labels[fallback] == 0):
            raise ValueError("A zero-occupancy row without a hard finger cannot define a conditional target")
        targets[np.flatnonzero(fallback), labels[fallback] - 1] = 1.0
    return targets, active


def combine_hierarchical_probabilities(
    press_probabilities: Sequence[Sequence[float]] | np.ndarray,
    conditional_finger_probabilities: Sequence[Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Combine ``P(rest/press)`` and ``P(finger | press)`` into six classes."""

    press = np.asarray(press_probabilities, dtype=np.float64)
    finger = np.asarray(conditional_finger_probabilities, dtype=np.float64)
    if press.ndim == 1:
        press = np.column_stack((1.0 - press, press))
    if press.ndim != 2 or press.shape[1] != 2:
        raise ValueError("press_probabilities must have shape [rows, 2] or contain P(press)")
    if finger.ndim != 2 or finger.shape != (press.shape[0], 5):
        raise ValueError("conditional_finger_probabilities must have shape [rows, 5]")
    if not np.all(np.isfinite(press)) or not np.all(np.isfinite(finger)):
        raise ValueError("probabilities must be finite")
    if np.any(press < 0.0) or np.any(finger < 0.0):
        raise ValueError("probabilities must be nonnegative")
    press = press / np.maximum(press.sum(axis=1, keepdims=True), 1e-12)
    finger = finger / np.maximum(finger.sum(axis=1, keepdims=True), 1e-12)
    combined = np.empty((press.shape[0], 6), dtype=np.float64)
    combined[:, 0] = press[:, 0]
    combined[:, 1:6] = press[:, 1, None] * finger
    combined /= np.maximum(combined.sum(axis=1, keepdims=True), 1e-12)
    return combined


def match_probability_marginals(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    target_prior: Sequence[float] | np.ndarray,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-8,
    damping: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply class biases so an unlabeled batch matches a calibration prior.

    This is transductive adaptation: the batch features/probabilities are used,
    but its labels are not. Callers must report it separately from a strict
    calibration-only Protocol 3 result.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    prior = np.asarray(target_prior, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or values.shape[1] != prior.size or values.shape[0] == 0:
        raise ValueError("probabilities and target_prior have incompatible shapes")
    if (
        not np.all(np.isfinite(values))
        or not np.all(np.isfinite(prior))
        or np.any(values < 0.0)
        or np.any(prior <= 0.0)
    ):
        raise ValueError("Probabilities must be nonnegative and target_prior strictly positive")
    if int(max_iterations) <= 0 or not np.isfinite(tolerance) or float(tolerance) <= 0.0:
        raise ValueError("max_iterations and tolerance must be positive")
    if not np.isfinite(damping) or not 0.0 < float(damping) <= 1.0:
        raise ValueError("damping must be in (0, 1]")
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
    prior /= prior.sum()
    biases = np.ones(prior.size, dtype=np.float64)
    adjusted = values.copy()
    for _ in range(int(max_iterations)):
        adjusted = values * biases[None, :]
        adjusted /= np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)
        marginal = adjusted.mean(axis=0)
        error = float(np.max(np.abs(marginal - prior)))
        if error <= float(tolerance):
            break
        biases *= np.power(prior / np.maximum(marginal, 1e-12), float(damping))
        biases /= np.exp(np.mean(np.log(np.maximum(biases, 1e-300))))
    adjusted = values * biases[None, :]
    adjusted /= np.maximum(adjusted.sum(axis=1, keepdims=True), 1e-12)
    return adjusted, biases


def compose_template_order_finger_probabilities(
    press_probabilities: Sequence[Sequence[float]] | np.ndarray,
    order_probabilities: Sequence[Sequence[float]] | np.ndarray,
    template_probabilities: Sequence[Sequence[float]] | np.ndarray,
    templates: Sequence[Sequence[int]],
) -> np.ndarray:
    """Map press, serial-position, and template heads into six finger classes."""

    press = np.asarray(press_probabilities, dtype=np.float64)
    order = np.asarray(order_probabilities, dtype=np.float64)
    template = np.asarray(template_probabilities, dtype=np.float64)
    clean_templates = tuple(tuple(int(finger) for finger in item) for item in templates)
    if press.ndim != 2 or press.shape[1] != 2:
        raise ValueError("press_probabilities must have shape [rows, 2]")
    if order.shape != (press.shape[0], 6):
        raise ValueError("order_probabilities must have shape [rows, 6]")
    if template.shape != (press.shape[0], len(clean_templates)):
        raise ValueError("template_probabilities do not align with templates")
    if not clean_templates or any(
        len(item) != 5 or set(item) != {1, 2, 3, 4, 5} for item in clean_templates
    ):
        raise ValueError("templates must be five-finger permutations")
    for name, values in (("press", press), ("order", order), ("template", template)):
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError(f"{name} probabilities must be finite and nonnegative")
    press /= np.maximum(press.sum(axis=1, keepdims=True), 1e-12)
    conditional_order = order[:, 1:6]
    conditional_order /= np.maximum(conditional_order.sum(axis=1, keepdims=True), 1e-12)
    template /= np.maximum(template.sum(axis=1, keepdims=True), 1e-12)
    conditional_finger = np.zeros((press.shape[0], 5), dtype=np.float64)
    for template_index, item in enumerate(clean_templates):
        for position, finger in enumerate(item):
            conditional_finger[:, finger - 1] += (
                template[:, template_index] * conditional_order[:, position]
            )
    conditional_finger /= np.maximum(conditional_finger.sum(axis=1, keepdims=True), 1e-12)
    return combine_hierarchical_probabilities(press, conditional_finger)


def balanced_window_sampling_weights(
    subject_ids: Sequence[Any] | np.ndarray,
    trial_ids: Sequence[Any] | np.ndarray,
    class_labels: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Return normalized weights balancing subjects, trials, and classes."""

    subject = np.asarray(subject_ids).reshape(-1)
    trial = np.asarray(trial_ids).reshape(-1)
    labels = np.asarray(class_labels).reshape(-1)
    if subject.size == 0 or subject.shape != trial.shape or subject.shape != labels.shape:
        raise ValueError("subject_ids, trial_ids, and class_labels must be aligned nonempty vectors")

    def inverse_count(values: Iterable[Any]) -> np.ndarray:
        values = list(values)
        counts: dict[Any, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return np.asarray([1.0 / counts[value] for value in values], dtype=np.float64)

    trial_keys = [(str(s), str(t)) for s, t in zip(subject.tolist(), trial.tolist(), strict=True)]
    weights = inverse_count(subject.tolist()) * inverse_count(trial_keys) * inverse_count(labels.tolist())
    weights /= weights.sum()
    return weights


def learn_finger_templates(
    finger_labels: Sequence[int] | np.ndarray,
    press_order: Sequence[int] | np.ndarray,
    trial_ids: Sequence[Any] | np.ndarray,
    *,
    calibration_indices: Sequence[int] | np.ndarray,
    evaluation_indices: Sequence[int] | np.ndarray | None = None,
    expected_presses: int = 5,
    expected_templates: int = 2,
) -> tuple[tuple[int, ...], ...]:
    """Learn ordered finger templates from calibration rows only.

    Every returned template is observed in a complete calibration trial. Target
    evaluation rows are accepted only as a guardrail and are never indexed.
    """

    labels = np.asarray(finger_labels, dtype=np.int64).reshape(-1)
    order = np.asarray(press_order, dtype=np.int64).reshape(-1)
    trials = np.asarray(trial_ids).reshape(-1)
    if labels.shape != order.shape or labels.shape != trials.shape:
        raise ValueError("finger_labels, press_order, and trial_ids must align")
    calibration = _index_vector(calibration_indices, "calibration_indices")
    if evaluation_indices is not None:
        evaluation = _index_vector(evaluation_indices, "evaluation_indices")
        if np.intersect1d(calibration, evaluation).size:
            raise ValueError("Template fitting cannot use evaluation rows")
    if calibration.size == 0:
        raise ValueError("Template fitting requires calibration rows")
    templates: list[tuple[int, ...]] = []
    for trial in np.unique(trials[calibration]):
        rows = calibration[trials[calibration] == trial]
        template: list[int] = []
        complete = True
        for position in range(1, int(expected_presses) + 1):
            candidates = labels[rows][order[rows] == position]
            candidates = candidates[candidates > 0]
            if candidates.size == 0:
                complete = False
                break
            values, counts = np.unique(candidates, return_counts=True)
            template.append(int(values[np.argmax(counts)]))
        candidate = tuple(template)
        if complete and len(set(candidate)) == expected_presses and candidate not in templates:
            templates.append(candidate)
    if len(templates) != int(expected_templates):
        raise ValueError(
            f"Expected exactly {int(expected_templates)} complete five-press templates "
            f"from calibration trials, observed {len(templates)}"
        )
    return tuple(templates)


def estimate_state_stay_probabilities(
    finger_labels: Sequence[int] | np.ndarray,
    press_order: Sequence[int] | np.ndarray,
    trial_ids: Sequence[Any] | np.ndarray,
    *,
    fitting_indices: Sequence[int] | np.ndarray,
    evaluation_indices: Sequence[int] | np.ndarray | None = None,
    expected_presses: int = 5,
) -> np.ndarray:
    """Estimate rest/press self-transition probabilities without evaluation labels."""

    labels = np.asarray(finger_labels, dtype=np.int64).reshape(-1)
    order = np.asarray(press_order, dtype=np.int64).reshape(-1)
    trials = np.asarray(trial_ids).reshape(-1)
    fitting = _index_vector(fitting_indices, "fitting_indices")
    if labels.shape != order.shape or labels.shape != trials.shape:
        raise ValueError("finger_labels, press_order, and trial_ids must align")
    if evaluation_indices is not None and np.intersect1d(fitting, _index_vector(evaluation_indices, "evaluation_indices")).size:
        raise ValueError("Duration fitting cannot use evaluation rows")
    run_lengths: list[list[int]] = [[] for _ in range(2 * expected_presses + 1)]
    for trial in np.unique(trials[fitting]):
        rows = fitting[trials[fitting] == trial]
        state = np.where(order[rows] > 0, 2 * order[rows] - 1, 0)
        # Rest states are placed between the last completed and next press.
        last_press = 0
        for index, value in enumerate(state.tolist()):
            if value > 0:
                last_press = int((value + 1) // 2)
            else:
                state[index] = 2 * last_press
        start = 0
        while start < state.size:
            stop = start + 1
            while stop < state.size and state[stop] == state[start]:
                stop += 1
            if 0 <= state[start] < len(run_lengths):
                run_lengths[int(state[start])].append(stop - start)
            start = stop
    probabilities = np.full(2 * expected_presses + 1, 0.88, dtype=np.float64)
    for index, lengths in enumerate(run_lengths):
        if lengths:
            mean_length = max(1.0, float(np.mean(lengths)))
            probabilities[index] = np.clip(1.0 - 1.0 / mean_length, 0.05, 0.995)
    return probabilities


def _trial_state_duration_matrix(
    finger_labels: np.ndarray,
    press_order: np.ndarray,
    trial_ids: np.ndarray,
    indices: np.ndarray,
    *,
    expected_presses: int,
) -> np.ndarray:
    """Return one canonical ordered-state duration vector per complete trial."""

    n_states = 2 * int(expected_presses) + 1
    durations: list[np.ndarray] = []
    for trial in np.unique(trial_ids[indices]):
        rows = indices[trial_ids[indices] == trial]
        active = (finger_labels[rows] > 0) & (press_order[rows] > 0)
        observed_order = press_order[rows][active]
        if set(observed_order.tolist()) != set(range(1, int(expected_presses) + 1)):
            continue
        states = np.zeros(rows.size, dtype=np.int64)
        last_press = 0
        for row, (is_press, order) in enumerate(
            zip(active.tolist(), press_order[rows].tolist(), strict=True)
        ):
            if is_press:
                last_press = int(order)
                states[row] = 2 * last_press - 1
            else:
                states[row] = 2 * last_press
        if np.any(np.diff(states) < 0):
            continue
        trial_durations = np.bincount(states, minlength=n_states)[:n_states]
        if np.all(trial_durations[1::2] > 0):
            durations.append(trial_durations.astype(np.float64))
    if not durations:
        raise ValueError("No complete ordered trials were available for duration fitting")
    return np.stack(durations, axis=0)


def estimate_state_duration_priors(
    finger_labels: Sequence[int] | np.ndarray,
    press_order: Sequence[int] | np.ndarray,
    trial_ids: Sequence[Any] | np.ndarray,
    *,
    source_indices: Sequence[int] | np.ndarray,
    calibration_indices: Sequence[int] | np.ndarray,
    evaluation_indices: Sequence[int] | np.ndarray | None = None,
    expected_presses: int = 5,
    calibration_prior_strength: float = 8.0,
    scale_floor: float = 1.5,
    bound_standard_deviations: float = 4.0,
) -> StateDurationPriors:
    """Estimate endpoint-matched durations from source and target calibration rows.

    Source trials form the population prior. The target calibration mean and
    variance receive an empirical-Bayes weight of ``n / (n + strength)`` where
    ``n`` is the number of complete target calibration trials. Target
    evaluation labels are accepted only as a disjointness guard and are never
    indexed.
    """

    labels = np.asarray(finger_labels, dtype=np.int64).reshape(-1)
    order = np.asarray(press_order, dtype=np.int64).reshape(-1)
    trials = np.asarray(trial_ids).reshape(-1)
    if labels.shape != order.shape or labels.shape != trials.shape:
        raise ValueError("finger_labels, press_order, and trial_ids must align")
    if int(expected_presses) <= 0:
        raise ValueError("expected_presses must be positive")
    strength = float(calibration_prior_strength)
    if not np.isfinite(strength) or strength < 0.0:
        raise ValueError("calibration_prior_strength must be finite and nonnegative")
    floor = float(scale_floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("scale_floor must be finite and positive")
    bound_sd = float(bound_standard_deviations)
    if not np.isfinite(bound_sd) or bound_sd <= 0.0:
        raise ValueError("bound_standard_deviations must be finite and positive")

    source = _index_vector(source_indices, "source_indices")
    calibration = _index_vector(calibration_indices, "calibration_indices")
    if source.size == 0 or calibration.size == 0:
        raise ValueError("Duration fitting requires source and calibration rows")
    if np.intersect1d(source, calibration).size:
        raise ValueError("Source and calibration rows must be disjoint")
    if evaluation_indices is not None:
        evaluation = _index_vector(evaluation_indices, "evaluation_indices")
        if np.intersect1d(source, evaluation).size or np.intersect1d(calibration, evaluation).size:
            raise ValueError("Duration fitting cannot use evaluation rows")

    source_durations = _trial_state_duration_matrix(
        labels,
        order,
        trials,
        source,
        expected_presses=int(expected_presses),
    )
    calibration_durations = _trial_state_duration_matrix(
        labels,
        order,
        trials,
        calibration,
        expected_presses=int(expected_presses),
    )
    calibration_count = int(calibration_durations.shape[0])
    calibration_weight = (
        1.0 if strength == 0.0 else calibration_count / (calibration_count + strength)
    )
    source_means = source_durations.mean(axis=0)
    calibration_means = calibration_durations.mean(axis=0)
    source_variances = source_durations.var(axis=0, ddof=1 if source_durations.shape[0] > 1 else 0)
    calibration_variances = calibration_durations.var(
        axis=0,
        ddof=1 if calibration_durations.shape[0] > 1 else 0,
    )
    means = (
        (1.0 - calibration_weight) * source_means
        + calibration_weight * calibration_means
    )
    variances = (
        (1.0 - calibration_weight)
        * (source_variances + np.square(source_means - means))
        + calibration_weight
        * (calibration_variances + np.square(calibration_means - means))
    )
    scales = np.maximum(np.sqrt(np.maximum(variances, 0.0)), floor)
    minimums = np.maximum(0, np.floor(means - bound_sd * scales)).astype(np.int64)
    maximums = np.ceil(means + bound_sd * scales).astype(np.int64)
    minimums[1::2] = np.maximum(1, minimums[1::2])
    maximums = np.maximum(maximums, minimums)
    return StateDurationPriors(
        means=means,
        scales=scales,
        minimums=minimums,
        maximums=maximums,
        source_trial_count=int(source_durations.shape[0]),
        calibration_trial_count=calibration_count,
        calibration_weight=float(calibration_weight),
    )


def _state_emissions(
    probabilities: np.ndarray,
    template: tuple[int, ...],
    *,
    order_probabilities: np.ndarray | None = None,
    overlap_probabilities: np.ndarray | None = None,
    order_weight: float = 0.35,
    overlap_weight: float = 0.15,
    rest_log_bias: float = 0.0,
) -> np.ndarray:
    n_states = 2 * len(template) + 1
    emissions = np.empty((probabilities.shape[0], n_states), dtype=np.float64)
    for state in range(n_states):
        is_press = state % 2 == 1
        finger = 0 if not is_press else template[(state - 1) // 2]
        emissions[:, state] = np.log(np.clip(probabilities[:, finger], 1e-12, 1.0))
        if not is_press:
            emissions[:, state] += float(rest_log_bias)
        if order_probabilities is not None:
            order_class = 0 if not is_press else (state + 1) // 2
            emissions[:, state] += float(order_weight) * np.log(
                np.clip(order_probabilities[:, order_class], 1e-12, 1.0)
            )
        if overlap_probabilities is not None:
            overlap_emission = (
                overlap_probabilities if is_press else 1.0 - overlap_probabilities
            )
            emissions[:, state] += float(overlap_weight) * np.log(
                np.clip(overlap_emission, 1e-12, 1.0)
            )
    return emissions


def _viterbi_for_template(
    probabilities: np.ndarray,
    template: tuple[int, ...],
    stay_probabilities: np.ndarray,
    *,
    require_complete: bool,
    order_probabilities: np.ndarray | None = None,
    overlap_probabilities: np.ndarray | None = None,
    order_weight: float = 0.35,
    overlap_weight: float = 0.15,
) -> tuple[np.ndarray, float]:
    emissions = _state_emissions(
        probabilities,
        template,
        order_probabilities=order_probabilities,
        overlap_probabilities=overlap_probabilities,
        order_weight=order_weight,
        overlap_weight=overlap_weight,
    )
    n_rows, n_states = emissions.shape
    if stay_probabilities.shape != (n_states,):
        raise ValueError(f"stay_probabilities must contain {n_states} values")
    stay = np.log(np.clip(stay_probabilities, 1e-6, 1.0 - 1e-6))
    advance = np.log(np.clip(1.0 - stay_probabilities, 1e-6, 1.0))
    scores = np.full((n_rows, n_states), -np.inf, dtype=np.float64)
    back = np.full((n_rows, n_states), -1, dtype=np.int16)
    scores[0, 0] = emissions[0, 0]
    # Starting in the first press allows trials whose first window overlaps it.
    if n_states > 1:
        scores[0, 1] = emissions[0, 1] + advance[0]
        back[0, 1] = 0
    for row in range(1, n_rows):
        for state in range(n_states):
            same_score = scores[row - 1, state] + stay[state]
            previous_score = -np.inf if state == 0 else scores[row - 1, state - 1] + advance[state - 1]
            if previous_score > same_score:
                scores[row, state] = previous_score + emissions[row, state]
                back[row, state] = state - 1
            else:
                scores[row, state] = same_score + emissions[row, state]
                back[row, state] = state
    if require_complete:
        candidates = np.asarray([n_states - 2, n_states - 1], dtype=int)
    else:
        candidates = np.arange(n_states, dtype=int)
    final_state = int(candidates[np.argmax(scores[-1, candidates])])
    states = np.empty(n_rows, dtype=np.int64)
    states[-1] = final_state
    for row in range(n_rows - 1, 0, -1):
        previous = int(back[row, states[row]])
        states[row - 1] = states[row] if previous < 0 else previous
    return states, float(scores[-1, final_state])


def _segments_from_states(states: np.ndarray, template: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
    segments: list[tuple[int, int, int]] = []
    for position, finger in enumerate(template, start=1):
        rows = np.flatnonzero(states == 2 * position - 1)
        if rows.size:
            segments.append((int(rows[0]), int(rows[-1]) + 1, int(finger)))
    return tuple(segments)


def structured_trial_decode(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    templates: Sequence[Sequence[int]],
    *,
    stay_probabilities: Sequence[float] | np.ndarray | None = None,
    order_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    overlap_probabilities: Sequence[float] | np.ndarray | None = None,
    template_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    order_weight: float = 0.35,
    overlap_weight: float = 0.15,
    template_weight: float = 0.15,
) -> StructuredTrialPrediction:
    """Offline Viterbi decode over calibration-observed five-press templates.

    Auxiliary order, overlap, and template predictions can refine the state
    emissions. Their fixed weights are intentionally independent of target
    evaluation labels.
    """

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] < 5:
        raise ValueError("probabilities must have shape [at least five windows, 6]")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
    clean_templates = tuple(tuple(int(finger) for finger in template) for template in templates)
    if not clean_templates or any(len(template) != 5 or set(template) != {1, 2, 3, 4, 5} for template in clean_templates):
        raise ValueError("templates must be five-finger permutations")
    order_values = _optional_probability_matrix(
        order_probabilities, rows=values.shape[0], columns=6, name="order_probabilities"
    )
    overlap_values = _optional_probability_vector(
        overlap_probabilities, rows=values.shape[0], name="overlap_probabilities"
    )
    template_values = _optional_probability_matrix(
        template_probabilities,
        rows=values.shape[0],
        columns=len(clean_templates),
        name="template_probabilities",
    )
    stay = np.asarray(stay_probabilities if stay_probabilities is not None else np.full(11, 0.88), dtype=np.float64)
    candidates: list[tuple[np.ndarray, float, tuple[int, ...]]] = []
    template_prior = (
        np.full(len(clean_templates), 1.0 / len(clean_templates), dtype=np.float64)
        if template_values is None
        else template_values.mean(axis=0)
    )
    template_prior /= np.maximum(template_prior.sum(), 1e-12)
    for template_index, template in enumerate(clean_templates):
        states, score = _viterbi_for_template(
            values,
            template,
            stay,
            require_complete=True,
            order_probabilities=order_values,
            overlap_probabilities=overlap_values,
            order_weight=order_weight,
            overlap_weight=overlap_weight,
        )
        score += float(template_weight) * float(
            np.log(np.clip(template_prior[template_index], 1e-12, 1.0))
        )
        candidates.append((states, score, template))
    states, score, template = max(candidates, key=lambda item: item[1])
    labels = np.zeros(values.shape[0], dtype=np.int64)
    for position, finger in enumerate(template, start=1):
        labels[states == 2 * position - 1] = finger
    return StructuredTrialPrediction(
        labels=labels,
        states=states,
        template=template,
        score=score,
        press_segments=_segments_from_states(states, template),
    )


def _hsmm_for_template(
    probabilities: np.ndarray,
    template: tuple[int, ...],
    duration_priors: StateDurationPriors,
    *,
    order_probabilities: np.ndarray | None = None,
    overlap_probabilities: np.ndarray | None = None,
    order_weight: float = 0.35,
    overlap_weight: float = 0.15,
    duration_weight: float = 1.0,
    press_duration_scale: float = 1.0,
    rest_duration_scale: float = 1.0,
    rest_log_bias: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Decode one template with explicit ordered-state duration likelihoods."""

    emissions = _state_emissions(
        probabilities,
        template,
        order_probabilities=order_probabilities,
        overlap_probabilities=overlap_probabilities,
        order_weight=order_weight,
        overlap_weight=overlap_weight,
        rest_log_bias=rest_log_bias,
    )
    n_rows, n_states = emissions.shape
    if duration_priors.means.shape != (n_states,):
        raise ValueError(f"duration_priors must contain {n_states} states")
    if not np.isfinite(duration_weight) or duration_weight < 0.0:
        raise ValueError("duration_weight must be finite and nonnegative")
    if (
        not np.isfinite(press_duration_scale)
        or press_duration_scale <= 0.0
        or not np.isfinite(rest_duration_scale)
        or rest_duration_scale <= 0.0
        or not np.isfinite(rest_log_bias)
    ):
        raise ValueError("Duration scales must be positive and rest_log_bias finite")

    scale_by_state = np.where(
        np.arange(n_states) % 2 == 1,
        float(press_duration_scale),
        float(rest_duration_scale),
    )
    means = duration_priors.means * scale_by_state
    scales = duration_priors.scales * np.sqrt(scale_by_state)
    minimums = np.floor(duration_priors.minimums * scale_by_state).astype(np.int64)
    maximums = np.ceil(duration_priors.maximums * scale_by_state).astype(np.int64)
    minimums[1::2] = np.maximum(1, minimums[1::2])
    maximums = np.maximum(maximums, minimums)
    maximums = np.minimum(maximums, n_rows)
    if int(minimums.sum()) > n_rows or int(maximums.sum()) < n_rows:
        raise ValueError("Duration-prior bounds cannot cover the supplied trial")

    cumulative = np.vstack((np.zeros((1, n_states)), np.cumsum(emissions, axis=0)))
    scores = np.full((n_states, n_rows + 1), -np.inf, dtype=np.float64)
    back_duration = np.full((n_states, n_rows + 1), -1, dtype=np.int16)
    for state in range(n_states):
        minimum = int(minimums[state])
        maximum = int(maximums[state])
        duration_values = np.arange(minimum, maximum + 1, dtype=np.int64)
        duration_scores = -0.5 * np.square(
            (duration_values.astype(np.float64) - means[state]) / scales[state]
        ) - np.log(scales[state])
        duration_scores *= float(duration_weight)
        for end in range(n_rows + 1):
            valid = duration_values <= end
            if not np.any(valid):
                continue
            durations = duration_values[valid]
            starts = end - durations
            if state == 0:
                base_scores = np.where(starts == 0, 0.0, -np.inf)
            else:
                base_scores = scores[state - 1, starts]
            candidates = (
                base_scores
                + cumulative[end, state]
                - cumulative[starts, state]
                + duration_scores[valid]
            )
            best = int(np.argmax(candidates))
            if np.isfinite(candidates[best]):
                scores[state, end] = float(candidates[best])
                back_duration[state, end] = int(durations[best])
    if not np.isfinite(scores[-1, n_rows]):
        raise ValueError("No explicit-duration path covers the supplied trial")
    states = np.empty(n_rows, dtype=np.int64)
    end = n_rows
    for state in range(n_states - 1, -1, -1):
        duration = int(back_duration[state, end])
        if duration < 0:
            raise RuntimeError("Explicit-duration backtrace is incomplete")
        start = end - duration
        states[start:end] = state
        end = start
    if end != 0:
        raise RuntimeError("Explicit-duration backtrace did not consume the trial")
    return states, float(scores[-1, n_rows])


def explicit_duration_trial_decode(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    templates: Sequence[Sequence[int]],
    *,
    duration_priors: StateDurationPriors,
    order_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    overlap_probabilities: Sequence[float] | np.ndarray | None = None,
    template_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    order_weight: float = 0.35,
    overlap_weight: float = 0.15,
    template_weight: float = 0.15,
    duration_weight: float = 1.0,
    press_duration_scale: float = 1.0,
    rest_duration_scale: float = 1.0,
    rest_log_bias: float = 0.0,
) -> StructuredTrialPrediction:
    """Offline semi-Markov decode with source-plus-calibration duration priors."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6 or values.shape[0] < 5:
        raise ValueError("probabilities must have shape [at least five windows, 6]")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probabilities must be finite and nonnegative")
    values /= np.maximum(values.sum(axis=1, keepdims=True), 1e-12)
    clean_templates = tuple(tuple(int(finger) for finger in template) for template in templates)
    if not clean_templates or any(
        len(template) != 5 or set(template) != {1, 2, 3, 4, 5}
        for template in clean_templates
    ):
        raise ValueError("templates must be five-finger permutations")
    order_values = _optional_probability_matrix(
        order_probabilities, rows=values.shape[0], columns=6, name="order_probabilities"
    )
    overlap_values = _optional_probability_vector(
        overlap_probabilities, rows=values.shape[0], name="overlap_probabilities"
    )
    template_values = _optional_probability_matrix(
        template_probabilities,
        rows=values.shape[0],
        columns=len(clean_templates),
        name="template_probabilities",
    )
    template_prior = (
        np.full(len(clean_templates), 1.0 / len(clean_templates), dtype=np.float64)
        if template_values is None
        else template_values.mean(axis=0)
    )
    template_prior /= np.maximum(template_prior.sum(), 1e-12)
    candidates: list[tuple[np.ndarray, float, tuple[int, ...]]] = []
    for template_index, template in enumerate(clean_templates):
        states, score = _hsmm_for_template(
            values,
            template,
            duration_priors,
            order_probabilities=order_values,
            overlap_probabilities=overlap_values,
            order_weight=order_weight,
            overlap_weight=overlap_weight,
            duration_weight=duration_weight,
            press_duration_scale=press_duration_scale,
            rest_duration_scale=rest_duration_scale,
            rest_log_bias=rest_log_bias,
        )
        score += float(template_weight) * float(
            np.log(np.clip(template_prior[template_index], 1e-12, 1.0))
        )
        candidates.append((states, score, template))
    states, score, template = max(candidates, key=lambda item: item[1])
    labels = np.zeros(values.shape[0], dtype=np.int64)
    for position, finger in enumerate(template, start=1):
        labels[states == 2 * position - 1] = finger
    return StructuredTrialPrediction(
        labels=labels,
        states=states,
        template=template,
        score=score,
        press_segments=_segments_from_states(states, template),
    )


def causal_trial_decode(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    templates: Sequence[Sequence[int]],
    *,
    stay_probabilities: Sequence[float] | np.ndarray | None = None,
    order_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    overlap_probabilities: Sequence[float] | np.ndarray | None = None,
    template_probabilities: Sequence[Sequence[float]] | np.ndarray | None = None,
    order_weight: float = 0.35,
    overlap_weight: float = 0.15,
    template_weight: float = 0.15,
) -> np.ndarray:
    """Prefix-only constrained predictions; future rows cannot change a prefix."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("probabilities must have shape [windows, 6]")
    clean_templates = tuple(tuple(int(finger) for finger in template) for template in templates)
    if not clean_templates:
        raise ValueError("At least one template is required")
    order_values = _optional_probability_matrix(
        order_probabilities, rows=values.shape[0], columns=6, name="order_probabilities"
    )
    overlap_values = _optional_probability_vector(
        overlap_probabilities, rows=values.shape[0], name="overlap_probabilities"
    )
    template_values = _optional_probability_matrix(
        template_probabilities,
        rows=values.shape[0],
        columns=len(clean_templates),
        name="template_probabilities",
    )
    stay = np.asarray(stay_probabilities if stay_probabilities is not None else np.full(11, 0.88), dtype=np.float64)
    output = np.zeros(values.shape[0], dtype=np.int64)
    # A prefix decode intentionally permits incomplete paths and emits only its
    # current final state. Recomputing prefixes is slower but easy to audit.
    for stop in range(1, values.shape[0] + 1):
        best: tuple[np.ndarray, float, tuple[int, ...]] | None = None
        prefix_template_prior = (
            np.full(len(clean_templates), 1.0 / len(clean_templates), dtype=np.float64)
            if template_values is None
            else template_values[:stop].mean(axis=0)
        )
        prefix_template_prior /= np.maximum(prefix_template_prior.sum(), 1e-12)
        for template_index, template in enumerate(clean_templates):
            states, score = _viterbi_for_template(
                values[:stop],
                template,
                stay,
                require_complete=False,
                order_probabilities=None if order_values is None else order_values[:stop],
                overlap_probabilities=None if overlap_values is None else overlap_values[:stop],
                order_weight=order_weight,
                overlap_weight=overlap_weight,
            )
            score += float(template_weight) * float(
                np.log(np.clip(prefix_template_prior[template_index], 1e-12, 1.0))
            )
            candidate = (states, score, template)
            if best is None or score > best[1]:
                best = candidate
        assert best is not None
        state = int(best[0][-1])
        output[stop - 1] = 0 if state % 2 == 0 else best[2][(state - 1) // 2]
    return output


def _optional_probability_matrix(
    values: Sequence[Sequence[float]] | np.ndarray | None,
    *,
    rows: int,
    columns: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (rows, columns):
        raise ValueError(f"{name} must have shape [{rows}, {columns}]")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return array / np.maximum(array.sum(axis=1, keepdims=True), 1e-12)


def _optional_probability_vector(
    values: Sequence[float] | np.ndarray | None,
    *,
    rows: int,
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (rows,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {rows} finite values")
    if np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be in [0, 1]")
    return array


def write_prediction_bundle(
    path: str | Path,
    *,
    row_indices: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    split_seed: int,
    model_seed: int,
    method: str,
    auxiliary: dict[str, np.ndarray] | None = None,
) -> Path:
    """Atomically persist row-aligned per-window predictions and auxiliary heads."""

    output = Path(path)
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.shape != (rows.size, 6):
        raise ValueError("probabilities must have one six-class row per row index")
    if np.unique(rows).size != rows.size:
        raise ValueError("row_indices must be unique")
    payload: dict[str, np.ndarray] = {
        "row_indices": rows,
        "probabilities": probs,
        "split_seed": np.asarray(int(split_seed), dtype=np.int64),
        "model_seed": np.asarray(int(model_seed), dtype=np.int64),
        "method": np.asarray(str(method)),
    }
    for name, values in (auxiliary or {}).items():
        array = np.asarray(values)
        if array.ndim == 0 or array.shape[0] != rows.size:
            raise ValueError(f"Auxiliary output {name!r} must align with row_indices")
        payload[f"aux_{name}"] = array
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, output)
    return output


def ensemble_prediction_bundles(paths: Sequence[str | Path]) -> dict[str, Any]:
    """Average bundles after enforcing identical split rows and distinct model seeds."""

    if not paths:
        raise ValueError("At least one prediction bundle is required")
    loaded: list[dict[str, np.ndarray]] = []
    for path in paths:
        with np.load(path, allow_pickle=False) as bundle:
            loaded.append({name: np.asarray(bundle[name]) for name in bundle.files})
    reference_rows = loaded[0]["row_indices"]
    split_seed = int(loaded[0]["split_seed"])
    model_seeds: list[int] = []
    for bundle in loaded:
        if int(bundle["split_seed"]) != split_seed:
            raise ValueError("Prediction bundles use different split seeds")
        if not np.array_equal(bundle["row_indices"], reference_rows):
            raise ValueError("Prediction bundles do not contain identical evaluation rows")
        model_seeds.append(int(bundle["model_seed"]))
    if len(set(model_seeds)) != len(model_seeds):
        raise ValueError("Prediction bundles must use distinct model seeds")
    probabilities = np.mean(np.stack([bundle["probabilities"] for bundle in loaded], axis=0), axis=0)
    result: dict[str, Any] = {
        "row_indices": reference_rows.copy(),
        "probabilities": probabilities,
        "split_seed": split_seed,
        "model_seeds": tuple(model_seeds),
    }
    auxiliary_names = set.intersection(*(set(bundle) for bundle in loaded)) - {
        "row_indices",
        "probabilities",
        "split_seed",
        "model_seed",
        "method",
    }
    for name in sorted(auxiliary_names):
        result[name] = np.mean(np.stack([bundle[name] for bundle in loaded], axis=0), axis=0)
    return result


def write_partition_audit(path: str | Path, partition: TargetCalibrationPartition, **metadata: Any) -> Path:
    """Write an inspectable label-access manifest for one target split."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "split_seed": int(partition.split_seed),
        "n_calibration_rows": int(partition.calibration_indices.size),
        "n_reserved_rows": int(partition.reserved_indices.size),
        "n_evaluation_rows": int(partition.evaluation_indices.size),
        "calibration_evaluation_disjoint": True,
        "evaluation_labels_available_to_fitting": False,
        **metadata,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


__all__ = [
    "StateDurationPriors",
    "StructuredTrialPrediction",
    "TargetCalibrationPartition",
    "balanced_window_sampling_weights",
    "causal_trial_decode",
    "combine_hierarchical_probabilities",
    "compose_template_order_finger_probabilities",
    "conditional_finger_targets",
    "ensemble_prediction_bundles",
    "estimate_state_duration_priors",
    "estimate_state_stay_probabilities",
    "explicit_duration_trial_decode",
    "learn_finger_templates",
    "match_probability_marginals",
    "structured_trial_decode",
    "write_partition_audit",
    "write_prediction_bundle",
]

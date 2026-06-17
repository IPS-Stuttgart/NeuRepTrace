"""Leakage-safe source-OOF probability stacking.

This module generalizes the useful PyMEGDec source-only logit-stacking idea to
NeuRepTrace probability-observation tables.  The stacker is fitted on
source-subject/source-fold out-of-fold probability rows and is then applied to a
separate target table.  Target labels are used only for reporting the resulting
prediction columns and optional metrics; they are not used when fitting weights.
"""

from __future__ import annotations

import argparse
import glob
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from neureptrace.metrics import brier_score_multiclass, expected_calibration_error
from neureptrace.observations import ProbabilityObservationTable, probability_columns, stable_hash

DEFAULT_CANDIDATE_COLUMN = "decoder"
DEFAULT_WEIGHTING = "stacked"
DEFAULT_POOLING = "linear"
DEFAULT_TEMPERATURE = 0.02
DEFAULT_MAX_ITER = 250
DEFAULT_LEARNING_RATE = 0.25
DEFAULT_MIN_PROBABILITY = 1.0e-12
DEFAULT_OUTPUT_DECODER = "source_oof_stacked_ensemble"
DEFAULT_OUTPUT_EMISSION_MODE = "source_oof_stacked"
WEIGHTING_MODES = {"uniform", "softmax", "stacked"}
POOLING_MODES = {"auto", "linear", "log"}

_BASE_ALIGNMENT_COLUMNS = (
    "subject",
    "session",
    "stream_id",
    "fold",
    "split_id",
    "seed",
    "train_time",
    "test_time",
    "time",
    "window_start",
    "window_stop",
    "sample_index",
    "sequence_id",
    "true_label",
    "true_class",
    "group",
)
_METRIC_GROUP_COLUMNS = ("subject", "fold", "decoder", "emission_mode", "time", "window_start", "window_stop")
_ROW_IDENTITY_COLUMNS = ("true_label", "true_class")


@dataclass(frozen=True, slots=True)
class SourceOOFStackingFit:
    """Fitted source-only stacking parameters."""

    candidates: tuple[str, ...]
    weights: tuple[float, ...]
    weighting: str
    pooling: str
    temperature: float | None
    source_oof_balanced_accuracy: float
    source_oof_log_loss: float


@dataclass(frozen=True, slots=True)
class AlignedProbabilityCube:
    """Candidate-aligned probability rows."""

    base: pd.DataFrame
    cube: np.ndarray
    label_positions: np.ndarray | None
    label_values: tuple[int, ...]
    probability_columns: tuple[str, ...]
    candidates: tuple[str, ...]
    alignment_columns: tuple[str, ...]


def _class_suffixes(prob_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(column.removeprefix("prob_class_") for column in prob_columns)


def _label_values(prob_columns: Sequence[str]) -> tuple[int, ...]:
    suffixes = _class_suffixes(prob_columns)
    if not all(suffix.isdigit() for suffix in suffixes):
        return tuple(range(len(prob_columns)))
    return tuple(int(suffix) for suffix in suffixes)


def _class_columns_for_probabilities(frame: pd.DataFrame, prob_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(column for column in (f"class_{suffix}" for suffix in _class_suffixes(prob_columns)) if column in frame.columns)


def _label_present_mask(labels: Sequence[object] | np.ndarray | pd.Series) -> pd.Series:
    label_series = pd.Series(labels)
    return ~(label_series.isna() | label_series.astype(str).str.strip().eq(""))


def _integer_label_array(labels: Sequence[object] | np.ndarray | pd.Series, *, name: str) -> np.ndarray:
    """Return labels as integer values after rejecting lossy numeric casts."""

    numeric = pd.to_numeric(pd.Series(labels), errors="coerce")
    if numeric.isna().any():
        raise ValueError(f"{name} values must be numeric.")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} values must be finite.")
    rounded = np.rint(values)
    integer_like = np.isclose(values, rounded, rtol=0.0, atol=1.0e-12)
    if not bool(integer_like.all()):
        bad_rows = numeric.index[~integer_like].tolist()[:5]
        raise ValueError(f"{name} values must be integer-valued; fractional values at row(s) {bad_rows}.")
    return rounded.astype(int)


def _label_positions(
    labels: Sequence[object] | np.ndarray | pd.Series,
    label_values: Sequence[int],
    *,
    name: str = "true_label",
) -> np.ndarray:
    integer_labels = _integer_label_array(labels, name=name)
    label_to_position = {int(label): position for position, label in enumerate(label_values)}
    positions = np.full(len(integer_labels), -1, dtype=int)
    for row_index, label in enumerate(integer_labels):
        position = label_to_position.get(int(label))
        if position is not None:
            positions[row_index] = position
    if bool((positions < 0).any()):
        missing = sorted(set(int(label) for label in integer_labels if int(label) not in label_to_position))
        raise ValueError(f"{name} values must index probability labels {list(label_values)}; missing labels: {missing[:5]}")
    return positions


def _candidate_order(frame: pd.DataFrame, candidate_column: str) -> tuple[str, ...]:
    if candidate_column not in frame.columns:
        raise ValueError(f"Observation table is missing candidate column {candidate_column!r}.")
    return tuple(dict.fromkeys(frame[candidate_column].astype(str).tolist()))


def _normalize_candidates(candidates: Sequence[str] | None, frame: pd.DataFrame, candidate_column: str) -> tuple[str, ...]:
    values = tuple(str(candidate) for candidate in candidates) if candidates is not None else _candidate_order(frame, candidate_column)
    if not values:
        raise ValueError("At least one candidate/decoder is required for probability stacking.")
    return values


def _infer_alignment_columns(frame: pd.DataFrame, prob_columns: Sequence[str], candidate_column: str) -> tuple[str, ...]:
    class_columns = _class_columns_for_probabilities(frame, prob_columns)
    return tuple(column for column in (*_BASE_ALIGNMENT_COLUMNS, *class_columns) if column in frame.columns and column != candidate_column)


def _check_unique_alignment(subset: pd.DataFrame, keys: Sequence[str], candidate: str) -> None:
    if not keys:
        return
    duplicate_count = int(subset.duplicated(list(keys), keep=False).sum())
    if duplicate_count:
        examples = subset.loc[subset.duplicated(list(keys), keep=False), list(keys)].head(5).to_dict("records")
        raise ValueError(f"Candidate {candidate!r} has {duplicate_count} duplicate rows for the alignment keys. Examples: {examples}")


def _row_identity_columns(frame: pd.DataFrame, prob_columns: Sequence[str]) -> tuple[str, ...]:
    return tuple(column for column in (*_ROW_IDENTITY_COLUMNS, *_class_columns_for_probabilities(frame, prob_columns)) if column in frame.columns)


def _normalized_identity_values(values: Sequence[object] | np.ndarray | pd.Series, *, column: str) -> pd.Series:
    series = pd.Series(values).reset_index(drop=True)
    if column == "true_label":
        normalized = pd.Series(np.full(len(series), "", dtype=object))
        present = _label_present_mask(series)
        if bool(present.any()):
            labels = _integer_label_array(series.loc[present], name=column)
            normalized.iloc[np.flatnonzero(present.to_numpy())] = [str(int(label)) for label in labels]
        return normalized
    return series.where(~series.isna(), "").astype(str).str.strip()


def _raise_identity_mismatch(
    *,
    column: str,
    reference_values: pd.Series,
    candidate_values: pd.Series,
    candidate: str,
    reference_candidate: str,
) -> None:
    mismatched = reference_values != candidate_values
    if not bool(mismatched.any()):
        return
    examples = [
        {
            "row": int(index),
            reference_candidate: reference_values.iloc[index],
            candidate: candidate_values.iloc[index],
        }
        for index in np.flatnonzero(mismatched.to_numpy())[:5]
    ]
    raise ValueError(
        f"Candidate {candidate!r} has inconsistent {column!r} values relative to "
        f"{reference_candidate!r}. Examples: {examples}"
    )


def _renormalize_probabilities(values: np.ndarray, *, min_probability: float = DEFAULT_MIN_PROBABILITY) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Probability values must be a two-dimensional matrix.")
    if min_probability <= 0.0 or min_probability >= 1.0:
        raise ValueError("min_probability must lie in (0, 1).")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probability values must be finite.")
    if np.any(probabilities < 0.0):
        raise ValueError("Probability values must be non-negative.")
    probabilities = np.clip(probabilities, float(min_probability), 1.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0) or not np.isfinite(row_sums).all():
        raise ValueError("Probability rows must have positive finite sums.")
    return probabilities / row_sums


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    if len(labels) == 0:
        return float("nan")
    effective_k = min(int(k), probabilities.shape[1])
    top_columns = np.argsort(probabilities, axis=1)[:, ::-1][:, :effective_k]
    return float(np.mean(np.any(top_columns == labels[:, None], axis=1)))


def _normalize_pooling(pooling: str, *, allow_auto: bool = True) -> str:
    value = str(pooling).strip().lower().replace("-", "_")
    aliases = {
        "arithmetic": "linear",
        "average": "linear",
        "mixture": "linear",
        "probability": "linear",
        "geometric": "log",
        "log_probability": "log",
        "log_prob": "log",
        "product": "log",
    }
    value = aliases.get(value, value)
    allowed = POOLING_MODES if allow_auto else POOLING_MODES - {"auto"}
    if value not in allowed:
        suffix = "" if allow_auto else " after resolving 'auto'"
        raise ValueError(f"Unknown pooling {pooling!r}{suffix}; choose one of {sorted(allowed)}.")
    return value


def align_probability_cube(
    observations: pd.DataFrame,
    *,
    candidate_column: str = DEFAULT_CANDIDATE_COLUMN,
    candidates: Sequence[str] | None = None,
    alignment_columns: Sequence[str] | None = None,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    require_labels: bool = True,
    label_name: str = "true_label",
) -> AlignedProbabilityCube:
    """Align candidate probability rows into a candidate × sample × class cube.

    The default alignment keys use the canonical NeuRepTrace observation columns
    but deliberately exclude the candidate column.  If no such columns are
    present, rows are aligned by order and all candidates must have the same
    number of rows.  Source OOF rows must be labeled for fitting.  Target rows
    may pass ``require_labels=False`` so source-fitted weights can be applied to
    unlabeled deployment streams.
    """

    prob_columns = probability_columns(observations)
    if not prob_columns:
        raise ValueError("Observation table must contain prob_class_* columns.")
    if candidate_column not in observations.columns:
        raise ValueError(f"Observation table is missing candidate column {candidate_column!r}.")
    if require_labels and "true_label" not in observations.columns:
        raise ValueError("Observation table must contain true_label for source-OOF stacking.")
    candidates = _normalize_candidates(candidates, observations, candidate_column)
    keys = tuple(alignment_columns) if alignment_columns is not None else _infer_alignment_columns(observations, prob_columns, candidate_column)

    subsets: dict[str, pd.DataFrame] = {}
    candidate_values = observations[candidate_column].astype(str)
    for candidate in candidates:
        subset = observations.loc[candidate_values == candidate].copy().reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"No observation rows found for candidate {candidate!r}.")
        _check_unique_alignment(subset, keys, candidate)
        subsets[candidate] = subset

    reference = subsets[candidates[0]].copy().reset_index(drop=True)
    identity_columns = _row_identity_columns(observations, prob_columns)
    matrices: list[np.ndarray] = []
    if keys:
        aligned = reference.loc[:, list(keys)].copy()
        for candidate in candidates:
            subset = subsets[candidate]
            checked_identity_columns = tuple(column for column in identity_columns if column not in keys and column in subset.columns and column in reference.columns)
            renamed_columns = {column: f"{column}__{candidate}" for column in (*prob_columns, *checked_identity_columns)}
            renamed = subset.loc[:, [*keys, *prob_columns, *checked_identity_columns]].rename(columns=renamed_columns)
            aligned_candidate = aligned.merge(renamed, on=list(keys), how="left", validate="one_to_one")
            candidate_prob_columns = [renamed_columns[column] for column in prob_columns]
            if aligned_candidate.loc[:, candidate_prob_columns].isna().any().any():
                examples = aligned_candidate.loc[aligned_candidate.loc[:, candidate_prob_columns].isna().any(axis=1), list(keys)].head(5).to_dict("records")
                raise ValueError(f"Candidate {candidate!r} does not align one-to-one with {candidates[0]!r}. Examples: {examples}")
            for column in checked_identity_columns:
                _raise_identity_mismatch(
                    column=column,
                    reference_values=_normalized_identity_values(reference[column], column=column),
                    candidate_values=_normalized_identity_values(aligned_candidate[renamed_columns[column]], column=column),
                    candidate=candidate,
                    reference_candidate=candidates[0],
                )
            matrices.append(_renormalize_probabilities(aligned_candidate.loc[:, candidate_prob_columns].to_numpy(dtype=float), min_probability=min_probability))
    else:
        n_rows = len(reference)
        for candidate, subset in subsets.items():
            if len(subset) != n_rows:
                raise ValueError("Cannot align candidates by row order because they have different row counts and no alignment columns were inferred.")
            for column in identity_columns:
                if column in reference.columns and column in subset.columns:
                    _raise_identity_mismatch(
                        column=column,
                        reference_values=_normalized_identity_values(reference[column], column=column),
                        candidate_values=_normalized_identity_values(subset[column], column=column),
                        candidate=candidate,
                        reference_candidate=candidates[0],
                    )
            matrices.append(_renormalize_probabilities(subset.loc[:, list(prob_columns)].to_numpy(dtype=float), min_probability=min_probability))

    label_values = _label_values(prob_columns)
    label_positions: np.ndarray | None = None
    if "true_label" in reference.columns:
        label_mask = _label_present_mask(reference["true_label"])
        if bool(label_mask.any()):
            if not bool(label_mask.all()):
                raise ValueError("true_label must be present for all rows when provided.")
            label_positions = _label_positions(reference["true_label"], label_values, name=label_name)
    if require_labels and label_positions is None:
        raise ValueError("Observation table must contain true_label for source-OOF stacking.")
    return AlignedProbabilityCube(
        base=reference,
        cube=np.stack(matrices, axis=0),
        label_positions=label_positions,
        label_values=label_values,
        probability_columns=prob_columns,
        candidates=candidates,
        alignment_columns=keys,
    )


def class_balanced_sample_weights(labels: Sequence[int] | np.ndarray, *, n_classes: int) -> np.ndarray:
    """Return inverse-frequency sample weights normalized to mean one."""

    labels = _integer_label_array(labels, name="labels")
    if labels.size == 0:
        raise ValueError("Need at least one source-OOF prediction row for stacking.")
    if int(n_classes) <= 0:
        raise ValueError("n_classes must be positive.")
    if np.any(labels < 0) or np.any(labels >= int(n_classes)):
        raise ValueError("labels must be integer class positions compatible with n_classes.")
    counts = np.bincount(labels, minlength=int(n_classes)).astype(float)
    weights = np.zeros(labels.shape[0], dtype=float)
    observed = counts[labels] > 0.0
    weights[observed] = 1.0 / counts[labels[observed]]
    mean_weight = float(weights.mean())
    if not np.isfinite(mean_weight) or mean_weight <= 0.0:
        raise ValueError("Cannot compute class-balanced stacking weights from the provided labels.")
    return weights / mean_weight


def fit_stacking_weights(
    probability_cube: np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_classes: int | None = None,
    pooling: str = DEFAULT_POOLING,
    max_iter: int = DEFAULT_MAX_ITER,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
) -> np.ndarray:
    """Fit non-negative candidate weights from source-only OOF probabilities.

    The objective is class-balanced log loss on source OOF rows.  Linear pooling
    fits the historical arithmetic probability mixture.  Log pooling fits a
    geometric probability pool, which rewards candidates that agree on the true
    class while suppressing candidate-specific false-positive classes.
    """

    pooling = _normalize_pooling(pooling, allow_auto=False)
    cube = np.asarray(probability_cube, dtype=float)
    labels = _integer_label_array(labels, name="labels")
    if cube.ndim != 3:
        raise ValueError("probability_cube must have shape (n_candidates, n_samples, n_classes).")
    n_candidates, n_samples, cube_classes = cube.shape
    if n_classes is None:
        n_classes = int(cube_classes)
    if n_candidates < 1 or n_samples != labels.shape[0] or cube_classes != int(n_classes):
        raise ValueError("probability_cube shape is inconsistent with labels or n_classes.")
    if np.any(labels < 0) or np.any(labels >= int(n_classes)):
        raise ValueError("labels must be integer class positions compatible with probability_cube.")
    if n_candidates == 1:
        return np.ones(1, dtype=float)
    if learning_rate <= 0.0 or not np.isfinite(float(learning_rate)):
        raise ValueError("learning_rate must be positive and finite.")

    cube = np.stack([_renormalize_probabilities(candidate, min_probability=min_probability) for candidate in cube], axis=0)
    true_probabilities = cube[:, np.arange(n_samples), labels]
    log_cube = np.log(cube)
    true_log_probabilities = log_cube[:, np.arange(n_samples), labels]
    sample_weights = class_balanced_sample_weights(labels, n_classes=int(n_classes))
    weights = np.full(n_candidates, 1.0 / float(n_candidates), dtype=float)

    for iteration in range(max(1, int(max_iter))):
        if pooling == "linear":
            denominator = np.clip(weights @ true_probabilities, float(min_probability), 1.0)
            gradient = -np.average(true_probabilities / denominator[None, :], axis=1, weights=sample_weights)
        else:
            pooled_log = np.tensordot(weights, log_cube, axes=(0, 0))
            pooled_log -= pooled_log.max(axis=1, keepdims=True)
            combined = np.exp(pooled_log)
            combined /= combined.sum(axis=1, keepdims=True)
            expected_log_probabilities = np.einsum("nc,inc->in", combined, log_cube)
            gradient = np.average(expected_log_probabilities - true_log_probabilities, axis=1, weights=sample_weights)
        gradient -= float(np.dot(gradient, weights))
        step = float(learning_rate) / np.sqrt(float(iteration + 1))
        updated = weights * np.exp(np.clip(-step * gradient, -50.0, 50.0))
        total = float(updated.sum())
        if not np.isfinite(total) or total <= 0.0:
            break
        weights = updated / total
    return weights / float(weights.sum())


def combine_probability_cube(
    probability_cube: np.ndarray,
    weights: Sequence[float],
    *,
    pooling: str = DEFAULT_POOLING,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
) -> np.ndarray:
    """Return the weighted, row-normalized probability matrix for a candidate cube."""

    pooling = _normalize_pooling(pooling, allow_auto=False)
    cube = np.asarray(probability_cube, dtype=float)
    weights_array = np.asarray(weights, dtype=float).reshape(-1)
    if cube.ndim != 3:
        raise ValueError("probability_cube must have shape (n_candidates, n_samples, n_classes).")
    if weights_array.shape[0] != cube.shape[0]:
        raise ValueError("weights must contain one value per candidate.")
    if not np.isfinite(weights_array).all() or (weights_array < 0.0).any() or float(weights_array.sum()) <= 0.0:
        raise ValueError("weights must be finite non-negative values with positive sum.")
    weights_array = weights_array / float(weights_array.sum())
    normalized = np.stack([_renormalize_probabilities(candidate, min_probability=min_probability) for candidate in cube], axis=0)
    if pooling == "linear":
        pooled = np.tensordot(weights_array, normalized, axes=(0, 0))
    else:
        pooled_log = np.tensordot(weights_array, np.log(normalized), axes=(0, 0))
        pooled_log -= pooled_log.max(axis=1, keepdims=True)
        pooled = np.exp(pooled_log)
    return _renormalize_probabilities(pooled, min_probability=min_probability)


def _candidate_balanced_scores(probability_cube: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.asarray([balanced_accuracy_score(labels, candidate.argmax(axis=1)) for candidate in probability_cube], dtype=float)


def _fixed_pooling_fit(
    cube: np.ndarray,
    labels: np.ndarray,
    *,
    candidates: Sequence[str],
    weighting: str,
    pooling: str,
    temperature: float | None,
    max_iter: int,
    learning_rate: float,
    min_probability: float,
) -> SourceOOFStackingFit:
    """Fit weights for one concrete pooling rule."""

    n_candidates, _, n_classes = cube.shape
    if weighting == "uniform" or n_candidates == 1:
        weights = np.full(n_candidates, 1.0 / float(n_candidates), dtype=float)
        used_temperature = None
    elif weighting == "softmax":
        used_temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
        if not np.isfinite(used_temperature) or used_temperature <= 0.0:
            raise ValueError("temperature must be positive and finite for softmax weighting.")
        scores = _candidate_balanced_scores(cube, labels)
        weights = np.exp(np.clip((scores - float(scores.max())) / used_temperature, -60.0, 0.0))
        weights = weights / float(weights.sum())
    else:
        used_temperature = None
        weights = fit_stacking_weights(
            cube,
            labels,
            n_classes=n_classes,
            pooling=pooling,
            max_iter=max_iter,
            learning_rate=learning_rate,
            min_probability=min_probability,
        )

    combined = combine_probability_cube(cube, weights, pooling=pooling, min_probability=min_probability)
    try:
        source_log_loss = float(log_loss(labels, combined, labels=list(range(n_classes))))
    except ValueError:
        source_log_loss = float("nan")
    return SourceOOFStackingFit(
        candidates=tuple(str(candidate) for candidate in candidates),
        weights=tuple(float(weight) for weight in weights),
        weighting=weighting,
        pooling=pooling,
        temperature=used_temperature,
        source_oof_balanced_accuracy=float(balanced_accuracy_score(labels, combined.argmax(axis=1))),
        source_oof_log_loss=source_log_loss,
    )


def _source_oof_fit_key(fit: SourceOOFStackingFit) -> tuple[float, float, int]:
    loss = fit.source_oof_log_loss
    finite_loss = loss if np.isfinite(loss) else float("inf")
    pooling_tiebreak = 0 if fit.pooling == "linear" else 1
    return finite_loss, -fit.source_oof_balanced_accuracy, pooling_tiebreak


def fit_source_oof_stacking(
    source_probability_cube: np.ndarray,
    source_labels: Sequence[int] | np.ndarray,
    *,
    candidates: Sequence[str],
    weighting: str = DEFAULT_WEIGHTING,
    pooling: str = DEFAULT_POOLING,
    temperature: float | None = DEFAULT_TEMPERATURE,
    max_iter: int = DEFAULT_MAX_ITER,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
) -> SourceOOFStackingFit:
    """Fit source-only ensemble weights and report source-OOF diagnostics."""

    weighting = str(weighting).strip().lower().replace("-", "_")
    if weighting not in WEIGHTING_MODES:
        raise ValueError(f"Unknown weighting {weighting!r}; choose one of {sorted(WEIGHTING_MODES)}.")
    pooling = _normalize_pooling(pooling)
    cube = np.asarray(source_probability_cube, dtype=float)
    labels = _integer_label_array(source_labels, name="source_labels")
    if cube.ndim != 3:
        raise ValueError("source_probability_cube must have shape (n_candidates, n_samples, n_classes).")
    n_candidates, n_samples, n_classes = cube.shape
    if len(candidates) != n_candidates:
        raise ValueError("candidates must contain one name per probability-cube candidate.")
    if labels.shape[0] != n_samples:
        raise ValueError("source_labels must contain one label per probability row.")
    if np.any(labels < 0) or np.any(labels >= int(n_classes)):
        raise ValueError("source_labels must be integer class positions compatible with source_probability_cube.")

    pooling_candidates = ("linear", "log") if pooling == "auto" else (pooling,)
    fits = [
        _fixed_pooling_fit(
            cube,
            labels,
            candidates=candidates,
            weighting=weighting,
            pooling=pooling_candidate,
            temperature=temperature,
            max_iter=max_iter,
            learning_rate=learning_rate,
            min_probability=min_probability,
        )
        for pooling_candidate in pooling_candidates
    ]
    return min(fits, key=_source_oof_fit_key)


def _predicted_classes(base: pd.DataFrame, predicted_labels: np.ndarray) -> list[str]:
    classes: list[str] = []
    for row_index, predicted_label in enumerate(predicted_labels):
        class_column = f"class_{predicted_label}"
        if class_column in base.columns:
            classes.append(str(base.iloc[row_index][class_column]))
        else:
            classes.append(str(predicted_label))
    return classes


def stack_probability_observations(
    source_oof_observations: pd.DataFrame,
    target_observations: pd.DataFrame,
    *,
    candidate_column: str = DEFAULT_CANDIDATE_COLUMN,
    candidates: Sequence[str] | None = None,
    alignment_columns: Sequence[str] | None = None,
    weighting: str = DEFAULT_WEIGHTING,
    pooling: str = DEFAULT_POOLING,
    temperature: float | None = DEFAULT_TEMPERATURE,
    max_iter: int = DEFAULT_MAX_ITER,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    output_decoder: str = DEFAULT_OUTPUT_DECODER,
    output_emission_mode: str = DEFAULT_OUTPUT_EMISSION_MODE,
) -> pd.DataFrame:
    """Fit source-OOF stacking weights and apply them to target observations.

    Source OOF observations must include numeric ``true_label`` values because
    those labels define the leakage-safe stacking objective.  Target labels are
    optional: when omitted, the returned table still contains predictions,
    confidences, stacked probabilities, and source-fit diagnostics, while
    label-dependent reporting fields are left blank.
    """

    source = align_probability_cube(
        source_oof_observations,
        candidate_column=candidate_column,
        candidates=candidates,
        alignment_columns=alignment_columns,
        min_probability=min_probability,
    )
    if source.label_positions is None:
        raise ValueError("Source OOF observations must contain true_label for stacking.")
    target = align_probability_cube(
        target_observations,
        candidate_column=candidate_column,
        candidates=source.candidates,
        alignment_columns=alignment_columns,
        min_probability=min_probability,
        require_labels=False,
        label_name="target true_label",
    )
    if target.probability_columns != source.probability_columns:
        raise ValueError("Source-OOF and target observations must use the same prob_class_* columns.")

    fit = fit_source_oof_stacking(
        source.cube,
        source.label_positions,
        candidates=source.candidates,
        weighting=weighting,
        pooling=pooling,
        temperature=temperature,
        max_iter=max_iter,
        learning_rate=learning_rate,
        min_probability=min_probability,
    )
    probabilities = combine_probability_cube(target.cube, fit.weights, pooling=fit.pooling, min_probability=min_probability)
    output = target.base.copy()
    label_values = target.label_values
    predicted_positions = probabilities.argmax(axis=1)
    predicted_labels = np.asarray([label_values[position] for position in predicted_positions], dtype=int)

    true_probabilities: np.ndarray
    correctness: np.ndarray
    if "true_label" in output.columns and bool(_label_present_mask(output["true_label"]).any()):
        label_mask = _label_present_mask(output["true_label"])
        if not bool(label_mask.all()):
            raise ValueError("target true_label must be present for all rows when provided.")
        true_label_values = _integer_label_array(output["true_label"], name="target true_label")
        label_to_position = {label: position for position, label in enumerate(label_values)}
        true_probabilities = np.full(len(output), np.nan, dtype=float)
        for row_index, true_label in enumerate(true_label_values):
            position = label_to_position.get(int(true_label))
            if position is None:
                raise ValueError(f"target true_label values must index probability labels {list(label_values)}; missing label: {int(true_label)}")
            true_probabilities[row_index] = probabilities[row_index, position]
        correctness = predicted_labels == true_label_values
    else:
        true_probabilities = np.full(len(output), "", dtype=object)
        correctness = np.full(len(output), "", dtype=object)

    for column_index, column in enumerate(target.probability_columns):
        output[column] = probabilities[:, column_index]
    output[candidate_column] = output_decoder
    output["decoder"] = output_decoder
    output["backend"] = "source_oof_stacking"
    output["emission_mode"] = output_emission_mode
    output["predicted_label"] = predicted_labels
    output["predicted_class"] = _predicted_classes(output, predicted_labels)
    output["probability_true_class"] = true_probabilities
    output["confidence"] = probabilities.max(axis=1)
    output["is_correct"] = correctness
    output["calibration_fold"] = "source_oof"
    output["source_oof_candidates"] = "|".join(fit.candidates)
    output["source_oof_weights"] = "|".join(f"{weight:.12g}" for weight in fit.weights)
    output["source_oof_weighting"] = fit.weighting
    output["source_oof_pooling"] = fit.pooling
    output["source_oof_temperature"] = "" if fit.temperature is None else fit.temperature
    output["source_oof_balanced_accuracy"] = fit.source_oof_balanced_accuracy
    output["source_oof_log_loss"] = fit.source_oof_log_loss
    output["source_oof_alignment_columns"] = "|".join(target.alignment_columns)
    output["model_hash"] = stable_hash(
        {
            "backend": "source_oof_stacking",
            "candidate_column": candidate_column,
            "candidates": list(fit.candidates),
            "weights": list(fit.weights),
            "weighting": fit.weighting,
            "pooling": fit.pooling,
            "temperature": fit.temperature,
            "alignment_columns": list(target.alignment_columns),
            "min_probability": min_probability,
        }
    )
    return ProbabilityObservationTable(output).standardized(defaults={"backend": "source_oof_stacking", "decoder": output_decoder, "emission_mode": output_emission_mode}).frame


def summarize_stacked_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize stacked observation rows by time/fold with standard metrics."""

    prob_columns = probability_columns(observations)
    if "true_label" not in observations.columns or not prob_columns:
        raise ValueError("Stacked observations must contain true_label and prob_class_* columns.")
    label_values = _label_values(prob_columns)
    label_value_set = set(label_values)
    group_columns = [column for column in _METRIC_GROUP_COLUMNS if column in observations.columns]
    rows: list[dict[str, object]] = []
    for group_key, group in observations.groupby(group_columns, dropna=False, sort=True):
        if len(group_columns) == 1 and not isinstance(group_key, tuple):
            group_key = (group_key,)
        probabilities = group.loc[:, list(prob_columns)].to_numpy(dtype=float)
        true_label_values = _integer_label_array(group["true_label"], name="true_label")
        missing_labels = sorted(set(int(label) for label in true_label_values if int(label) not in label_value_set))
        if missing_labels:
            raise ValueError(f"true_label values must index probability labels {list(label_values)}; missing labels: {missing_labels[:5]}")
        predicted_label_values = np.asarray([label_values[position] for position in probabilities.argmax(axis=1)], dtype=int)
        row = dict(zip(group_columns, group_key, strict=True))
        row.update(
            {
                "accuracy": float(accuracy_score(true_label_values, predicted_label_values)),
                "balanced_accuracy": float(balanced_accuracy_score(true_label_values, predicted_label_values)),
                "top2_accuracy": _top_k_accuracy(probabilities, true_label_values, k=2),
                "top3_accuracy": _top_k_accuracy(probabilities, true_label_values, k=3),
                "log_loss": float(log_loss(true_label_values, probabilities, labels=list(label_values))),
                "brier": float(brier_score_multiclass(probabilities, true_label_values)),
                "ece": float(expected_calibration_error(probabilities, true_label_values)),
                "n_test": int(len(group)),
                "n_classes": int(len(prob_columns)),
                "source_oof_candidates": str(group.iloc[0].get("source_oof_candidates", "")),
                "source_oof_weights": str(group.iloc[0].get("source_oof_weights", "")),
                "source_oof_weighting": str(group.iloc[0].get("source_oof_weighting", "")),
                "source_oof_pooling": str(group.iloc[0].get("source_oof_pooling", "")),
                "source_oof_balanced_accuracy": group.iloc[0].get("source_oof_balanced_accuracy", ""),
                "source_oof_log_loss": group.iloc[0].get("source_oof_log_loss", ""),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _read_csv_inputs(paths: Sequence[str | Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for pattern in paths:
        matches = sorted(glob.glob(str(pattern)))
        if not matches and Path(pattern).exists():
            matches = [str(pattern)]
        if not matches:
            raise FileNotFoundError(f"No CSV files match {pattern!r}.")
        frames.extend(pd.read_csv(path) for path in matches)
    if not frames:
        raise ValueError("No CSV inputs were provided.")
    return pd.concat(frames, ignore_index=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point for source-OOF probability stacking."""

    parser = argparse.ArgumentParser(description="Fit a leakage-safe source-OOF probability stacker and apply it to target observation rows.")
    parser.add_argument("--source-oof", nargs="+", required=True, help="Source out-of-fold observation CSVs or glob patterns used to fit weights.")
    parser.add_argument(
        "--target",
        nargs="+",
        required=True,
        help="Target observation CSVs or glob patterns to ensemble with fitted source weights; true_label is optional unless --metrics-out is requested.",
    )
    parser.add_argument("--out", type=Path, required=True, help="CSV path for stacked target observations.")
    parser.add_argument("--metrics-out", type=Path, help="Optional CSV path for grouped metrics computed from the stacked observations.")
    parser.add_argument("--candidate-column", default=DEFAULT_CANDIDATE_COLUMN, help="Column identifying base candidates/decoders. Defaults to decoder.")
    parser.add_argument("--candidate", action="append", dest="candidates", help="Candidate/decoder to include. May be repeated; defaults to order in source rows.")
    parser.add_argument(
        "--alignment-column",
        action="append",
        dest="alignment_columns",
        help="Column used to align candidates. May be repeated; defaults to canonical observation keys.",
    )
    parser.add_argument("--weighting", choices=sorted(WEIGHTING_MODES), default=DEFAULT_WEIGHTING)
    parser.add_argument(
        "--pooling",
        choices=sorted(POOLING_MODES),
        default=DEFAULT_POOLING,
        help="Probability pooling rule. 'linear' preserves the historical arithmetic mixture; 'log' uses geometric pooling; 'auto' selects by source-OOF log loss.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Softmax weighting temperature.")
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER, help="Projected-gradient iterations for stacked weighting.")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--min-probability", type=float, default=DEFAULT_MIN_PROBABILITY)
    parser.add_argument("--output-decoder", default=DEFAULT_OUTPUT_DECODER)
    parser.add_argument("--output-emission-mode", default=DEFAULT_OUTPUT_EMISSION_MODE)
    args = parser.parse_args(argv)

    try:
        source_oof = _read_csv_inputs(args.source_oof)
        target = _read_csv_inputs(args.target)
        stacked = stack_probability_observations(
            source_oof,
            target,
            candidate_column=args.candidate_column,
            candidates=args.candidates,
            alignment_columns=args.alignment_columns,
            weighting=args.weighting,
            pooling=args.pooling,
            temperature=args.temperature,
            max_iter=args.max_iter,
            learning_rate=args.learning_rate,
            min_probability=args.min_probability,
            output_decoder=args.output_decoder,
            output_emission_mode=args.output_emission_mode,
        )
        ProbabilityObservationTable(stacked).to_csv(args.out)
        print(f"Wrote stacked observations: {args.out}")
        if args.metrics_out is not None:
            metrics = summarize_stacked_metrics(stacked)
            args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
            metrics.to_csv(args.metrics_out, index=False)
            print(f"Wrote stacked metrics: {args.metrics_out}")
    except Exception as exc:
        print(f"Probability stacking failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Leakage-free structured decisions for repeated event predictions.

The functions in this module operate only on model probabilities and task
constraints.  They never accept evaluation labels.  This makes them suitable
for Protocol-3 target-calibration experiments where the scored target labels
must stay outside the fitting and decision APIs.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

_EPSILON = np.finfo(np.float64).tiny


@dataclass(frozen=True, slots=True)
class StructuredPredictionResult:
    """Predictions produced by a trial-level structural constraint."""

    predictions: np.ndarray
    group_ids: tuple[Hashable, ...]
    selected_structures: tuple[Hashable, ...]
    log_scores: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SequenceTemplateModel:
    """Deterministic class sequences learned from calibration rows."""

    template_ids: tuple[Hashable, ...]
    positions: tuple[Hashable, ...]
    class_labels: tuple[Hashable, ...]
    template_class_indices: np.ndarray
    calibration_rows: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = np.asarray(self.template_class_indices, dtype=int)
        expected = (len(self.template_ids), len(self.positions))
        if matrix.shape != expected:
            raise ValueError(f"template_class_indices must have shape {expected}, got {matrix.shape}.")
        if matrix.size and (np.any(matrix < 0) or np.any(matrix >= len(self.class_labels))):
            raise ValueError("template_class_indices contains an out-of-range class index.")
        immutable = matrix.copy()
        immutable.setflags(write=False)
        object.__setattr__(self, "template_class_indices", immutable)


def _atomic_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.ndim == 1:
        materialized = values.tolist()
    else:
        try:
            materialized = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must be a one-dimensional sequence.") from exc
    result = np.empty(len(materialized), dtype=object)
    result[:] = materialized
    return result


def _hashable_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    vector = _atomic_vector(values, name=name)
    for index, value in enumerate(vector.tolist()):
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError(f"{name}[{index}] must be hashable.") from exc
    return vector


def _stable_unique(values: np.ndarray) -> tuple[Hashable, ...]:
    try:
        return tuple(dict.fromkeys(values.tolist()))
    except TypeError as exc:
        raise ValueError("Structured identifiers and class labels must be hashable.") from exc


def _equal_mask(values: np.ndarray, target: Hashable) -> np.ndarray:
    return np.fromiter((value == target for value in values.tolist()), dtype=bool, count=values.shape[0])


def _probability_matrix(probabilities: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    raw = np.asarray(probabilities)
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise ValueError("probabilities must be real-valued.")
    try:
        matrix = np.asarray(probabilities, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("probabilities must be a numeric two-dimensional matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] < 2:
        raise ValueError("probabilities must have shape (n_rows, n_classes) with at least two classes.")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0.0):
        raise ValueError("probabilities must contain finite non-negative values.")
    row_max = np.max(matrix, axis=1, keepdims=True)
    if np.any(row_max <= 0.0):
        raise ValueError("Every probability row must contain positive mass.")
    scaled = matrix / row_max
    return scaled / np.sum(scaled, axis=1, keepdims=True)


def _class_labels(class_labels: Sequence[Hashable] | np.ndarray | None, n_classes: int) -> tuple[Hashable, ...]:
    if class_labels is None:
        return tuple(range(n_classes))
    labels = _hashable_vector(class_labels, name="class_labels")
    if labels.shape[0] != n_classes:
        raise ValueError(f"class_labels must contain {n_classes} entries.")
    unique = _stable_unique(labels)
    if len(unique) != n_classes:
        raise ValueError("class_labels must contain unique values.")
    return tuple(labels.tolist())


def decode_unique_class_assignments(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    group_ids: Sequence[Hashable] | np.ndarray,
    *,
    class_labels: Sequence[Hashable] | np.ndarray | None = None,
) -> StructuredPredictionResult:
    """Assign every class exactly once inside each complete event group.

    The maximum-log-probability assignment is solved independently for each
    group with the Hungarian algorithm.  A group must contain exactly one row
    per model class.  The returned predictions follow the original row order.
    """

    matrix = _probability_matrix(probabilities)
    groups = _hashable_vector(group_ids, name="group_ids")
    if groups.shape[0] != matrix.shape[0]:
        raise ValueError("group_ids must contain one value per probability row.")
    labels = _class_labels(class_labels, matrix.shape[1])
    ordered_groups = _stable_unique(groups)
    predictions = np.empty(matrix.shape[0], dtype=object)
    structures: list[Hashable] = []
    scores: list[float] = []
    for group_id in ordered_groups:
        indices = np.flatnonzero(_equal_mask(groups, group_id))
        if indices.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Group {group_id!r} contains {indices.shape[0]} rows; "
                f"unique-class decoding requires exactly {matrix.shape[1]}."
            )
        costs = -np.log(np.clip(matrix[indices], _EPSILON, 1.0))
        row_indices, class_indices = linear_sum_assignment(costs)
        assignment = np.empty(indices.shape[0], dtype=int)
        assignment[row_indices] = class_indices
        predictions[indices] = [labels[index] for index in assignment.tolist()]
        structures.append(tuple(int(index) for index in assignment.tolist()))
        scores.append(float(np.log(np.clip(matrix[indices, assignment], _EPSILON, 1.0)).sum()))
    return StructuredPredictionResult(
        predictions=predictions,
        group_ids=ordered_groups,
        selected_structures=tuple(structures),
        log_scores=np.asarray(scores, dtype=float),
        metadata={
            "constraint": "one_occurrence_per_class",
            "n_groups": len(ordered_groups),
            "n_classes": matrix.shape[1],
            "uses_evaluation_labels": False,
        },
    )


def learn_sequence_templates(
    calibration_labels: Sequence[Hashable] | np.ndarray,
    calibration_template_ids: Sequence[Hashable] | np.ndarray,
    calibration_positions: Sequence[Hashable] | np.ndarray,
    *,
    class_labels: Sequence[Hashable] | np.ndarray | None = None,
    require_permutations: bool = True,
) -> SequenceTemplateModel:
    """Learn deterministic position-by-class templates from calibration rows.

    Repeated calibration trials are allowed, but every observed template and
    position pair must carry one unambiguous class label.  No evaluation row is
    accepted by this API.
    """

    labels = _hashable_vector(calibration_labels, name="calibration_labels")
    template_ids = _hashable_vector(calibration_template_ids, name="calibration_template_ids")
    positions = _hashable_vector(calibration_positions, name="calibration_positions")
    if labels.shape[0] == 0 or template_ids.shape[0] != labels.shape[0] or positions.shape[0] != labels.shape[0]:
        raise ValueError("Calibration labels, template IDs, and positions must be non-empty and row-aligned.")
    ordered_templates = _stable_unique(template_ids)
    ordered_positions = _stable_unique(positions)
    if class_labels is None:
        ordered_classes = _stable_unique(labels)
    else:
        class_vector = _hashable_vector(class_labels, name="class_labels")
        ordered_classes = _stable_unique(class_vector)
        if len(ordered_classes) != class_vector.shape[0]:
            raise ValueError("class_labels must contain unique values.")
    class_to_index = {label: index for index, label in enumerate(ordered_classes)}
    unknown = [label for label in labels.tolist() if label not in class_to_index]
    if unknown:
        raise ValueError(f"calibration_labels contains a class absent from class_labels: {unknown[0]!r}.")

    matrix = np.full((len(ordered_templates), len(ordered_positions)), -1, dtype=int)
    for template_index, template_id in enumerate(ordered_templates):
        template_mask = _equal_mask(template_ids, template_id)
        for position_index, position in enumerate(ordered_positions):
            values = _stable_unique(labels[template_mask & _equal_mask(positions, position)])
            if not values:
                raise ValueError(f"Template {template_id!r} has no calibration row at position {position!r}.")
            if len(values) != 1:
                raise ValueError(f"Template {template_id!r}, position {position!r} has inconsistent calibration labels: {values!r}.")
            matrix[template_index, position_index] = class_to_index[values[0]]
        if require_permutations and len(set(matrix[template_index].tolist())) != matrix.shape[1]:
            raise ValueError(f"Template {template_id!r} is not a one-occurrence-per-class permutation.")
    if require_permutations and matrix.shape[1] != len(ordered_classes):
        raise ValueError("Permutation templates require one position per class.")
    return SequenceTemplateModel(
        template_ids=ordered_templates,
        positions=ordered_positions,
        class_labels=ordered_classes,
        template_class_indices=matrix,
        calibration_rows=int(labels.shape[0]),
        metadata={
            "learned_from": "target_calibration_rows",
            "requires_permutations": bool(require_permutations),
            "uses_evaluation_labels": False,
        },
    )


def decode_sequence_templates(
    probabilities: Sequence[Sequence[float]] | np.ndarray,
    group_ids: Sequence[Hashable] | np.ndarray,
    positions: Sequence[Hashable] | np.ndarray,
    templates: SequenceTemplateModel,
) -> StructuredPredictionResult:
    """Choose the highest-scoring learned sequence template for each group."""

    matrix = _probability_matrix(probabilities)
    groups = _hashable_vector(group_ids, name="group_ids")
    row_positions = _hashable_vector(positions, name="positions")
    if groups.shape[0] != matrix.shape[0] or row_positions.shape[0] != matrix.shape[0]:
        raise ValueError("group_ids and positions must contain one value per probability row.")
    if matrix.shape[1] != len(templates.class_labels):
        raise ValueError("Probability columns must match templates.class_labels.")
    position_to_index = {position: index for index, position in enumerate(templates.positions)}
    ordered_groups = _stable_unique(groups)
    predictions = np.empty(matrix.shape[0], dtype=object)
    selected_templates: list[Hashable] = []
    selected_scores: list[float] = []
    for group_id in ordered_groups:
        indices = np.flatnonzero(_equal_mask(groups, group_id))
        if indices.shape[0] != len(templates.positions):
            raise ValueError(
                f"Group {group_id!r} contains {indices.shape[0]} rows; "
                f"template decoding requires {len(templates.positions)}."
            )
        try:
            position_indices = np.asarray([position_to_index[position] for position in row_positions[indices].tolist()], dtype=int)
        except KeyError as exc:
            raise ValueError(f"Group {group_id!r} contains unknown position {exc.args[0]!r}.") from exc
        if len(set(position_indices.tolist())) != len(templates.positions):
            raise ValueError(f"Group {group_id!r} must contain each template position exactly once.")
        row_for_position = np.empty(len(templates.positions), dtype=int)
        row_for_position[position_indices] = indices
        ordered_probabilities = matrix[row_for_position]
        template_scores = np.empty(len(templates.template_ids), dtype=float)
        for template_index, class_indices in enumerate(templates.template_class_indices):
            template_scores[template_index] = float(
                np.log(np.clip(ordered_probabilities[np.arange(len(templates.positions)), class_indices], _EPSILON, 1.0)).sum()
            )
        best_index = int(np.argmax(template_scores))
        best_classes = templates.template_class_indices[best_index]
        for position_index, row_index in enumerate(row_for_position.tolist()):
            predictions[row_index] = templates.class_labels[int(best_classes[position_index])]
        selected_templates.append(templates.template_ids[best_index])
        selected_scores.append(float(template_scores[best_index]))
    return StructuredPredictionResult(
        predictions=predictions,
        group_ids=ordered_groups,
        selected_structures=tuple(selected_templates),
        log_scores=np.asarray(selected_scores, dtype=float),
        metadata={
            "constraint": "calibration_sequence_templates",
            "n_groups": len(ordered_groups),
            "n_templates": len(templates.template_ids),
            "n_positions": len(templates.positions),
            "template_calibration_rows": templates.calibration_rows,
            "uses_evaluation_labels": False,
        },
    )

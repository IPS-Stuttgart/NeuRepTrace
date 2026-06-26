"""Target-subject projection helpers for NeuRepTrace M-CCA models.

These utilities cover the common held-out-subject calibration case: a multiset
CCA model is fitted on training subjects, a small row-aligned calibration matrix
is available for a target subject, and the target features should be projected
into the learned M-CCA component space without adding dataset-specific loading or
windowing assumptions to NeuRepTrace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from neureptrace.decoding.mcca import (
    MCCAModel,
    _class_mean_matrix,
    _class_repetition_matrix,
    _contains_label,
    _feature_matrix,
    _normalize_sample_mode,
    _ordered_unique_labels,
)
from neureptrace.decoding.sampling import DEFAULT_CLASS_LIMIT_SEED, DEFAULT_CLASS_LIMIT_SELECTION


@dataclass(frozen=True)
class TargetMCCAProjection:
    """Linear map from a calibrated held-out subject into an M-CCA space.

    The projection is fitted from row-aligned target calibration features to an
    existing M-CCA component template.  Use :meth:`transform` when the scored
    features have the same feature layout as the calibration rows.  Projects
    with dataset-specific window or channel adapters can apply ``projection``
    and ``feature_mean`` themselves, then call :meth:`add_template_mean` on the
    projected rows.
    """

    feature_mean: np.ndarray
    template_mean: np.ndarray
    projection: np.ndarray
    regularization: float
    n_alignment_rows: int

    def transform(self, features: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        """Transform target-subject rows into the fitted component space."""

        matrix = _feature_matrix(features, name="features")
        if matrix.shape[1] != self.projection.shape[0]:
            raise ValueError(
                "features column count does not match the target M-CCA projection: "
                f"{matrix.shape[1]} != {self.projection.shape[0]}."
            )
        return self.add_template_mean((matrix - self.feature_mean) @ self.projection, strict=True)

    def add_template_mean(self, transformed: Sequence[Sequence[float]] | np.ndarray, *, strict: bool = False) -> np.ndarray:
        """Add the fitted template mean to already projected target rows.

        When ``transformed`` contains repeated component blocks, the template
        mean is tiled across blocks.  With ``strict=False`` incompatible column
        counts are returned unchanged for compatibility with dataset-specific
        projection adapters.
        """

        return _add_template_mean(transformed, self.template_mean, strict=strict)


def class_alignment_matrix(
    features: Sequence[Sequence[float]] | np.ndarray,
    labels: Sequence | np.ndarray,
    *,
    classes: Sequence | np.ndarray | None = None,
    sample_mode: str = "class_mean",
    n_repetitions_per_class: int | None = None,
    repetition_selection: str = DEFAULT_CLASS_LIMIT_SELECTION,
    repetition_seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED,
    selected_offsets_by_class: Mapping[int, Sequence[int] | np.ndarray] | None = None,
) -> np.ndarray:
    """Build one subject's class-aligned feature matrix.

    ``classes`` fixes the row order.  This is useful for held-out target
    calibration where the target rows must match an already fitted class
    alignment or M-CCA component template. For ``class_repetition``, use the
    same ``repetition_selection`` and ``repetition_seed`` as the fitted template.
    Passing ``selected_offsets_by_class`` is safer when the fitted training
    alignment stored common within-class offsets and the target subject has a
    different number of available repetitions.
    """

    sample_mode = _normalize_sample_mode(sample_mode)
    matrix = _feature_matrix(features, name="features")
    vector = _label_vector(labels, expected_length=matrix.shape[0], name="labels")
    if classes is None:
        class_order = _ordered_unique_labels(vector)
    else:
        class_order = _label_vector(classes, expected_length=None, name="classes")
    _check_requested_classes(vector, class_order)

    if sample_mode == "class_mean":
        return _class_mean_matrix(matrix, vector, class_order)

    selected_offsets: dict[int, np.ndarray] | None = None
    if selected_offsets_by_class is not None:
        selected_offsets, selected_repetitions = _normalize_selected_offsets_by_class(
            selected_offsets_by_class,
            labels=vector,
            classes=class_order,
        )
        if n_repetitions_per_class is None:
            n_repetitions_per_class = selected_repetitions
        elif int(n_repetitions_per_class) != selected_repetitions:
            raise ValueError(
                "n_repetitions_per_class must match selected_offsets_by_class length: "
                f"{n_repetitions_per_class} != {selected_repetitions}."
            )
    elif n_repetitions_per_class is None:
        n_repetitions_per_class = _minimum_class_count(vector, class_order)
    repetitions = int(n_repetitions_per_class)
    if repetitions < 1:
        raise ValueError("n_repetitions_per_class must be positive or None.")
    return _class_repetition_matrix(
        matrix,
        vector,
        class_order,
        repetitions,
        selection=repetition_selection,
        seed=repetition_seed,
        selected_offsets_by_class=selected_offsets,
    )


def _normalize_selected_offsets_by_class(
    selected_offsets_by_class: Mapping[int, Sequence[int] | np.ndarray],
    *,
    labels: np.ndarray,
    classes: np.ndarray,
) -> tuple[dict[int, np.ndarray], int]:
    """Validate source-template repetition offsets for a target subject."""

    normalized: dict[int, np.ndarray] = {}
    sizes: list[int] = []
    for class_position, class_label in enumerate(classes):
        try:
            offsets = _normalize_selected_offset_vector(selected_offsets_by_class[class_position])
        except KeyError as exc:
            raise ValueError(f"selected_offsets_by_class is missing class position {class_position}.") from exc
        class_count = _count_label(labels, class_label)
        if int(np.min(offsets)) < 0 or int(np.max(offsets)) >= class_count:
            raise ValueError(
                f"selected offsets for class {class_label!r} are outside the target subject's "
                f"available repetitions: max offset {int(np.max(offsets))}, count {class_count}."
            )
        normalized[class_position] = offsets
        sizes.append(int(offsets.size))
    unique_sizes = set(sizes)
    if len(unique_sizes) != 1:
        raise ValueError(f"selected_offsets_by_class entries must have equal lengths, got {sizes}.")
    return normalized, int(sizes[0])


def _normalize_selected_offset_vector(offsets: Sequence[int] | np.ndarray) -> np.ndarray:
    raw_offsets = np.asarray(offsets, dtype=object)
    if raw_offsets.ndim != 1:
        raise ValueError("selected_offsets_by_class entries must be one-dimensional.")
    if raw_offsets.size < 1:
        raise ValueError("selected_offsets_by_class entries must not be empty.")
    if any(isinstance(value, (bool, np.bool_)) for value in raw_offsets.tolist()):
        raise ValueError("selected_offsets_by_class entries must contain integer offsets, not booleans or boolean masks.")
    try:
        numeric = np.asarray(raw_offsets, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("selected_offsets_by_class entries must contain integer offsets.") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric % 1.0 == 0.0):
        raise ValueError("selected_offsets_by_class entries must contain integer offsets.")
    return numeric.astype(int, copy=False)


def fit_target_mcca_projection(
    features: Sequence[Sequence[float]] | np.ndarray,
    template: MCCAModel | Sequence[Sequence[float]] | np.ndarray,
    *,
    regularization: float | None = None,
) -> TargetMCCAProjection:
    """Fit a calibrated target-subject projection into an M-CCA space.

    Parameters
    ----------
    features:
        Row-aligned target calibration matrix.  The row order must match the
        fitted M-CCA template.
    template:
        Either a fitted :class:`~neureptrace.decoding.mcca.MCCAModel` or an explicit
        component-template matrix with the same number of rows as ``features``.
        When a model is passed, ``model.component_scores`` is used.
    regularization:
        Non-negative ridge term for the target projection solve.  When omitted
        and ``template`` is an ``MCCAModel``, the model regularization is reused;
        otherwise a default of ``1e-6`` is used.
    """

    matrix = _feature_matrix(features, name="features")
    if isinstance(template, MCCAModel):
        template_matrix = template.component_scores
        if regularization is None:
            regularization = template.regularization
    else:
        template_matrix = _feature_matrix(template, name="template")
        if regularization is None:
            regularization = 1e-6
    regularization = float(regularization)
    if regularization < 0:
        raise ValueError("regularization must be non-negative.")
    if matrix.shape[0] != template_matrix.shape[0]:
        raise ValueError(
            "Target alignment rows must match the M-CCA template rows: "
            f"{matrix.shape[0]} != {template_matrix.shape[0]}."
        )

    feature_mean = np.mean(matrix, axis=0)
    template_mean = np.mean(template_matrix, axis=0)
    centered = matrix - feature_mean
    centered_template = template_matrix - template_mean
    gram = centered @ centered.T
    regularized = gram + regularization * np.eye(gram.shape[0], dtype=float)
    try:
        dual_weights = np.linalg.solve(regularized, centered_template)
    except np.linalg.LinAlgError:
        dual_weights = np.linalg.pinv(regularized) @ centered_template
    projection = centered.T @ dual_weights
    return TargetMCCAProjection(
        feature_mean=feature_mean,
        template_mean=template_mean,
        projection=projection,
        regularization=regularization,
        n_alignment_rows=int(matrix.shape[0]),
    )


def _check_requested_classes(labels: np.ndarray, classes: np.ndarray) -> None:
    if classes.size == 0:
        raise ValueError("classes must contain at least one label.")
    missing = [label for label in classes if not _contains_label(labels, label)]
    if missing:
        raise ValueError(f"classes include labels absent from labels: {missing!r}.")


def _label_vector(labels: Sequence | np.ndarray, *, expected_length: int | None, name: str) -> np.ndarray:
    """Return a 1D object vector while preserving tuple-like anchor labels.

    Target-calibration alignment often uses composite metadata anchors such as
    ``(run, stimulus)``.  ``np.asarray(labels).ravel()`` turns rectangular lists
    of tuples into a flattened array of tuple fields, which changes the apparent
    row count and breaks row-aligned target M-CCA calibration.  Building the
    object vector by assignment keeps each tuple as one atomic label.
    """

    if isinstance(labels, np.ndarray) and labels.ndim == 1:
        vector = labels.astype(object, copy=False).reshape(-1)
    elif isinstance(labels, np.ndarray):
        rows = [tuple(row.tolist()) for row in np.asarray(labels, dtype=object).reshape(labels.shape[0], -1)]
        vector = np.empty(len(rows), dtype=object)
        vector[:] = rows
    else:
        try:
            items = list(labels)
        except TypeError:
            items = [labels]
        vector = np.empty(len(items), dtype=object)
        vector[:] = items
    if expected_length is not None and vector.shape[0] != expected_length:
        raise ValueError(f"{name} length must match feature rows: {vector.shape[0]} != {expected_length}.")
    return vector


def _minimum_class_count(labels: np.ndarray, classes: np.ndarray) -> int:
    return min(_count_label(labels, class_label) for class_label in classes)


def _label_mask(labels: Sequence | np.ndarray, class_label: object) -> np.ndarray:
    """Return a label mask while preserving composite tuple-like labels."""

    return np.asarray(
        [_contains_label([label], class_label) for label in _label_vector(labels, expected_length=None, name="labels")],
        dtype=bool,
    )


def _count_label(labels: Sequence | np.ndarray, class_label: object) -> int:
    """Count labels using the same robust equality semantics as class lookup."""

    return int(np.sum(_label_mask(labels, class_label)))


def _add_template_mean(transformed: Sequence[Sequence[float]] | np.ndarray, template_mean: Sequence[float] | np.ndarray, *, strict: bool) -> np.ndarray:
    matrix = _feature_matrix(transformed, name="transformed")
    mean = np.asarray(template_mean, dtype=float).ravel()
    if mean.size == 0:
        raise ValueError("template_mean must contain at least one value.")
    if not np.all(np.isfinite(mean)):
        raise ValueError("template_mean contains non-finite values.")
    if matrix.shape[1] == mean.shape[0]:
        return matrix + mean
    if matrix.shape[1] % mean.shape[0] == 0:
        repeats = matrix.shape[1] // mean.shape[0]
        return matrix + np.tile(mean, repeats)
    if strict:
        raise ValueError(
            "transformed column count must match or be a multiple of the template width: "
            f"{matrix.shape[1]} vs {mean.shape[0]}."
        )
    return matrix


__all__ = ["TargetMCCAProjection", "class_alignment_matrix", "fit_target_mcca_projection"]

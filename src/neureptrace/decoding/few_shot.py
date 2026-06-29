"""Few-shot target calibration helpers for supervised Category-3 adaptation.

The helpers in this module deliberately separate the held-out target subject into
labeled calibration rows and disjoint evaluation rows.  Calibration rows may be
used for fitting; evaluation rows must only be scored.  This makes the protocol a
supervised/calibrated target-alignment or target-adaptation protocol rather than
a strict source-only or unlabeled target-adaptive benchmark.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neureptrace.decoding import make_decoder, predict_emission_probabilities
from neureptrace.observations import stable_hash

FEW_SHOT_TARGET_CALIBRATION_PROTOCOL = "supervised_target_few_shot_calibration"
FEW_SHOT_TARGET_CALIBRATION_CATEGORY = "3_supervised_calibrated_target_alignment"


@dataclass(frozen=True, slots=True)
class FewShotTargetCalibrationSplit:
    """Disjoint target calibration/evaluation row indices.

    Indices are expressed in the coordinate system of the target fold supplied to
    :func:`fit_few_shot_target_calibrated_decoder` unless a caller passes global
    indices into :func:`select_few_shot_target_calibration_split` directly.
    """

    evaluation_indices: np.ndarray
    calibration_indices: np.ndarray


@dataclass(frozen=True, slots=True)
class FewShotTargetCalibrationResult:
    """Fitted few-shot target-calibrated decoder and fold-local outputs."""

    model: object
    probabilities: np.ndarray
    evaluation_indices: np.ndarray
    calibration_indices: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def _as_1d_object_array(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    """Return a one-dimensional object vector without expanding composite labels."""

    if isinstance(values, np.ndarray):
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            items = [array.item()]
        elif array.ndim == 1:
            items = array.tolist()
        elif array.ndim == 2 and array.shape[1] == 1:
            items = array.reshape(-1).tolist()
        elif array.ndim == 2:
            items = [tuple(row.tolist()) for row in array]
        else:
            raise ValueError(f"{name} must be one-dimensional or a two-dimensional composite-label matrix.")
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError:
            items = [values]
    vector = np.empty(len(items), dtype=object)
    vector[:] = items
    return vector


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
    except Exception:
        return False
    if isinstance(equal, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(equal)


def _unique_labels_in_order(labels: np.ndarray) -> list[Any]:
    unique: list[Any] = []
    for label in labels.tolist():
        if not any(_labels_equal(label, seen) for seen in unique):
            unique.append(label)
    return unique


def _label_index(labels: np.ndarray, class_label: Any) -> int | None:
    for index, candidate in enumerate(labels.tolist()):
        if _labels_equal(candidate, class_label):
            return index
    return None


def _matching_positions(labels: np.ndarray, class_label: Any) -> np.ndarray:
    return np.asarray([index for index, label in enumerate(labels.tolist()) if _labels_equal(label, class_label)], dtype=int)


def _as_feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row.")
    if matrix.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    return matrix


def _stable_rng_seed(seed: int, context: Sequence[Hashable], class_position: int, class_label: Any) -> int:
    return int(
        stable_hash(
            {
                "seed": int(seed),
                "context": tuple(str(item) for item in context),
                "class_position": int(class_position),
                "class_label": str(class_label),
            },
            length=16,
        ),
        16,
    ) % (2**32)


def _normalize_positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(numeric)


def _normalize_nonnegative_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-negative integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0 or numeric < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return int(numeric)


def _normalize_index_vector(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    flat = array.reshape(-1)
    if flat.dtype == np.bool_ or any(isinstance(value, (bool, np.bool_)) for value in flat.tolist()):
        raise ValueError(f"{name} must contain integer row indices, not booleans or a boolean mask.")
    try:
        numeric = np.asarray(flat, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer row indices.") from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric % 1.0 == 0.0):
        raise ValueError(f"{name} must contain integer row indices.")
    return numeric.astype(int, copy=False)


def _reject_duplicate_indices(indices: np.ndarray, *, name: str) -> None:
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} contains duplicate target row indices.")


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Predicted probabilities must be a two-dimensional array.")
    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Predicted probabilities must be finite.")
    probabilities = np.maximum(probabilities, 0.0)
    row_sums = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Predicted probability rows must have positive mass.")
    return probabilities / row_sums


def _align_probability_columns(probabilities: np.ndarray, *, model: object, classes: np.ndarray) -> np.ndarray:
    """Align estimator probability columns to a caller-supplied class order."""

    probabilities = np.asarray(probabilities, dtype=float)
    classes = _as_1d_object_array(classes, name="classes")
    model_classes = getattr(model, "classes_", None)
    if model_classes is None:
        if probabilities.shape[1] != classes.shape[0]:
            raise ValueError(
                "Cannot align probability columns because the fitted model does not expose classes_ "
                f"and emitted {probabilities.shape[1]} columns for {classes.shape[0]} requested classes."
            )
        return _normalize_probability_rows(probabilities)

    model_classes = _as_1d_object_array(model_classes, name="model.classes_")
    if model_classes.shape[0] != probabilities.shape[1]:
        raise ValueError(
            f"Fitted model reports {model_classes.shape[0]} classes but emitted "
            f"{probabilities.shape[1]} probability columns."
        )
    aligned = np.zeros((probabilities.shape[0], classes.shape[0]), dtype=float)
    for source_column, class_label in enumerate(model_classes.tolist()):
        target_column = _label_index(classes, class_label)
        if target_column is None:
            raise ValueError(f"Fitted model emitted unknown class {class_label!r}.")
        aligned[:, target_column] = probabilities[:, source_column]
    return _normalize_probability_rows(aligned)


def select_few_shot_target_calibration_split(
    labels: Sequence[Any] | np.ndarray,
    target_indices: Sequence[int] | np.ndarray | None = None,
    *,
    per_class: int | str = 1,
    seed: int | str = 13,
    context: Sequence[Hashable] = (),
    min_evaluation_per_class: int | str = 1,
) -> FewShotTargetCalibrationSplit:
    """Select deterministic few-shot calibration rows from a target fold.

    The selected calibration rows are balanced by class.  Each class present in
    the supplied target fold contributes exactly ``per_class`` calibration rows,
    and at least ``min_evaluation_per_class`` rows remain for scoring.  The
    function uses labels only to create the calibration/evaluation split, which
    makes downstream use Category 3 rather than Category 1/2.
    """

    label_vector = _as_1d_object_array(labels, name="labels")
    if target_indices is None:
        indices = np.arange(label_vector.shape[0], dtype=int)
    else:
        indices = _normalize_index_vector(target_indices, name="target_indices")
    if indices.size == 0:
        raise ValueError("few-shot target calibration requires at least one target row.")
    if np.any(indices < 0) or np.any(indices >= label_vector.shape[0]):
        raise ValueError("target_indices contains an out-of-range row index.")
    _reject_duplicate_indices(indices, name="target_indices")

    per_class_count = _normalize_positive_int(per_class, name="few_shot_target_calibration_per_class")
    min_eval = _normalize_nonnegative_int(min_evaluation_per_class, name="few_shot_min_evaluation_per_class")
    seed_value = _normalize_nonnegative_int(seed, name="few_shot_target_calibration_seed")

    target_labels = label_vector[indices]
    classes = _unique_labels_in_order(target_labels)
    calibration_mask = np.zeros(indices.shape[0], dtype=bool)
    for class_position, class_label in enumerate(classes):
        positions = _matching_positions(target_labels, class_label)
        required = per_class_count + min_eval
        if positions.size < required:
            raise ValueError(
                "few-shot target calibration needs at least "
                f"{required} target rows for class {class_label!r}; got {positions.size}."
            )
        rng = np.random.default_rng(_stable_rng_seed(seed_value, context, class_position, class_label))
        selected = rng.choice(positions, size=per_class_count, replace=False)
        calibration_mask[selected] = True

    calibration_indices = indices[calibration_mask]
    evaluation_indices = indices[~calibration_mask]
    return FewShotTargetCalibrationSplit(
        evaluation_indices=evaluation_indices.astype(int, copy=False),
        calibration_indices=calibration_indices.astype(int, copy=False),
    )


def few_shot_target_calibration_metadata(
    *,
    n_source_rows: int,
    n_target_rows: int,
    n_calibration_rows: int,
    n_evaluation_rows: int,
    per_class: int,
    seed: int,
    target_repeats: int = 1,
) -> dict[str, Any]:
    """Return protocol metadata suitable for metrics or observation tables."""

    return {
        "few_shot_target_calibration": True,
        "few_shot_protocol": FEW_SHOT_TARGET_CALIBRATION_PROTOCOL,
        "few_shot_protocol_category": FEW_SHOT_TARGET_CALIBRATION_CATEGORY,
        "few_shot_uses_target_features": True,
        "few_shot_uses_target_labels": True,
        "few_shot_valid_for_strict_source_only": False,
        "few_shot_valid_for_unlabeled_target_adaptation": False,
        "few_shot_valid_for_benchmark": False,
        "few_shot_target_calibration_per_class": int(per_class),
        "few_shot_target_calibration_seed": int(seed),
        "few_shot_target_repeats": int(target_repeats),
        "few_shot_n_source_rows": int(n_source_rows),
        "few_shot_n_target_rows": int(n_target_rows),
        "few_shot_n_target_calibration_rows": int(n_calibration_rows),
        "few_shot_n_target_evaluation_rows": int(n_evaluation_rows),
    }


def fit_few_shot_target_calibrated_decoder(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    target_labels: Sequence[Any] | np.ndarray,
    classes: Sequence[Any] | np.ndarray | None = None,
    split: FewShotTargetCalibrationSplit | None = None,
    per_class: int | str = 1,
    seed: int | str = 13,
    context: Sequence[Hashable] = (),
    min_evaluation_per_class: int | str = 1,
    target_repeats: int | str = 1,
    decoder_name: str = "logistic",
    emission_mode: str = "uncalibrated",
    max_iter: int = 1000,
    feature_preprocessor: str = "none",
    pca_components: int | float | None = None,
    tune_hyperparameters: bool = False,
    tuning_cv: Any = 3,
    tuning_scoring: str = "accuracy",
    tuning_c_grid: Sequence[float] | str | None = None,
    decoder_kwargs: Mapping[str, Any] | None = None,
) -> FewShotTargetCalibrationResult:
    """Fit a source+few-shot target decoder and predict held-out target rows.

    ``target_labels`` are used only to choose and label the calibration subset.
    The returned ``probabilities`` correspond to ``result.evaluation_indices``;
    callers should score those rows only.  No target evaluation labels are used in
    fitting, tuning, or probability alignment.
    """

    source_matrix = _as_feature_matrix(source_features, name="source_features")
    target_matrix = _as_feature_matrix(target_features, name="target_features")
    if source_matrix.shape[1] != target_matrix.shape[1]:
        raise ValueError(
            "source_features and target_features must have the same feature width: "
            f"{source_matrix.shape[1]} != {target_matrix.shape[1]}."
        )

    source_label_vector = _as_1d_object_array(source_labels, name="source_labels")
    target_label_vector = _as_1d_object_array(target_labels, name="target_labels")
    if source_matrix.shape[0] != source_label_vector.shape[0]:
        raise ValueError("source_features and source_labels must have the same row count.")
    if target_matrix.shape[0] != target_label_vector.shape[0]:
        raise ValueError("target_features and target_labels must have the same row count.")

    per_class_count = _normalize_positive_int(per_class, name="few_shot_target_calibration_per_class")
    seed_value = _normalize_nonnegative_int(seed, name="few_shot_target_calibration_seed")
    repeats = _normalize_positive_int(target_repeats, name="few_shot_target_repeats")
    if split is None:
        split = select_few_shot_target_calibration_split(
            target_label_vector,
            per_class=per_class_count,
            seed=seed_value,
            context=context,
            min_evaluation_per_class=min_evaluation_per_class,
        )

    calibration_indices = _normalize_index_vector(split.calibration_indices, name="calibration_indices")
    evaluation_indices = _normalize_index_vector(split.evaluation_indices, name="evaluation_indices")
    if calibration_indices.size == 0:
        raise ValueError("few-shot target calibration selected no calibration rows.")
    if evaluation_indices.size == 0:
        raise ValueError("few-shot target calibration selected no evaluation rows.")
    for name, indices in {"calibration_indices": calibration_indices, "evaluation_indices": evaluation_indices}.items():
        if np.any(indices < 0) or np.any(indices >= target_matrix.shape[0]):
            raise ValueError(f"{name} contains an out-of-range target row index.")
        _reject_duplicate_indices(indices, name=name)
    if np.intersect1d(calibration_indices, evaluation_indices).size:
        raise ValueError("few-shot calibration and evaluation indices must be disjoint.")

    calibration_features = target_matrix[calibration_indices]
    calibration_labels = target_label_vector[calibration_indices]
    if repeats > 1:
        calibration_features = np.repeat(calibration_features, repeats, axis=0)
        calibration_labels = np.repeat(calibration_labels, repeats, axis=0)
    fit_features = np.vstack([source_matrix, calibration_features])
    fit_labels = np.concatenate([source_label_vector, calibration_labels])

    model_kwargs = dict(decoder_kwargs or {})
    model = make_decoder(
        decoder_name,
        max_iter=max_iter,
        emission_mode=emission_mode,
        feature_preprocessor=feature_preprocessor,
        pca_components=pca_components,
        tune_hyperparameters=tune_hyperparameters,
        tuning_cv=tuning_cv,
        tuning_scoring=tuning_scoring,
        tuning_c_grid=tuning_c_grid,
        **model_kwargs,
    )
    model.fit(fit_features, fit_labels)

    observed_class_labels = np.concatenate([source_label_vector, calibration_labels]).astype(object)
    class_order = (
        _as_1d_object_array(classes, name="classes")
        if classes is not None
        else _as_1d_object_array(_unique_labels_in_order(observed_class_labels), name="classes")
    )
    probabilities = _align_probability_columns(
        predict_emission_probabilities(
            model,
            target_matrix[evaluation_indices],
            emission_mode=emission_mode,
        ),
        model=model,
        classes=class_order,
    )

    metadata = few_shot_target_calibration_metadata(
        n_source_rows=source_matrix.shape[0],
        n_target_rows=target_matrix.shape[0],
        n_calibration_rows=calibration_indices.size,
        n_evaluation_rows=evaluation_indices.size,
        per_class=per_class_count,
        seed=seed_value,
        target_repeats=repeats,
    )
    return FewShotTargetCalibrationResult(
        model=model,
        probabilities=probabilities,
        evaluation_indices=evaluation_indices.copy(),
        calibration_indices=calibration_indices.copy(),
        metadata=metadata,
    )

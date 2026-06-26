"""Treat composite class labels atomically in calibration split helpers."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps
from typing import Any

import numpy as np

_ALL_PROTOCOLS_PATCH_MARKER = "_neureptrace_tuple_label_all_protocols_patch_installed"
_FEW_SHOT_PATCH_MARKER = "_neureptrace_tuple_label_few_shot_patch_installed"
_INDEX_ERROR = "{name} must contain integer row indices."
_BOOLEAN_INDEX_ERROR = "{name} must contain integer row indices, not booleans or a boolean mask."
_DUPLICATE_INDEX_ERROR = "{name} must not contain duplicate target row indices."
_SHAPE_INDEX_ERROR = "{name} must be one-dimensional."


def _object_value_vector(values: Sequence[Any]) -> np.ndarray:
    vector = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        vector[index] = value
    return vector


def _atomic_label_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    """Return a 1-D object vector while preserving tuple/list row labels.

    ``np.asarray([("run-1", "face"), ...], dtype=object)`` has shape
    ``(n, 2)``.  For class labels, each row is one composite value rather than
    two independent labels, so collapse row-shaped label arrays into tuple
    objects before downstream class counting or equality checks.
    """

    array = np.asarray(values, dtype=object)
    if array.ndim == 0:
        return _object_value_vector([array.item()])
    if array.ndim == 1:
        return array.reshape(-1)
    rows = [tuple(row.tolist()) for row in array.reshape(array.shape[0], -1)]
    return _object_value_vector(rows)


def _values_equal(left: object, right: object) -> bool:
    try:
        equal = left == right
    except (TypeError, ValueError):
        return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        return False


def _value_mask(values: Sequence[Any] | np.ndarray, target: object) -> np.ndarray:
    return np.asarray([_values_equal(value, target) for value in _atomic_label_vector(values, name="values")], dtype=bool)


def _ordered_unique_values(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    unique: list[object] = []
    for value in _atomic_label_vector(values, name="values"):
        if not any(_values_equal(existing, value) for existing in unique):
            unique.append(value)
    return _object_value_vector(unique)


def _unique_label_values(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    vector = _atomic_label_vector(values, name="values")
    try:
        return _object_value_vector(np.unique(vector).tolist())
    except (TypeError, ValueError):
        return _ordered_unique_values(vector)


def _normalize_manual_split_indices(values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    if array.ndim != 1:
        raise ValueError(_SHAPE_INDEX_ERROR.format(name=name))
    flat = array.reshape(-1)
    if flat.dtype == np.bool_ or any(isinstance(value, (bool, np.bool_)) for value in flat.tolist()):
        raise ValueError(_BOOLEAN_INDEX_ERROR.format(name=name))
    try:
        numeric = np.asarray(flat, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(_INDEX_ERROR.format(name=name)) from exc
    if not np.all(np.isfinite(numeric)) or not np.all(numeric % 1.0 == 0.0):
        raise ValueError(_INDEX_ERROR.format(name=name))
    indices = numeric.astype(int, copy=False)
    if np.unique(indices).size != indices.size:
        raise ValueError(_DUPLICATE_INDEX_ERROR.format(name=name))
    return indices


def _patch_all_protocols() -> None:
    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _ALL_PROTOCOLS_PATCH_MARKER, False):
        return

    original_select = all_protocols.select_bushmeg_target_calibration_split
    original_category_split = all_protocols.category3_calibration_evaluation_split

    @wraps(original_select)
    def select_bushmeg_target_calibration_split(
        target_labels: Sequence[Any] | np.ndarray,
        *,
        per_class: int,
        seed: int,
        min_evaluation_per_class: int = 1,
        context: Sequence[Any] = (),
    ):
        context_tuple = tuple(str(item) for item in context)
        try:
            per_class_count = int(per_class)
        except (TypeError, ValueError):
            per_class_count = 0
        try:
            seed_value = int(seed)
        except (TypeError, ValueError):
            seed_value = 0
        try:
            min_eval_count = int(min_evaluation_per_class)
        except (TypeError, ValueError):
            min_eval_count = 0

        if per_class_count < 1:
            return all_protocols._empty_bushmeg_target_calibration_split(
                per_class=per_class_count,
                seed=seed_value,
                min_evaluation_per_class=min_eval_count,
                context=context_tuple,
                skip_reason="Protocol 3 target calibration requires per_class >= 1.",
                skip_reason_code="invalid_per_class",
            )
        if min_eval_count < 1:
            return all_protocols._empty_bushmeg_target_calibration_split(
                per_class=per_class_count,
                seed=seed_value,
                min_evaluation_per_class=min_eval_count,
                context=context_tuple,
                skip_reason="Protocol 3 target calibration requires min_evaluation_per_class >= 1.",
                skip_reason_code="invalid_min_evaluation_per_class",
            )

        labels_array = _atomic_label_vector(target_labels, name="target_labels")
        if labels_array.size == 0:
            return all_protocols._empty_bushmeg_target_calibration_split(
                per_class=per_class_count,
                seed=seed_value,
                min_evaluation_per_class=min_eval_count,
                context=context_tuple,
                skip_reason="Protocol 3 target labels must not be empty.",
                skip_reason_code="empty_labels",
            )

        classes = _unique_label_values(labels_array)
        required_per_class = per_class_count + min_eval_count
        for class_value in classes:
            class_count = int(np.count_nonzero(_value_mask(labels_array, class_value)))
            if class_count < required_per_class:
                readable_class = all_protocols._readable_label_value(class_value)
                return all_protocols._empty_bushmeg_target_calibration_split(
                    per_class=per_class_count,
                    seed=seed_value,
                    min_evaluation_per_class=min_eval_count,
                    context=context_tuple,
                    skip_reason=(
                        "Protocol 3 target calibration is infeasible: "
                        f"class {readable_class!r} has {class_count} row(s), but needs at least "
                        f"{per_class_count} calibration row(s) plus {min_eval_count} evaluation row(s)."
                    ),
                    skip_reason_code="insufficient_rows_per_class",
                    n_classes=int(classes.size),
                )

        effective_seed = all_protocols._stable_target_calibration_seed(seed_value, per_class=per_class_count, context=context_tuple)
        rng = np.random.default_rng(effective_seed)
        calibration: list[int] = []
        evaluation_mask = np.ones(labels_array.shape[0], dtype=bool)
        for class_value in classes:
            class_indices = np.flatnonzero(_value_mask(labels_array, class_value))
            selected = rng.choice(class_indices, size=per_class_count, replace=False)
            calibration.extend(int(index) for index in selected)
            evaluation_mask[selected] = False

        calibration_indices = np.asarray(sorted(calibration), dtype=int)
        evaluation_indices = np.flatnonzero(evaluation_mask).astype(int, copy=False)
        if np.intersect1d(calibration_indices, evaluation_indices).size:
            return all_protocols._empty_bushmeg_target_calibration_split(
                per_class=per_class_count,
                seed=seed_value,
                min_evaluation_per_class=min_eval_count,
                context=context_tuple,
                skip_reason="Protocol 3 target calibration/evaluation rows overlap after selection.",
                skip_reason_code="overlapping_rows",
                n_classes=int(classes.size),
            )

        evaluation_labels = labels_array[evaluation_indices]
        for class_value in classes:
            evaluation_count = int(np.count_nonzero(_value_mask(evaluation_labels, class_value)))
            if evaluation_count < min_eval_count:
                readable_class = all_protocols._readable_label_value(class_value)
                return all_protocols._empty_bushmeg_target_calibration_split(
                    per_class=per_class_count,
                    seed=seed_value,
                    min_evaluation_per_class=min_eval_count,
                    context=context_tuple,
                    skip_reason=(
                        "Protocol 3 target calibration consumed too many rows: "
                        f"class {readable_class!r} has {evaluation_count} evaluation row(s), "
                        f"expected at least {min_eval_count}."
                    ),
                    skip_reason_code="evaluation_class_consumed",
                    n_classes=int(classes.size),
                )

        return all_protocols.BushmegTargetCalibrationSplit(
            calibration_indices=calibration_indices,
            evaluation_indices=evaluation_indices,
            per_class=per_class_count,
            seed=seed_value,
            min_evaluation_per_class=min_eval_count,
            context=context_tuple,
            effective_seed=effective_seed,
            skipped=False,
            skip_reason="",
            skip_reason_code="",
            n_classes=int(classes.size),
        )

    @wraps(original_category_split)
    def category3_calibration_evaluation_split(
        labels: Sequence[Any] | np.ndarray,
        *,
        calibration_per_class: int = 1,
        seed: int = 13,
    ) -> tuple[np.ndarray, np.ndarray]:
        split = select_bushmeg_target_calibration_split(
            labels,
            per_class=calibration_per_class,
            seed=seed,
            min_evaluation_per_class=1,
        )
        if split.skipped:
            raise ValueError(split.skip_reason)
        all_protocols.validate_disjoint_calibration_evaluation(split.calibration_indices, split.evaluation_indices)
        return split.calibration_indices, split.evaluation_indices

    all_protocols.select_bushmeg_target_calibration_split = select_bushmeg_target_calibration_split
    all_protocols.category3_calibration_evaluation_split = category3_calibration_evaluation_split
    setattr(all_protocols, _ALL_PROTOCOLS_PATCH_MARKER, True)


def _patch_few_shot() -> None:
    few_shot = importlib.import_module("neureptrace.decoding.few_shot")
    if getattr(few_shot, _FEW_SHOT_PATCH_MARKER, False):
        return

    original_select = few_shot.select_few_shot_target_calibration_split
    original_align = few_shot._align_probability_columns
    original_fit = few_shot.fit_few_shot_target_calibrated_decoder

    def _as_1d_object_array(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
        return _atomic_label_vector(values, name=name)

    @wraps(original_select)
    def select_few_shot_target_calibration_split(
        labels: Sequence[Any] | np.ndarray,
        target_indices: Sequence[int] | np.ndarray | None = None,
        *,
        per_class: int | str = 1,
        seed: int | str = 13,
        context: Sequence[Any] = (),
        min_evaluation_per_class: int | str = 1,
    ):
        label_vector = _atomic_label_vector(labels, name="labels")
        if target_indices is None:
            indices = np.arange(label_vector.shape[0], dtype=int)
        else:
            indices = _normalize_manual_split_indices(target_indices, name="target_indices")
        if indices.size == 0:
            raise ValueError("few-shot target calibration requires at least one target row.")
        if np.any(indices < 0) or np.any(indices >= label_vector.shape[0]):
            raise ValueError("target_indices contains an out-of-range row index.")

        per_class_count = few_shot._normalize_positive_int(per_class, name="few_shot_target_calibration_per_class")
        min_eval = few_shot._normalize_nonnegative_int(min_evaluation_per_class, name="few_shot_min_evaluation_per_class")
        seed_value = few_shot._normalize_nonnegative_int(seed, name="few_shot_target_calibration_seed")

        target_labels = label_vector[indices]
        classes = _ordered_unique_values(target_labels)
        calibration_mask = np.zeros(indices.shape[0], dtype=bool)
        for class_position, class_label in enumerate(classes):
            positions = np.flatnonzero(_value_mask(target_labels, class_label))
            required = per_class_count + min_eval
            if positions.size < required:
                raise ValueError(
                    "few-shot target calibration needs at least "
                    f"{required} target rows for class {class_label!r}; got {positions.size}."
                )
            rng = np.random.default_rng(few_shot._stable_rng_seed(seed_value, context, class_position, class_label))
            selected = rng.choice(positions, size=per_class_count, replace=False)
            calibration_mask[selected] = True

        calibration_indices = indices[calibration_mask]
        evaluation_indices = indices[~calibration_mask]
        return few_shot.FewShotTargetCalibrationSplit(
            evaluation_indices=evaluation_indices.astype(int, copy=False),
            calibration_indices=calibration_indices.astype(int, copy=False),
        )

    @wraps(original_align)
    def _align_probability_columns(probabilities: np.ndarray, *, model: object, classes: Sequence[Any] | np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probabilities, dtype=float)
        classes_vector = _atomic_label_vector(classes, name="classes")
        model_classes = getattr(model, "classes_", None)
        if model_classes is None:
            if probabilities.shape[1] != classes_vector.shape[0]:
                raise ValueError(
                    "Cannot align probability columns because the fitted model does not expose classes_ "
                    f"and emitted {probabilities.shape[1]} columns for {classes_vector.shape[0]} requested classes."
                )
            return few_shot._normalize_probability_rows(probabilities)

        model_class_vector = _atomic_label_vector(model_classes, name="model_classes")
        if model_class_vector.shape[0] != probabilities.shape[1]:
            raise ValueError(
                f"Fitted model reports {model_class_vector.shape[0]} classes but emitted "
                f"{probabilities.shape[1]} probability columns."
            )
        aligned = np.zeros((probabilities.shape[0], classes_vector.shape[0]), dtype=float)
        for source_column, class_label in enumerate(model_class_vector):
            matches = [index for index, requested in enumerate(classes_vector) if _values_equal(requested, class_label)]
            if not matches:
                raise ValueError(f"Fitted model emitted unknown class {class_label!r}.")
            aligned[:, matches[0]] = probabilities[:, source_column]
        return few_shot._normalize_probability_rows(aligned)

    @wraps(original_fit)
    def fit_few_shot_target_calibrated_decoder(*args: Any, **kwargs: Any):
        split = kwargs.get("split")
        if split is not None:
            kwargs = dict(kwargs)
            kwargs["split"] = few_shot.FewShotTargetCalibrationSplit(
                evaluation_indices=_normalize_manual_split_indices(split.evaluation_indices, name="evaluation_indices"),
                calibration_indices=_normalize_manual_split_indices(split.calibration_indices, name="calibration_indices"),
            )
        if kwargs.get("classes") is not None:
            kwargs = dict(kwargs)
            kwargs["classes"] = _atomic_label_vector(kwargs["classes"], name="classes")
        return original_fit(*args, **kwargs)

    few_shot._as_1d_object_array = _as_1d_object_array
    few_shot.select_few_shot_target_calibration_split = select_few_shot_target_calibration_split
    few_shot._align_probability_columns = _align_probability_columns
    few_shot.fit_few_shot_target_calibrated_decoder = fit_few_shot_target_calibrated_decoder
    setattr(few_shot, _FEW_SHOT_PATCH_MARKER, True)


def install() -> None:
    """Patch calibration split helpers to preserve tuple/composite labels."""

    _patch_all_protocols()
    _patch_few_shot()


__all__ = ["install"]

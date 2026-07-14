"""Runtime guardrails for Protocol 3 calibration/evaluation index vectors.

Protocol 3 uses disjoint same-subject calibration and evaluation rows. The
core validator historically delegated to ``np.asarray(..., dtype=int)`` and
``np.intersect1d``. That silently flattened malformed matrix-shaped split
indices and coerced booleans or fractional values into row numbers. This patch
normalizes only genuine one-dimensional integer index vectors before the
disjointness check runs.

The same bool-is-int pitfall also applies to the Protocol 3 split-size options:
``per_class=True`` or ``per_class=np.array([True])`` silently meant one
calibration row per class. For YAML-backed or programmatic experimental configs
that should be rejected explicitly rather than interpreted as a numeric count or
seed.

The split selector must also preserve one logical label per row. A target label
vector containing tuple/list composite labels otherwise becomes a two-dimensional
NumPy array and is incorrectly skipped as ``labels_not_one_dimensional``.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.bushmeg_all_protocols"
_PATCH_MARKER = "_neureptrace_protocol3_index_validation_patch_installed"
_FINDER_MARKER = "_neureptrace_protocol3_index_validation_finder"


def _as_index_vector(values: Any, *, name: str) -> np.ndarray:
    """Return a validated 1D integer index vector."""

    array = np.asarray(values)
    if array.ndim == 0:
        array = array.reshape(1)
    elif array.ndim == 1:
        array = array.reshape(-1)
    elif array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    else:
        raise ValueError(f"{name} must be a one-dimensional index vector; got shape {array.shape}.")

    if np.issubdtype(array.dtype, np.bool_):
        raise ValueError(f"{name} must contain integer row indices, not boolean values.")
    if array.size == 0:
        return array.astype(int, copy=False)

    try:
        numeric = np.asarray(array, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer row indices.") from exc

    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite integer row indices.")
    if not np.all(np.equal(numeric, np.trunc(numeric))):
        raise ValueError(f"{name} must contain integer row indices.")

    return numeric.astype(int, copy=False)


def _is_boolean_integer_option(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return True
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    return bool(np.issubdtype(array.dtype, np.bool_))


def _reject_boolean_integer_option(value: Any, *, name: str) -> None:
    if _is_boolean_integer_option(value):
        raise ValueError(f"{name} must be an integer value, not a boolean value.")


def _label_vector(labels: Any) -> np.ndarray:
    """Return one label object per row, preserving composite tuple/list labels."""

    try:
        labels_array = np.asarray(labels, dtype=object)
    except ValueError as exc:
        raise ValueError("target labels must form a row-indexed label vector") from exc

    if labels_array.ndim == 0:
        vector = np.empty(1, dtype=object)
        vector[0] = labels_array.item()
        return vector
    if labels_array.ndim == 1:
        return labels_array.reshape(-1)

    rows = labels_array.reshape(labels_array.shape[0], -1)
    if rows.shape[1] == 1:
        return rows[:, 0].reshape(-1)

    vector = np.empty(rows.shape[0], dtype=object)
    for index, row in enumerate(rows):
        vector[index] = tuple(row.tolist())
    return vector


def _values_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
    except (TypeError, ValueError):
        try:
            return bool(np.array_equal(left, right))
        except Exception:
            return False
    try:
        return bool(equal)
    except (TypeError, ValueError):
        try:
            return bool(np.array_equal(left, right))
        except Exception:
            return False


def _unique_label_values(labels: np.ndarray) -> list[Any]:
    try:
        return list(np.unique(labels))
    except (TypeError, ValueError):
        unique: list[Any] = []
        for label in labels.tolist():
            if not any(_values_equal(label, seen) for seen in unique):
                unique.append(label)
        return unique


def _indices_for_label(labels: np.ndarray, class_value: Any) -> np.ndarray:
    matches = [index for index, label in enumerate(labels.tolist()) if _values_equal(label, class_value)]
    return np.asarray(matches, dtype=int)


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_select_split = module.select_bushmeg_target_calibration_split
    original_category3_split = module.category3_calibration_evaluation_split

    def validate_disjoint_calibration_evaluation(calibration_indices: Any, evaluation_indices: Any) -> None:
        calibration = _as_index_vector(calibration_indices, name="calibration_indices")
        evaluation = _as_index_vector(evaluation_indices, name="evaluation_indices")
        overlap = np.intersect1d(calibration, evaluation)
        if overlap.size:
            preview = ",".join(map(str, overlap[:10]))
            raise ValueError(f"Protocol 3 calibration/evaluation rows must be disjoint; overlapping row(s): {preview}.")

    @wraps(original_select_split)
    def select_bushmeg_target_calibration_split(
        target_labels: Any,
        *,
        per_class: Any,
        seed: Any,
        min_evaluation_per_class: Any = 1,
        context: Any = (),
    ) -> Any:
        _reject_boolean_integer_option(per_class, name="per_class")
        _reject_boolean_integer_option(seed, name="seed")
        _reject_boolean_integer_option(min_evaluation_per_class, name="min_evaluation_per_class")

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
            return module._empty_bushmeg_target_calibration_split(
                per_class=per_class_count,
                seed=seed_value,
                min_evaluation_per_class=min_eval_count,
                context=context_tuple,
                skip_reason="Protocol 3 target calibration requires per_class >= 1.",
                skip_reason_code="invalid_per_class",
            )
        if min_eval_count < 1:
            return module._empty_bushmeg_target_calibration_split(
                per_class=per_class_count,
                seed=seed_value,
                min_evaluation_per_class=min_eval_count,
                context=context_tuple,
                skip_reason="Protocol 3 target calibration requires min_evaluation_per_class >= 1.",
                skip_reason_code="invalid_min_evaluation_per_class",
            )

        try:
            labels_array = _label_vector(target_labels)
        except (TypeError, ValueError):
            return module._empty_bushmeg_target_calibration_split(
                per_class=per_class_count,
                seed=seed_value,
                min_evaluation_per_class=min_eval_count,
                context=context_tuple,
                skip_reason="Protocol 3 target labels must be one-dimensional.",
                skip_reason_code="labels_not_one_dimensional",
            )
        if labels_array.ndim != 1:
            return module._empty_bushmeg_target_calibration_split(
                per_class=per_class_count,
                seed=seed_value,
                min_evaluation_per_class=min_eval_count,
                context=context_tuple,
                skip_reason="Protocol 3 target labels must be one-dimensional.",
                skip_reason_code="labels_not_one_dimensional",
            )
        if labels_array.size == 0:
            return module._empty_bushmeg_target_calibration_split(
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
            class_count = int(_indices_for_label(labels_array, class_value).size)
            if class_count < required_per_class:
                readable_class = module._readable_label_value(class_value)
                return module._empty_bushmeg_target_calibration_split(
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
                    n_classes=int(len(classes)),
                )

        effective_seed = module._stable_target_calibration_seed(seed_value, per_class=per_class_count, context=context_tuple)
        rng = np.random.default_rng(effective_seed)
        calibration: list[int] = []
        evaluation_mask = np.ones(labels_array.shape[0], dtype=bool)
        for class_value in classes:
            class_indices = _indices_for_label(labels_array, class_value)
            selected = rng.choice(class_indices, size=per_class_count, replace=False)
            calibration.extend(int(index) for index in selected)
            evaluation_mask[selected] = False

        calibration_indices = np.asarray(sorted(calibration), dtype=int)
        evaluation_indices = np.flatnonzero(evaluation_mask).astype(int, copy=False)
        if np.intersect1d(calibration_indices, evaluation_indices).size:
            return module._empty_bushmeg_target_calibration_split(
                per_class=per_class_count,
                seed=seed_value,
                min_evaluation_per_class=min_eval_count,
                context=context_tuple,
                skip_reason="Protocol 3 target calibration/evaluation rows overlap after selection.",
                skip_reason_code="overlapping_rows",
                n_classes=int(len(classes)),
            )

        evaluation_labels = labels_array[evaluation_indices]
        for class_value in classes:
            evaluation_count = int(_indices_for_label(evaluation_labels, class_value).size)
            if evaluation_count < min_eval_count:
                readable_class = module._readable_label_value(class_value)
                return module._empty_bushmeg_target_calibration_split(
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
                    n_classes=int(len(classes)),
                )

        return module.BushmegTargetCalibrationSplit(
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
            n_classes=int(len(classes)),
        )

    @wraps(original_category3_split)
    def category3_calibration_evaluation_split(labels: Any, *, calibration_per_class: Any = 1, seed: Any = 13) -> Any:
        _reject_boolean_integer_option(calibration_per_class, name="calibration_per_class")
        _reject_boolean_integer_option(seed, name="seed")
        return original_category3_split(labels, calibration_per_class=calibration_per_class, seed=seed)

    module.validate_disjoint_calibration_evaluation = validate_disjoint_calibration_evaluation
    module.select_bushmeg_target_calibration_split = select_bushmeg_target_calibration_split
    module.category3_calibration_evaluation_split = category3_calibration_evaluation_split
    setattr(module, _PATCH_MARKER, True)


class _Protocol3IndexValidationLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader) -> None:
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):  # type: ignore[override]
        create_module = getattr(self.wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped_loader.exec_module(module)
        _patch_module(module)


class _Protocol3IndexValidationFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _Protocol3IndexValidationLoader):
            return spec
        spec.loader = _Protocol3IndexValidationLoader(spec.loader)
        return spec


def install() -> None:
    """Install Protocol 3 split-index and split-count validation."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)
        return

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _Protocol3IndexValidationFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)


__all__ = ["install"]

"""Runtime patch for tuple/composite labels in few-shot calibration."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from collections.abc import Sequence
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.decoding.few_shot"
_PATCH_MARKER = "_neureptrace_few_shot_tuple_labels_patch_installed"
_FINDER_MARKER = "_neureptrace_few_shot_tuple_labels_finder"
_DUPLICATE_INDEX_ERROR = "{name} must not contain duplicate target row indices."


def _as_label_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    """Return a one-dimensional object vector without expanding tuple labels."""

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
        except TypeError as exc:
            raise ValueError(f"{name} must be one-dimensional.") from exc
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


def _normalize_unique_indices(module: ModuleType, values: Sequence[int] | np.ndarray, *, name: str) -> np.ndarray:
    indices = module._normalize_index_vector(values, name=name)
    if np.unique(indices).size != indices.size:
        raise ValueError(_DUPLICATE_INDEX_ERROR.format(name=name))
    return indices


def _observed_few_shot_class_order(
    module: ModuleType,
    source_labels: Sequence[Any] | np.ndarray,
    target_labels: Sequence[Any] | np.ndarray,
    split: Any,
) -> np.ndarray:
    """Return source-plus-calibration classes without sorting object labels."""

    source_vector = _as_label_vector(source_labels, name="source_labels")
    target_vector = _as_label_vector(target_labels, name="target_labels")
    calibration_indices = _normalize_unique_indices(module, split.calibration_indices, name="calibration_indices")
    if np.any(calibration_indices < 0) or np.any(calibration_indices >= target_vector.shape[0]):
        raise ValueError("calibration_indices contains an out-of-range target row index.")

    calibration_labels = target_vector[calibration_indices]
    observed = np.empty(source_vector.shape[0] + calibration_labels.shape[0], dtype=object)
    observed[: source_vector.shape[0]] = source_vector
    observed[source_vector.shape[0] :] = calibration_labels
    return _as_label_vector(_unique_labels_in_order(observed), name="classes")


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_fit = module.fit_few_shot_target_calibrated_decoder

    def _align_probability_columns(probabilities: np.ndarray, *, model: object, classes: np.ndarray) -> np.ndarray:
        """Align estimator probability columns to a caller-supplied class order."""

        probabilities = np.asarray(probabilities, dtype=float)
        classes = _as_label_vector(classes, name="classes")
        model_classes = getattr(model, "classes_", None)
        if model_classes is None:
            if probabilities.shape[1] != classes.shape[0]:
                raise ValueError(
                    "Cannot align probability columns because the fitted model does not expose classes_ "
                    f"and emitted {probabilities.shape[1]} columns for {classes.shape[0]} requested classes."
                )
            return module._normalize_probability_rows(probabilities)

        model_classes = _as_label_vector(model_classes, name="model.classes_")
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
        return module._normalize_probability_rows(aligned)

    def select_few_shot_target_calibration_split(
        labels: Sequence[Any] | np.ndarray,
        target_indices: Sequence[int] | np.ndarray | None = None,
        *,
        per_class: int | str = 1,
        seed: int | str = 13,
        context: Sequence[Any] = (),
        min_evaluation_per_class: int | str = 1,
    ) -> Any:
        """Select deterministic few-shot calibration rows from a target fold."""

        label_vector = _as_label_vector(labels, name="labels")
        if target_indices is None:
            indices = np.arange(label_vector.shape[0], dtype=int)
        else:
            indices = _normalize_unique_indices(module, target_indices, name="target_indices")
        if indices.size == 0:
            raise ValueError("few-shot target calibration requires at least one target row.")
        if np.any(indices < 0) or np.any(indices >= label_vector.shape[0]):
            raise ValueError("target_indices contains an out-of-range row index.")

        per_class_count = module._normalize_positive_int(per_class, name="few_shot_target_calibration_per_class")
        min_eval = module._normalize_nonnegative_int(min_evaluation_per_class, name="few_shot_min_evaluation_per_class")
        seed_value = module._normalize_nonnegative_int(seed, name="few_shot_target_calibration_seed")

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
            rng = np.random.default_rng(module._stable_rng_seed(seed_value, context, class_position, class_label))
            selected = rng.choice(positions, size=per_class_count, replace=False)
            calibration_mask[selected] = True

        calibration_indices = indices[calibration_mask]
        evaluation_indices = indices[~calibration_mask]
        return module.FewShotTargetCalibrationSplit(
            evaluation_indices=evaluation_indices.astype(int, copy=False),
            calibration_indices=calibration_indices.astype(int, copy=False),
        )

    @wraps(original_fit)
    def fit_few_shot_target_calibrated_decoder(*args: Any, **kwargs: Any) -> Any:
        if args:
            return original_fit(*args, **kwargs)

        kwargs = dict(kwargs)
        if kwargs.get("classes") is None and "source_labels" in kwargs and "target_labels" in kwargs:
            split = kwargs.get("split")
            if split is None:
                split = select_few_shot_target_calibration_split(
                    kwargs["target_labels"],
                    per_class=kwargs.get("per_class", 1),
                    seed=kwargs.get("seed", 13),
                    context=kwargs.get("context", ()),
                    min_evaluation_per_class=kwargs.get("min_evaluation_per_class", 1),
                )
                kwargs["split"] = split
            kwargs["classes"] = _observed_few_shot_class_order(
                module,
                kwargs["source_labels"],
                kwargs["target_labels"],
                split,
            )
        elif kwargs.get("classes") is not None:
            kwargs["classes"] = _as_label_vector(kwargs["classes"], name="classes")
        return original_fit(**kwargs)

    module._as_1d_object_array = _as_label_vector
    module._align_probability_columns = _align_probability_columns
    module.select_few_shot_target_calibration_split = select_few_shot_target_calibration_split
    module.fit_few_shot_target_calibrated_decoder = fit_few_shot_target_calibrated_decoder
    setattr(module, _PATCH_MARKER, True)


class _FewShotTupleLabelsPatchLoader(importlib.abc.Loader):
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

    def get_code(self, fullname: str):
        get_code = getattr(self.wrapped_loader, "get_code", None)
        if get_code is None:
            raise ImportError(f"Loader for {fullname!r} does not provide executable code.")
        return get_code(fullname)

    def get_source(self, fullname: str):
        get_source = getattr(self.wrapped_loader, "get_source", None)
        if get_source is None:
            return None
        return get_source(fullname)

    def is_package(self, fullname: str) -> bool:
        is_package = getattr(self.wrapped_loader, "is_package", None)
        if is_package is None:
            return False
        return bool(is_package(fullname))


class _FewShotTupleLabelsPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _FewShotTupleLabelsPatchLoader):
            return spec
        spec.loader = _FewShotTupleLabelsPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install tuple-label handling for few-shot target calibration."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _FewShotTupleLabelsPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)


__all__ = ["install"]

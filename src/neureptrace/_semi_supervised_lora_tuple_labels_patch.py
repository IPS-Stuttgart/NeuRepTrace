"""Runtime patch for composite labels in semi-supervised LoRA few-shot decoding."""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from collections.abc import Sequence
from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.decoding.semi_supervised_lora_few_shot"
_PATCH_MARKER = "_neureptrace_semi_supervised_lora_tuple_labels_patch_installed"
_FINDER_MARKER = "_neureptrace_semi_supervised_lora_tuple_labels_finder"


def _coerce_label(value: Any) -> Any:
    """Keep one row-level composite value atomic and hashable where possible."""

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return tuple(value.tolist())
    if isinstance(value, list):
        return tuple(value)
    return value


def _as_label_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    """Return a one-dimensional object vector without expanding tuple labels."""

    if isinstance(values, np.ndarray):
        if values.ndim == 0:
            items = [values.item()]
        elif values.ndim == 1:
            items = [_coerce_label(item) for item in values.tolist()]
        elif values.dtype == object:
            items = [_coerce_label(row) for row in values.tolist()]
        else:
            raise ValueError(f"{name} must be one-dimensional.")
    elif isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            items = [_coerce_label(item) for item in list(values)]
        except TypeError as exc:
            raise ValueError(f"{name} must be one-dimensional.") from exc

    vector = np.empty(len(items), dtype=object)
    vector[:] = items
    return vector


def _values_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
    except Exception:
        return False
    if isinstance(equal, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(equal)


def _unique_in_order(values: np.ndarray) -> list[Any]:
    unique: list[Any] = []
    for value in values.tolist():
        if not any(_values_equal(value, seen) for seen in unique):
            unique.append(value)
    return unique


def _matching_positions(values: np.ndarray, target: Any) -> np.ndarray:
    return np.asarray([index for index, value in enumerate(values.tolist()) if _values_equal(value, target)], dtype=int)


def _patch_module(module: ModuleType) -> None:
    if getattr(module, _PATCH_MARKER, False):
        return

    original_classifier_fit = module.SemiSupervisedLoRAFewShotClassifier.fit
    original_fit_decoder = module.fit_semi_supervised_lora_few_shot_decoder

    def _source_episode_indices(
        *,
        labels: np.ndarray,
        groups: np.ndarray,
        n_classes: int,
        support_per_class: int,
        query_per_class: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        groups = _as_label_vector(groups, name="source_groups")
        unique_groups = _unique_in_order(groups)
        if not unique_groups:
            return None

        for group_position in rng.permutation(len(unique_groups)):
            group = unique_groups[int(group_position)]
            group_indices = _matching_positions(groups, group)
            support: list[int] = []
            query: list[int] = []
            ok = True
            for class_index in range(int(n_classes)):
                class_indices = group_indices[labels[group_indices] == class_index]
                needed = int(support_per_class) + int(query_per_class)
                if class_indices.shape[0] < needed:
                    ok = False
                    break
                chosen = rng.choice(class_indices, size=needed, replace=False)
                support.extend(chosen[:support_per_class].tolist())
                query.extend(chosen[support_per_class:].tolist())
            if ok:
                return np.asarray(support, dtype=int), np.asarray(query, dtype=int)
        return None

    @wraps(original_classifier_fit)
    def fit_classifier(self, source_features: Any, source_labels: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        source_labels = _as_label_vector(source_labels, name="source_labels")
        if "target_calibration_labels" in kwargs:
            kwargs["target_calibration_labels"] = _as_label_vector(
                kwargs["target_calibration_labels"], name="target_calibration_labels"
            )
        if kwargs.get("classes") is not None:
            kwargs["classes"] = _as_label_vector(kwargs["classes"], name="classes")
        if kwargs.get("source_groups") is not None:
            kwargs["source_groups"] = _as_label_vector(kwargs["source_groups"], name="source_groups")
        return original_classifier_fit(self, source_features, source_labels, *args, **kwargs)

    @wraps(original_fit_decoder)
    def fit_decoder(*args: Any, **kwargs: Any) -> Any:
        kwargs = dict(kwargs)
        for key in ("source_labels", "target_labels", "classes", "source_groups"):
            if kwargs.get(key) is not None:
                kwargs[key] = _as_label_vector(kwargs[key], name=key)
        return original_fit_decoder(*args, **kwargs)

    module._as_1d_object_array = _as_label_vector
    module._source_episode_indices = _source_episode_indices
    module.SemiSupervisedLoRAFewShotClassifier.fit = fit_classifier
    module.fit_semi_supervised_lora_few_shot_decoder = fit_decoder
    setattr(module, _PATCH_MARKER, True)


class _SemiSupervisedLoRATupleLabelsPatchLoader(importlib.abc.Loader):
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


class _SemiSupervisedLoRATupleLabelsPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _SemiSupervisedLoRATupleLabelsPatchLoader):
            return spec
        spec.loader = _SemiSupervisedLoRATupleLabelsPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install composite-label handling for semi-supervised LoRA few-shot decoding."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_module(loaded)

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _SemiSupervisedLoRATupleLabelsPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)


__all__ = ["install"]

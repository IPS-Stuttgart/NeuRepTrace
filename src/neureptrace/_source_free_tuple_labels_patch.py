"""Preserve composite class labels in source-free adaptation."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_source_free_tuple_labels_patch_installed"
_normalize_probability_rows: Any | None = None


def _as_label_vector(labels: Any, *, name: str) -> np.ndarray:
    if labels is None:
        raise ValueError(f"{name} must be supplied.")
    if isinstance(labels, np.ndarray):
        if labels.ndim == 0:
            values = [labels.item()]
        elif labels.ndim == 1:
            values = labels.tolist()
        elif labels.dtype == object:
            values = [tuple(row) for row in labels.tolist()]
        else:
            raise ValueError(f"{name} must be a one-dimensional sequence of class labels.")
    else:
        try:
            values = list(labels)
        except TypeError as exc:
            raise ValueError(f"{name} must be a sequence of class labels.") from exc
    vector = np.empty(len(values), dtype=object)
    vector[:] = values
    return vector


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        equal = left == right
    except Exception:
        return False
    if isinstance(equal, np.ndarray):
        return bool(np.array_equal(left, right))
    return bool(equal)


def _ensure_unique_labels(labels: np.ndarray, *, name: str) -> None:
    for left_index, left_label in enumerate(labels.tolist()):
        for right_label in labels.tolist()[left_index + 1 :]:
            if _labels_equal(left_label, right_label):
                raise ValueError(f"{name} must be unique.")


def _label_index(labels: np.ndarray, class_label: Any) -> int | None:
    for index, candidate in enumerate(labels.tolist()):
        if _labels_equal(candidate, class_label):
            return index
    return None


def _resolve_classes(model: Any, classes: np.ndarray | list[Any] | tuple[Any, ...] | None) -> np.ndarray:
    if classes is not None:
        resolved = _as_label_vector(classes, name="classes")
    elif hasattr(model, "classes_"):
        resolved = _as_label_vector(model.classes_, name="source_model.classes_")
    else:
        raise ValueError("classes must be supplied when source_model does not expose classes_.")
    if resolved.ndim != 1 or resolved.shape[0] < 2:
        raise ValueError("Source-free adaptation needs at least two classes.")
    _ensure_unique_labels(resolved, name="classes")
    return resolved


def _align_probability_columns(probabilities: np.ndarray, *, model_classes: np.ndarray, classes: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    model_classes = _as_label_vector(model_classes, name="source_model.classes_")
    classes = _as_label_vector(classes, name="classes")
    if probabilities.ndim != 2:
        raise ValueError("source_model probabilities must be a two-dimensional matrix.")
    if probabilities.shape[1] != model_classes.shape[0]:
        raise ValueError("source_model probability columns do not match source_model.classes_.")
    aligned = np.zeros((probabilities.shape[0], classes.shape[0]), dtype=float)
    for output_index, class_label in enumerate(classes.tolist()):
        input_index = _label_index(model_classes, class_label)
        if input_index is None:
            raise ValueError(f"source_model is missing requested class {class_label!r}.")
        aligned[:, output_index] = probabilities[:, input_index]
    if _normalize_probability_rows is None:
        raise RuntimeError("Source-free tuple-label patch has not been installed.")
    return _normalize_probability_rows(aligned)


def install() -> None:
    """Patch source-free adaptation class-label handling."""

    source_free = importlib.import_module("neureptrace.decoding.source_free")
    if getattr(source_free._resolve_classes, _PATCH_MARKER, False):
        return
    global _normalize_probability_rows
    _normalize_probability_rows = source_free._normalize_probability_rows
    setattr(_resolve_classes, _PATCH_MARKER, True)
    source_free._resolve_classes = _resolve_classes
    source_free._align_probability_columns = _align_probability_columns


__all__ = ["install"]

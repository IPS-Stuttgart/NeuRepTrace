"""Resolve tied temporal-smoothing top-k metrics with exact-k semantics."""

from __future__ import annotations

import importlib

import numpy as np

_PATCH_MARKER = "_neureptrace_temporal_smoothing_topk_tie_patch_installed"


def _stable_top_columns(probabilities: np.ndarray, *, k: int) -> np.ndarray:
    probability_matrix = np.asarray(probabilities, dtype=float)
    if probability_matrix.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix.")
    effective_k = min(max(int(k), 0), probability_matrix.shape[1])
    if effective_k == 0:
        return np.empty((probability_matrix.shape[0], 0), dtype=int)
    return np.argsort(-probability_matrix, axis=1, kind="mergesort")[:, :effective_k]


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    """Compute exact-k top-k accuracy with stable class-index tie handling."""

    label_indices = np.asarray(labels, dtype=int).reshape(-1)
    if label_indices.size == 0:
        return float("nan")
    top_columns = _stable_top_columns(probabilities, k=k)
    if top_columns.shape[0] != label_indices.size:
        raise ValueError("labels must have one entry per probability row.")
    return float(np.mean(np.any(top_columns == label_indices[:, None], axis=1)))


def _top_k_accuracy_from_label_values(probabilities: np.ndarray, labels: np.ndarray, label_values: tuple[int, ...], *, k: int) -> float:
    """Compute exact-k top-k accuracy for arbitrary integer class labels."""

    label_array = np.asarray(labels, dtype=int).reshape(-1)
    if label_array.size == 0:
        return float("nan")
    top_columns = _stable_top_columns(probabilities, k=k)
    if top_columns.shape[0] != label_array.size:
        raise ValueError("labels must have one entry per probability row.")
    top_labels = np.asarray(label_values, dtype=int)[top_columns]
    return float(np.mean(np.any(top_labels == label_array[:, None], axis=1)))


def install() -> None:
    """Patch temporal-smoothing top-k metrics to keep tied rows exact-k."""

    temporal_smoothing = importlib.import_module("neureptrace.temporal_smoothing")
    if getattr(temporal_smoothing, _PATCH_MARKER, False):
        return

    temporal_smoothing._top_k_accuracy = _top_k_accuracy
    temporal_smoothing._top_k_accuracy_from_label_values = _top_k_accuracy_from_label_values
    setattr(temporal_smoothing, _PATCH_MARKER, True)


__all__ = ["install"]

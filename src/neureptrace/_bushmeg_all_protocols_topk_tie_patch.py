"""Resolve tied all-protocol top-k prediction metrics with exact-k semantics."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_bushmeg_all_protocols_topk_tie_patch_installed"


def _normalize_k(k: Any) -> int:
    if isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be a positive integer.")
    try:
        k_float = float(k)
    except (TypeError, ValueError) as exc:
        raise ValueError("k must be a positive integer.") from exc
    if not np.isfinite(k_float) or k_float % 1.0 != 0.0 or k_float < 1.0:
        raise ValueError("k must be a positive integer.")
    return int(k_float)


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    """Compute exact-k top-k accuracy with stable class-index tie handling."""

    probability_matrix = np.asarray(probabilities, dtype=float)
    label_indices = np.asarray(labels, dtype=int).reshape(-1)
    if probability_matrix.size == 0:
        return float("nan")
    if probability_matrix.ndim != 2:
        raise ValueError("probabilities must be a two-dimensional matrix.")
    if label_indices.size != probability_matrix.shape[0]:
        raise ValueError("labels must have one entry per probability row.")
    if probability_matrix.shape[1] == 0:
        return float("nan")

    k_value = min(_normalize_k(k), probability_matrix.shape[1])
    if k_value >= probability_matrix.shape[1]:
        valid_labels = (0 <= label_indices) & (label_indices < probability_matrix.shape[1])
        return float(np.mean(valid_labels))

    top_k = np.argsort(-probability_matrix, axis=1, kind="mergesort")[:, :k_value]
    hits = np.any(top_k == label_indices[:, None], axis=1)
    return float(np.mean(hits))


def install() -> None:
    """Patch all-protocol prediction metric recomputation to keep top-k exact."""

    report_patch = importlib.import_module("neureptrace._bushmeg_all_protocols_report_protocol_labels_patch")
    report_patch.install()

    all_protocols = importlib.import_module("neureptrace.bushmeg_all_protocols")
    if getattr(all_protocols, _PATCH_MARKER, False):
        return

    metric_patch = importlib.import_module("neureptrace._bushmeg_all_protocols_prediction_metric_patch")
    metric_patch._top_k_accuracy = _top_k_accuracy
    all_protocols._top_k_accuracy = _top_k_accuracy
    setattr(all_protocols, _PATCH_MARKER, True)


__all__ = ["install"]

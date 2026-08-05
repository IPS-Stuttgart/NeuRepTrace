"""Public API for progressive sequence-level neural fine-tuning.

The implementation is split into small internal modules; this facade preserves a
single stable import path for data packing, nested calibration, model fitting,
and permutation-constrained decoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from neureptrace.decoding._progressive_sequence_core import (
    PROGRESSIVE_SEQUENCE_CATEGORY,
    PROGRESSIVE_SEQUENCE_PROTOCOL,
    NestedTrialCalibrationSplit,
    PackedTrialEvents,
    PermutationDecodingResult,
    ProgressiveSequenceCalibrationResult,
    _as_feature_tensor,
    _as_label_matrix,
    _as_object_vector,
    pack_complete_trial_events,
    permutation_constrained_decode,
    select_nested_trial_calibration_splits,
    sinkhorn_trial_probabilities,
)
from neureptrace.decoding._progressive_sequence_model import TorchProgressiveSequenceClassifier

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

__all__ = (
    "PROGRESSIVE_SEQUENCE_CATEGORY",
    "PROGRESSIVE_SEQUENCE_PROTOCOL",
    "NestedTrialCalibrationSplit",
    "PackedTrialEvents",
    "PermutationDecodingResult",
    "ProgressiveSequenceCalibrationResult",
    "TorchProgressiveSequenceClassifier",
    "fit_progressive_sequence_target_calibrated_decoder",
    "pack_complete_trial_events",
    "permutation_constrained_decode",
    "select_nested_trial_calibration_splits",
    "sinkhorn_trial_probabilities",
)


def fit_progressive_sequence_target_calibrated_decoder(
    *,
    source_features: Sequence | np.ndarray,
    source_labels: Sequence | np.ndarray,
    target_features: Sequence | np.ndarray,
    target_labels: Sequence | np.ndarray,
    split: NestedTrialCalibrationSplit,
    source_subjects: Sequence[Any] | np.ndarray | None = None,
    target_strata: Sequence[Any] | np.ndarray | None = None,
    **model_kwargs: Any,
) -> ProgressiveSequenceCalibrationResult:
    """Fit on source plus labeled target calibration trials and score evaluation trials."""

    source_x = _as_feature_tensor(source_features, name="source_features")
    source_y = _as_label_matrix(source_labels, name="source_labels")
    target_x = _as_feature_tensor(target_features, name="target_features")
    target_y = _as_label_matrix(target_labels, name="target_labels")
    if source_x.shape[1:] != target_x.shape[1:] or source_y.shape[1] != target_y.shape[1]:
        raise ValueError("Source and target tensors must have matching event and feature dimensions.")
    if source_y.shape != source_x.shape[:2] or target_y.shape != target_x.shape[:2]:
        raise ValueError("Label matrices must match their feature tensors.")
    calibration_indices = np.asarray(split.calibration_indices, dtype=int).reshape(-1)
    evaluation_indices = np.asarray(split.evaluation_indices, dtype=int).reshape(-1)
    if calibration_indices.size == 0 or evaluation_indices.size == 0:
        raise ValueError("Both calibration and evaluation trial sets must be non-empty.")
    if np.intersect1d(calibration_indices, evaluation_indices).size:
        raise ValueError("Calibration and evaluation trial indices must be disjoint.")
    if np.any(calibration_indices < 0) or np.any(evaluation_indices < 0) or np.any(calibration_indices >= target_x.shape[0]) or np.any(evaluation_indices >= target_x.shape[0]):
        raise ValueError("Split contains an out-of-range target trial index.")

    target_stratum_vector = None if target_strata is None else _as_object_vector(target_strata, name="target_strata")
    if target_stratum_vector is not None and target_stratum_vector.shape[0] != target_x.shape[0]:
        raise ValueError("target_strata must contain one value per target trial.")
    seed = model_kwargs.get("random_state", split.seed)
    model_kwargs = {**model_kwargs, "random_state": seed}
    model = TorchProgressiveSequenceClassifier(**model_kwargs)
    model.fit_source(source_x, source_y, source_subjects=source_subjects)
    model.adapt_target(
        target_x[calibration_indices],
        target_y[calibration_indices],
        target_strata=None if target_stratum_vector is None else target_stratum_vector[calibration_indices],
    )
    probabilities = model.predict_proba(target_x[evaluation_indices], constrained=False)
    independent_encoded = np.argmax(probabilities, axis=2)
    if model.enforce_permutation_labels:
        constrained = permutation_constrained_decode(
            probabilities,
            temperature=model.sinkhorn_temperature,
            sinkhorn_iterations=model.sinkhorn_iterations,
        )
        constrained_probabilities = constrained.probabilities
        prediction_encoded = constrained.assignments
    else:
        constrained_probabilities = probabilities.copy()
        prediction_encoded = independent_encoded
    metadata = model.metadata()
    metadata.update(
        {
            "progressive_sequence_n_source_trials": int(source_x.shape[0]),
            "progressive_sequence_n_target_trials": int(target_x.shape[0]),
            "progressive_sequence_n_target_calibration_trials": int(calibration_indices.size),
            "progressive_sequence_n_target_evaluation_trials": int(evaluation_indices.size),
            "progressive_sequence_calibration_per_stratum": int(split.per_stratum),
            "progressive_sequence_max_calibration_per_stratum": int(split.max_per_stratum),
            "progressive_sequence_calibration_seed": int(split.seed),
        }
    )
    return ProgressiveSequenceCalibrationResult(
        model=model,
        probabilities=probabilities,
        constrained_probabilities=constrained_probabilities,
        predictions=model.classes_[prediction_encoded],
        independent_predictions=model.classes_[independent_encoded],
        evaluation_indices=evaluation_indices.copy(),
        calibration_indices=calibration_indices.copy(),
        metadata=metadata,
    )

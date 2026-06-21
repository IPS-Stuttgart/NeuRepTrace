"""Category-2 calibration-run alignment for held-out-subject decoding.

This module implements the clean hyperalignment/SRM-style protocol where a
source common space is fitted from source subjects' unlabeled calibration-run
anchors and the held-out target subject's projection is fitted from a disjoint
unlabeled calibration run.  Downstream classifiers should be trained only on the
aligned source decoding rows and their source labels.

The protocol uses ``X_s, y_s, X_t^calib`` and source calibration anchors, but it
never uses held-out target decoding labels or target class prototypes.  It is
therefore an unlabeled target-adaptive/category-2 protocol, not strict
source-only/category 1 and not supervised target calibration/category 3.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neureptrace.decoding.hyperalignment_initialization import (
    fit_class_hyperalignment,
    fit_projection_to_hyperalignment,
    transform_with_projection,
)
from neureptrace.decoding.mcca import fit_class_mcca
from neureptrace.decoding.mcca_target import class_alignment_matrix, fit_target_mcca_projection
from neureptrace.decoding.sampling import DEFAULT_CLASS_LIMIT_SEED, DEFAULT_CLASS_LIMIT_SELECTION
from neureptrace.decoding.source_alignment import normalize_alignment_components, normalize_alignment_repetition_cap

CATEGORY2_UNLABELED_CALIBRATION_ALIGNMENT = "unlabeled_target_calibration_alignment"
UNLABELED_CALIBRATION_ALIGNMENT_METHODS = ("procrustes", "hyperalignment", "mcca")
UNLABELED_CALIBRATION_ANCHOR_MODES = (
    "anchor_mean",
    "anchor_repetition",
    "stimulus_id_mean",
    "stimulus_id_repetition",
    "event_code_mean",
    "event_code_repetition",
    "run_event_index_within_stimulus",
)


@dataclass(frozen=True, slots=True)
class UnlabeledCalibrationAlignmentConfig:
    """Configuration for category-2 calibration-run common-space alignment."""

    method: str = "hyperalignment"
    anchor_mode: str = "stimulus_id_mean"
    repetition_cap: int | None = None
    repetition_selection: str = DEFAULT_CLASS_LIMIT_SELECTION
    repetition_seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED
    components: int | float = 64
    hyperalignment_iterations: int = 10
    mcca_regularization: float = 1e-6
    mcca_subject_pca_components: int | float | None = None

    @property
    def sample_mode(self) -> str:
        """Return the row-construction mode expected by M-CCA/hyperalignment helpers."""

        return _sample_mode_from_anchor_mode(self.anchor_mode)


@dataclass(frozen=True, slots=True)
class UnlabeledCalibrationAlignmentResult:
    """Aligned source/target feature matrices and protocol metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    metadata: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    model: Any | None = field(default=None, repr=False, compare=False)
    target_projection: Any | None = field(default=None, repr=False, compare=False)


def unlabeled_calibration_alignment_config(
    *,
    method: str = "hyperalignment",
    anchor_mode: str = "stimulus_id_mean",
    repetition_cap: int | str | None = None,
    repetition_selection: str = DEFAULT_CLASS_LIMIT_SELECTION,
    repetition_seed: int | str | None = DEFAULT_CLASS_LIMIT_SEED,
    components: int | float | str | None = 64,
    hyperalignment_iterations: int | str = 10,
    mcca_regularization: float | str = 1e-6,
    mcca_subject_pca_components: int | float | str | None = None,
) -> UnlabeledCalibrationAlignmentConfig:
    """Build a normalized category-2 calibration-run alignment config."""

    subject_pca_components = None
    if mcca_subject_pca_components not in {None, "", "none", "None"}:
        subject_pca_components = normalize_alignment_components(mcca_subject_pca_components)
    return UnlabeledCalibrationAlignmentConfig(
        method=_normalize_method(method),
        anchor_mode=_normalize_anchor_mode(anchor_mode),
        repetition_cap=normalize_alignment_repetition_cap(repetition_cap),
        repetition_selection=str(repetition_selection),
        repetition_seed=repetition_seed,
        components=normalize_alignment_components(components),
        hyperalignment_iterations=_normalize_integer(hyperalignment_iterations, name="hyperalignment_iterations", minimum=1),
        mcca_regularization=_normalize_nonnegative_float(mcca_regularization, name="mcca_regularization"),
        mcca_subject_pca_components=subject_pca_components,
    )


def align_train_test_with_unlabeled_calibration(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence[Any] | np.ndarray,
    train_subject_ids: Sequence[Hashable] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    source_calibration_features: Sequence[Sequence[float]] | np.ndarray,
    source_calibration_subject_ids: Sequence[Hashable] | np.ndarray,
    source_calibration_anchor_values: Sequence[Any] | np.ndarray,
    target_calibration_features: Sequence[Sequence[float]] | np.ndarray,
    target_calibration_anchor_values: Sequence[Any] | np.ndarray,
    config: UnlabeledCalibrationAlignmentConfig | None = None,
) -> UnlabeledCalibrationAlignmentResult:
    """Align decoding rows using a separate unlabeled source/target calibration run.

    Parameters
    ----------
    train_features, train_labels, train_subject_ids:
        Labeled source decoding rows used by the downstream classifier after
        projection into the common space.
    test_features:
        Held-out target decoding rows to score after fitting the target projection
        from ``target_calibration_features``.
    source_calibration_features, source_calibration_subject_ids, source_calibration_anchor_values:
        Source-subject calibration-run rows.  Anchors should identify shared,
        label-free calibration events such as movie frames, stimulus ids, or
        time bins.  These rows fit the source common space.
    target_calibration_features, target_calibration_anchor_values:
        Disjoint target-subject calibration-run rows with the same label-free
        anchors.  These rows fit only the held-out target projection.
    config:
        Normalized configuration.  If omitted, hyperalignment with stimulus-id
        mean anchors is used.
    """

    cfg = unlabeled_calibration_alignment_config() if config is None else config
    train_matrix = _feature_matrix(train_features, name="train_features")
    test_matrix = _feature_matrix(test_features, name="test_features")
    train_label_vector = _object_vector(train_labels, expected_length=train_matrix.shape[0], name="train_labels")
    train_subject_vector = _object_vector(train_subject_ids, expected_length=train_matrix.shape[0], name="train_subject_ids")
    source_calibration_matrix = _feature_matrix(source_calibration_features, name="source_calibration_features")
    source_calibration_subject_vector = _object_vector(
        source_calibration_subject_ids,
        expected_length=source_calibration_matrix.shape[0],
        name="source_calibration_subject_ids",
    )
    source_anchor_vector = _object_vector(
        source_calibration_anchor_values,
        expected_length=source_calibration_matrix.shape[0],
        name="source_calibration_anchor_values",
    )
    target_calibration_matrix = _feature_matrix(target_calibration_features, name="target_calibration_features")
    target_anchor_vector = _object_vector(
        target_calibration_anchor_values,
        expected_length=target_calibration_matrix.shape[0],
        name="target_calibration_anchor_values",
    )

    _check_same_width(train_matrix, test_matrix, left_name="train_features", right_name="test_features")
    _check_same_width(train_matrix, source_calibration_matrix, left_name="train_features", right_name="source_calibration_features")
    _check_same_width(train_matrix, target_calibration_matrix, left_name="train_features", right_name="target_calibration_features")
    _reject_missing_anchors(source_anchor_vector, name="source_calibration_anchor_values")
    _reject_missing_anchors(target_anchor_vector, name="target_calibration_anchor_values")
    if train_label_vector.shape[0] != train_matrix.shape[0]:  # pragma: no cover - guarded by _object_vector
        raise ValueError("train_labels must match train_features rows.")

    source_subjects = tuple(dict.fromkeys(source_calibration_subject_vector.tolist()))
    if len(source_subjects) < 2:
        raise ValueError("Unlabeled calibration alignment requires at least two source calibration subjects.")
    source_feature_blocks = {
        subject_id: source_calibration_matrix[source_calibration_subject_vector == subject_id]
        for subject_id in source_subjects
    }
    source_anchor_blocks = {
        subject_id: source_anchor_vector[source_calibration_subject_vector == subject_id]
        for subject_id in source_subjects
    }
    missing_train_subjects = [subject_id for subject_id in tuple(dict.fromkeys(train_subject_vector.tolist())) if subject_id not in source_feature_blocks]
    if missing_train_subjects:
        raise ValueError(
            "Every source decoding subject must have calibration-run rows. "
            f"Missing source calibration subjects: {missing_train_subjects!r}."
        )

    sample_mode = cfg.sample_mode
    n_repetitions = cfg.repetition_cap if sample_mode == "class_repetition" else None
    if cfg.method in {"procrustes", "hyperalignment"}:
        iterations = 1 if cfg.method == "procrustes" else cfg.hyperalignment_iterations
        model, alignment = fit_class_hyperalignment(
            source_feature_blocks,
            source_anchor_blocks,
            sample_mode=sample_mode,
            n_repetitions_per_class=n_repetitions,
            repetition_selection=cfg.repetition_selection,
            repetition_seed=cfg.repetition_seed,
            n_components=cfg.components,
            n_iterations=iterations,
            initialization="mean" if cfg.method == "procrustes" else "pca",
        )
        target_anchors = _target_anchor_matrix(target_calibration_matrix, target_anchor_vector, alignment=alignment)
        target_projection = fit_projection_to_hyperalignment(target_anchors, template=model.template)
        transformed_test = transform_with_projection(test_matrix, target_projection)
        target_transform_type = "unlabeled_calibration_template_procrustes"
    elif cfg.method == "mcca":
        model, alignment = fit_class_mcca(
            source_feature_blocks,
            source_anchor_blocks,
            sample_mode=sample_mode,
            n_repetitions_per_class=n_repetitions,
            repetition_selection=cfg.repetition_selection,
            repetition_seed=cfg.repetition_seed,
            n_components=cfg.components,
            regularization=cfg.mcca_regularization,
            subject_pca_components=cfg.mcca_subject_pca_components,
        )
        target_anchors = _target_anchor_matrix(target_calibration_matrix, target_anchor_vector, alignment=alignment)
        target_projection = fit_target_mcca_projection(target_anchors, model, regularization=cfg.mcca_regularization)
        transformed_test = target_projection.transform(test_matrix)
        target_transform_type = "unlabeled_calibration_template_ridge_least_squares"
    else:  # pragma: no cover - guarded by config normalization
        raise ValueError(f"Unsupported unlabeled calibration alignment method: {cfg.method}")

    transformed_train = np.empty((train_matrix.shape[0], int(model.n_components)), dtype=float)
    for subject_id in tuple(dict.fromkeys(train_subject_vector.tolist())):
        mask = train_subject_vector == subject_id
        transformed_train[mask] = model.transform(subject_id, train_matrix[mask])

    metadata = _metadata(
        cfg,
        alignment=alignment,
        model=model,
        train_matrix=train_matrix,
        test_matrix=test_matrix,
        source_calibration_matrix=source_calibration_matrix,
        target_calibration_matrix=target_calibration_matrix,
        source_subject_count=len(source_subjects),
        target_alignment_rows=int(target_projection.n_alignment_rows),
        target_transform_type=target_transform_type,
    )
    diagnostics = {
        "alignment_method": cfg.method,
        "alignment_protocol": CATEGORY2_UNLABELED_CALIBRATION_ALIGNMENT,
        "sample_mode": sample_mode,
        "n_source_subjects": len(source_subjects),
        "n_anchor_values": int(alignment.n_classes),
        "n_alignment_rows": int(alignment.n_alignment_rows),
        "requested_components": cfg.components,
        "actual_components": int(model.n_components),
        "feature_dim": int(train_matrix.shape[1]),
        "decode_feature_dim": int(transformed_test.shape[1]),
        "uses_unlabeled_target_data": True,
        "target_transform_type": target_transform_type,
        "target_calibration_rows": int(target_calibration_matrix.shape[0]),
        "target_alignment_rows": int(target_projection.n_alignment_rows),
    }
    return UnlabeledCalibrationAlignmentResult(
        train_features=transformed_train,
        test_features=transformed_test,
        metadata=metadata,
        diagnostics=diagnostics,
        model=model,
        target_projection=target_projection,
    )


def _target_anchor_matrix(target_calibration_matrix: np.ndarray, target_anchor_vector: np.ndarray, *, alignment: Any) -> np.ndarray:
    repetition_selection = alignment.repetition_selection if alignment.repetition_selection is not None else DEFAULT_CLASS_LIMIT_SELECTION
    repetition_seed = alignment.repetition_seed if alignment.repetition_seed is not None else DEFAULT_CLASS_LIMIT_SEED
    return class_alignment_matrix(
        target_calibration_matrix,
        target_anchor_vector,
        classes=alignment.classes,
        sample_mode=alignment.sample_mode,
        n_repetitions_per_class=alignment.n_repetitions_per_class,
        repetition_selection=repetition_selection,
        repetition_seed=repetition_seed,
        selected_offsets_by_class=alignment.selected_offsets_by_class,
    )


def _metadata(
    config: UnlabeledCalibrationAlignmentConfig,
    *,
    alignment: Any,
    model: Any,
    train_matrix: np.ndarray,
    test_matrix: np.ndarray,
    source_calibration_matrix: np.ndarray,
    target_calibration_matrix: np.ndarray,
    source_subject_count: int,
    target_alignment_rows: int,
    target_transform_type: str,
) -> dict[str, Any]:
    return {
        "alignment_method": config.method,
        "alignment_anchor_mode": config.anchor_mode,
        "alignment_sample_mode": alignment.sample_mode,
        "alignment_repetition_cap": "" if config.repetition_cap is None else int(config.repetition_cap),
        "alignment_components": config.components,
        "alignment_protocol": CATEGORY2_UNLABELED_CALIBRATION_ALIGNMENT,
        "alignment_protocol_note": "uses disjoint unlabeled target calibration anchors; category-2 calibration-run alignment",
        "alignment_target_projection": CATEGORY2_UNLABELED_CALIBRATION_ALIGNMENT,
        "alignment_target_projection_fit": target_transform_type,
        "alignment_n_components": int(model.n_components),
        "alignment_n_source_subjects": int(source_subject_count),
        "alignment_n_anchor_values": int(alignment.n_classes),
        "alignment_common_anchor_count": int(alignment.n_classes),
        "alignment_anchor_rows_used": int(alignment.n_alignment_rows),
        "alignment_source_decoding_rows": int(train_matrix.shape[0]),
        "alignment_source_calibration_rows": int(source_calibration_matrix.shape[0]),
        "alignment_target_decoding_rows": int(test_matrix.shape[0]),
        "alignment_target_calibration_rows": int(target_calibration_matrix.shape[0]),
        "alignment_target_alignment_rows": int(target_alignment_rows),
        "alignment_uses_unlabeled_target_data": True,
        "alignment_target_labels_used": False,
        "alignment_target_pseudo_labels_used": False,
        "alignment_target_anchor_values_used": True,
        "alignment_target_calibrated": False,
        "alignment_unlabeled_target_calibrated": True,
        "alignment_oracle_target_calibrated": False,
        "alignment_pseudo_label_target_calibrated": False,
        "alignment_debug_upper_bound": False,
        "alignment_strict_source_only": False,
        "alignment_valid_for_benchmark": True,
        "alignment_valid_for_strict_source_only": False,
    }


def _normalize_method(method: str) -> str:
    normalized = str(method).strip().lower().replace("-", "_")
    normalized = {"srm": "mcca", "multiset_cca": "mcca", "multiway_cca": "mcca"}.get(normalized, normalized)
    if normalized not in UNLABELED_CALIBRATION_ALIGNMENT_METHODS:
        raise ValueError(
            f"Unknown unlabeled calibration alignment method {method!r}. "
            f"Available methods: {', '.join(UNLABELED_CALIBRATION_ALIGNMENT_METHODS)}."
        )
    return normalized


def _normalize_anchor_mode(anchor_mode: str) -> str:
    normalized = str(anchor_mode).strip().lower().replace("-", "_")
    if normalized not in UNLABELED_CALIBRATION_ANCHOR_MODES:
        raise ValueError(
            f"Unknown unlabeled calibration anchor mode {anchor_mode!r}. "
            f"Available modes: {', '.join(UNLABELED_CALIBRATION_ANCHOR_MODES)}."
        )
    return normalized


def _sample_mode_from_anchor_mode(anchor_mode: str) -> str:
    normalized = _normalize_anchor_mode(anchor_mode)
    if normalized in {"anchor_mean", "stimulus_id_mean", "event_code_mean"}:
        return "class_mean"
    return "class_repetition"


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row.")
    if matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


def _object_vector(values: Sequence[Any] | np.ndarray, *, expected_length: int | None, name: str) -> np.ndarray:
    if isinstance(values, np.ndarray) and values.ndim == 1:
        vector = values.astype(object, copy=False).reshape(-1)
    elif isinstance(values, np.ndarray):
        rows = [tuple(row.tolist()) for row in np.asarray(values, dtype=object).reshape(values.shape[0], -1)]
        vector = np.empty(len(rows), dtype=object)
        vector[:] = rows
    else:
        items = list(values)
        vector = np.empty(len(items), dtype=object)
        vector[:] = items
    if expected_length is not None and vector.shape[0] != expected_length:
        raise ValueError(f"{name} length must match feature rows: {vector.shape[0]} != {expected_length}.")
    return vector


def _check_same_width(left: np.ndarray, right: np.ndarray, *, left_name: str, right_name: str) -> None:
    if left.shape[1] != right.shape[1]:
        raise ValueError(f"{left_name} and {right_name} must have the same feature width: {left.shape[1]} != {right.shape[1]}.")


def _reject_missing_anchors(values: np.ndarray, *, name: str) -> None:
    missing = []
    for index, value in enumerate(values.tolist()):
        if value is None:
            missing.append(index)
        elif isinstance(value, str) and value.strip().lower() in {"", "na", "n/a", "nan", "none", "null", "<na>", "<nat>", "nat"}:
            missing.append(index)
        elif isinstance(value, (float, np.floating)) and np.isnan(value):
            missing.append(index)
    if missing:
        preview = ", ".join(str(index) for index in missing[:5])
        raise ValueError(f"{name} contains missing anchor values at rows: {preview}.")


def _normalize_integer(value: int | str, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not np.isfinite(numeric) or numeric % 1.0 != 0.0:
        raise ValueError(f"{name} must be an integer.")
    integer = int(numeric)
    if minimum is not None and integer < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return integer


def _normalize_nonnegative_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite and non-negative.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite and non-negative.") from exc
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return numeric


__all__ = [
    "CATEGORY2_UNLABELED_CALIBRATION_ALIGNMENT",
    "UNLABELED_CALIBRATION_ALIGNMENT_METHODS",
    "UNLABELED_CALIBRATION_ANCHOR_MODES",
    "UnlabeledCalibrationAlignmentConfig",
    "UnlabeledCalibrationAlignmentResult",
    "align_train_test_with_unlabeled_calibration",
    "unlabeled_calibration_alignment_config",
]

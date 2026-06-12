"""Feature alignment helpers for source-only and target-calibrated protocols.

The default supervised alignment protocol fits anchors from source subjects
only. Held-out target rows are transformed with the source-fitted group
projection. The ``target_calibrated_alignment`` protocol fits a target
projection from disjoint target calibration rows before scoring separate target
evaluation rows. The ``pseudo_label_target_calibrated_alignment`` protocol fits
that target projection from classifier-generated pseudo labels. The explicit
``oracle_target_calibrated_alignment`` debug protocol is the only mode that
accepts scored held-out target labels or anchors, and it is reported as an upper
bound that is not valid for benchmark claims.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neureptrace.decoding.hyperalignment_initialization import fit_class_hyperalignment
from neureptrace.decoding.hyperalignment_initialization import fit_projection_to_hyperalignment
from neureptrace.decoding.hyperalignment_initialization import transform_with_projection
from neureptrace.decoding.mcca import fit_class_mcca
from neureptrace.decoding.sampling import DEFAULT_CLASS_LIMIT_SEED, DEFAULT_CLASS_LIMIT_SELECTION, select_class_limited_indices

SOURCE_ALIGNMENT_CLASS_ANCHORED_METHODS = ("procrustes", "hyperalignment", "mcca")
SOURCE_ALIGNMENT_UNSUPERVISED_METHODS = (
    "euclidean",
    "coral",
    "target_baseline_covariance",
    "subject_sensor_covariance",
)
SOURCE_ALIGNMENT_METHODS = ("none", *SOURCE_ALIGNMENT_CLASS_ANCHORED_METHODS, *SOURCE_ALIGNMENT_UNSUPERVISED_METHODS)
SOURCE_ALIGNMENT_CLASS_ANCHOR_MODES = ("class_mean", "class_repetition")
SOURCE_ALIGNMENT_STIMULUS_ANCHOR_MODES = (
    "stimulus_id_mean",
    "stimulus_id_repetition",
    "event_code_mean",
    "run_event_index_within_stimulus",
)
SOURCE_ALIGNMENT_ANCHOR_MODES = (*SOURCE_ALIGNMENT_CLASS_ANCHOR_MODES, *SOURCE_ALIGNMENT_STIMULUS_ANCHOR_MODES)
TARGET_CALIBRATED_ALIGNMENT = "target_calibrated_alignment"
ORACLE_TARGET_CALIBRATED_ALIGNMENT = "oracle_target_calibrated_alignment"
PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT = "pseudo_label_target_calibrated_alignment"
SOURCE_ALIGNMENT_TARGET_PROJECTIONS = (
    "group_projection",
    TARGET_CALIBRATED_ALIGNMENT,
    ORACLE_TARGET_CALIBRATED_ALIGNMENT,
    PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT,
)
DEFAULT_ALIGNMENT_TIMES = (0.088, 0.136, 0.184, 0.232, 0.280)
SAME_DECODE_WINDOW_ALIGNMENT = "same_decode_window"
SAME_DECODE_WINDOW_ALIGNMENT_ALIASES = {
    "same_decode_window",
    "same_window",
    "decode_window",
    "current_decode_window",
    "same-time",
    "same_time",
}
DEFAULT_ALIGNMENT_REPETITION_CAP = 16
DEFAULT_ALIGNMENT_COMPONENTS = 64
MAX_FULL_COVARIANCE_FEATURES = 256
MIN_COVARIANCE_EIGENVALUE = 1e-6
ALIGNMENT_DIAGNOSTIC_COLUMNS = (
    "dataset",
    "test_subject",
    "alignment_method",
    "sample_mode",
    "n_source_subjects",
    "n_classes",
    "n_alignment_rows",
    "n_repetitions_per_class",
    "requested_components",
    "actual_components",
    "feature_dim",
    "decode_feature_dim",
    "alignment_window_center",
    "alignment_window_size",
    "decode_window_center",
    "decode_window_size",
    "uses_channel_projection_collapse",
    "alignment_dimensionality_reduction",
    "anchor_row_correlation_before",
    "anchor_row_correlation_after",
    "source_inner_decoding_before_alignment",
    "source_inner_decoding_after_alignment",
    "source_inner_raw_balanced_accuracy",
    "source_inner_aligned_balanced_accuracy",
    "source_inner_aligned_minus_raw",
    "source_inner_validation_type",
    "uses_unlabeled_target_data",
    "covariance_alignment_estimator",
    "target_transform_type",
    "alignment_target_projection",
    "alignment_target_projection_fit",
    "alignment_target_alignment_rows",
    "alignment_target_labels_used",
    "alignment_target_pseudo_labels_used",
    "alignment_target_anchor_values_used",
    "alignment_target_calibrated",
    "alignment_target_calibration_per_anchor",
    "alignment_target_calibration_seed",
    "alignment_oracle_target_calibrated",
    "alignment_pseudo_label_target_calibrated",
    "alignment_debug_upper_bound",
    "alignment_valid_for_benchmark",
    "alignment_protocol",
    "alignment_protocol_note",
)
ALIGNMENT_ANCHOR_AVAILABILITY_COLUMNS = (
    "dataset",
    "test_subject",
    "alignment_method",
    "alignment_anchor_mode",
    "alignment_anchor_column",
    "sample_mode",
    "alignment_target_projection",
    "alignment_protocol",
    "n_source_subjects",
    "n_source_rows",
    "source_anchor_value_source",
    "n_source_anchor_values",
    "n_common_source_anchors",
    "common_source_anchor_values_preview",
    "source_anchor_rows_total",
    "source_anchor_rows_retained",
    "source_anchor_rows_dropped",
    "min_source_unique_anchors_per_subject",
    "median_source_unique_anchors_per_subject",
    "max_source_unique_anchors_per_subject",
    "min_source_common_anchor_rows_per_subject",
    "median_source_common_anchor_rows_per_subject",
    "max_source_common_anchor_rows_per_subject",
    "estimated_alignment_rows",
    "estimated_repetitions_per_anchor",
    "target_anchor_values_used",
    "n_target_rows",
    "n_target_anchor_values",
    "target_missing_common_anchor_count",
    "target_missing_common_anchor_values_preview",
    "target_calibration_anchor_values_used",
    "n_target_calibration_rows",
    "n_target_calibration_anchor_values",
    "target_calibration_missing_common_anchor_count",
    "target_calibration_missing_common_anchor_values_preview",
    "prefit_status",
    "prefit_failure_reason",
    "alignment_window_center",
    "alignment_window_size",
    "decode_window_center",
    "decode_window_size",
)


@dataclass(frozen=True, slots=True)
class SourceAlignmentConfig:
    """Configuration for strict source-only common-space alignment."""

    method: str = "none"
    anchor_mode: str = "class_mean"
    anchor_column: str = ""
    repetition_cap: int | None = DEFAULT_ALIGNMENT_REPETITION_CAP
    components: int | float = DEFAULT_ALIGNMENT_COMPONENTS
    times: tuple[float, ...] = DEFAULT_ALIGNMENT_TIMES
    same_decode_window: bool = False
    target_projection: str = "group_projection"
    target_calibration_per_anchor: int = 1
    target_calibration_seed: int = 13
    hyperalignment_iterations: int = 10
    mcca_regularization: float = 1e-6
    mcca_subject_pca_components: int | float | None = None

    @property
    def enabled(self) -> bool:
        return self.method != "none"

    @property
    def oracle_target_calibrated(self) -> bool:
        return self.enabled and self.target_projection == ORACLE_TARGET_CALIBRATED_ALIGNMENT

    @property
    def target_calibrated(self) -> bool:
        return self.enabled and self.target_projection == TARGET_CALIBRATED_ALIGNMENT

    @property
    def pseudo_label_target_calibrated(self) -> bool:
        return self.enabled and self.target_projection == PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT

    @property
    def fits_target_projection(self) -> bool:
        return self.oracle_target_calibrated or self.target_calibrated or self.pseudo_label_target_calibrated

    def static_metadata(self) -> dict[str, Any]:
        oracle = self.oracle_target_calibrated
        target_calibrated = self.target_calibrated
        pseudo_label_target_calibrated = self.pseudo_label_target_calibrated
        unsupervised = _uses_unlabeled_covariance_alignment(self.method)
        return {
            "alignment_method": self.method,
            "alignment_anchor_mode": self.anchor_mode,
            "alignment_anchor_column": self.anchor_column,
            "alignment_repetition_cap": "" if self.repetition_cap is None else int(self.repetition_cap),
            "alignment_components": self.components,
            "alignment_times": (
                SAME_DECODE_WINDOW_ALIGNMENT
                if self.same_decode_window
                else "|".join(f"{time:.6g}" for time in self.times)
            ),
            "alignment_window_mode": SAME_DECODE_WINDOW_ALIGNMENT if self.same_decode_window else "fixed_centers",
            "alignment_same_decode_window": bool(self.same_decode_window),
            "alignment_target_projection": self.target_projection,
            "alignment_strict_source_only": bool(
                self.enabled and not oracle and not target_calibrated and not pseudo_label_target_calibrated and not unsupervised
            ),
            "alignment_uses_unlabeled_target_data": bool(unsupervised or pseudo_label_target_calibrated),
            "alignment_uses_class_labels": bool(
                self.method in SOURCE_ALIGNMENT_CLASS_ANCHORED_METHODS
                and self.anchor_mode in SOURCE_ALIGNMENT_CLASS_ANCHOR_MODES
            ),
            "alignment_target_calibrated": bool(target_calibrated),
            "alignment_target_calibration_per_anchor": int(self.target_calibration_per_anchor),
            "alignment_target_calibration_seed": int(self.target_calibration_seed),
            "alignment_oracle_target_calibrated": bool(oracle),
            "alignment_pseudo_label_target_calibrated": bool(pseudo_label_target_calibrated),
            "alignment_debug_upper_bound": bool(oracle),
            "alignment_valid_for_benchmark": bool(not oracle and not target_calibrated),
            "alignment_protocol": (
                ORACLE_TARGET_CALIBRATED_ALIGNMENT
                if oracle
                else TARGET_CALIBRATED_ALIGNMENT
                if target_calibrated
                else PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT
                if pseudo_label_target_calibrated
                else "unlabeled_target_covariance_alignment"
                if unsupervised
                else "strict_source_only"
            ),
            "alignment_protocol_note": (
                "debug upper bound only; not valid for benchmark"
                if oracle
                else "uses disjoint target calibration rows; not valid for strict source-only benchmark"
                if target_calibrated
                else "uses target features with classifier-generated pseudo labels; category-2 transductive adaptation"
                if pseudo_label_target_calibrated
                else ""
            ),
        }


@dataclass(frozen=True, slots=True)
class SourceAlignmentResult:
    """Aligned train/test feature matrices and fold-local metadata."""

    train_features: np.ndarray
    test_features: np.ndarray
    metadata: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _SourceAlignmentFit:
    model: Any
    alignment: Any
    transformed_by_subject: dict[Hashable, np.ndarray]
    anchor_filter_metadata: dict[str, Any]
    anchor_before: Mapping[Hashable, np.ndarray]
    anchor_after: dict[Hashable, np.ndarray]
    n_components: int


@dataclass(frozen=True, slots=True)
class _CovarianceStats:
    mean: np.ndarray
    sqrt: np.ndarray
    inv_sqrt: np.ndarray
    estimator: str


def normalize_source_alignment_method(method: str | None) -> str:
    normalized = "none" if method is None else str(method).strip().lower().replace("-", "_")
    if normalized in {"off", "false", "identity", "raw"}:
        normalized = "none"
    normalized = {
        "euclidean_alignment": "euclidean",
        "covariance_whitening": "euclidean",
        "whitening": "euclidean",
        "coral_alignment": "coral",
        "coral_covariance": "coral",
        "target_covariance": "target_baseline_covariance",
        "target_covariance_alignment": "target_baseline_covariance",
        "target_baseline_covariance_alignment": "target_baseline_covariance",
        "subject_covariance": "subject_sensor_covariance",
        "subject_covariance_normalization": "subject_sensor_covariance",
        "subject_wise_sensor_covariance_normalization": "subject_sensor_covariance",
        "sensor_covariance_normalization": "subject_sensor_covariance",
    }.get(normalized, normalized)
    if normalized not in SOURCE_ALIGNMENT_METHODS:
        raise ValueError(
            f"Unknown source alignment method {method!r}. "
            f"Available methods: {', '.join(SOURCE_ALIGNMENT_METHODS)}."
        )
    return normalized


def normalize_source_alignment_anchor_mode(anchor_mode: str | None) -> str:
    normalized = "class_mean" if anchor_mode is None else str(anchor_mode).strip().lower().replace("-", "_")
    if normalized not in SOURCE_ALIGNMENT_ANCHOR_MODES:
        raise ValueError(
            f"Unknown source alignment anchor mode {anchor_mode!r}. "
            f"Available modes: {', '.join(SOURCE_ALIGNMENT_ANCHOR_MODES)}."
        )
    return normalized


def normalize_source_alignment_target_projection(target_projection: str | None) -> str:
    normalized = "group_projection" if target_projection is None else str(target_projection).strip().lower().replace("-", "_")
    if normalized in {"target", "target_calibrated", "target_calibration", "calibrated_target"}:
        normalized = TARGET_CALIBRATED_ALIGNMENT
    if normalized in {"oracle", "oracle_target", "oracle_target_calibrated"}:
        normalized = ORACLE_TARGET_CALIBRATED_ALIGNMENT
    if normalized in {"pseudo", "pseudo_label", "pseudo_label_target", "pseudo_label_target_calibrated"}:
        normalized = PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT
    if normalized not in SOURCE_ALIGNMENT_TARGET_PROJECTIONS:
        raise ValueError(
            f"Unknown alignment target projection {target_projection!r}. "
            f"Available target projections: {', '.join(SOURCE_ALIGNMENT_TARGET_PROJECTIONS)}."
        )
    return normalized


def parse_alignment_times(times: Sequence[float] | str | None) -> tuple[tuple[float, ...], bool]:
    if times is None:
        return (), True
    if isinstance(times, str):
        text = times.strip()
        if not text:
            return (), True
        normalized = text.lower().replace("-", "_").replace(" ", "_")
        if normalized in SAME_DECODE_WINDOW_ALIGNMENT_ALIASES:
            return (), True
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [part.strip() for chunk in text.split(",") for part in chunk.split() if part.strip()]
        values = tuple(float(part) for part in parts)
    else:
        values = tuple(float(value) for value in times)
    if not values:
        raise ValueError("alignment_times must contain at least one time center.")
    if any(not np.isfinite(value) for value in values):
        raise ValueError("alignment_times must be finite.")
    return values, False


def normalize_alignment_repetition_cap(value: int | str | None) -> int | None:
    if value is None:
        return DEFAULT_ALIGNMENT_REPETITION_CAP
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            return DEFAULT_ALIGNMENT_REPETITION_CAP
        if text in {"none", "null", "all", "full"}:
            return None
        parsed = int(text)
    else:
        parsed = int(value)
    if parsed < 1:
        raise ValueError("alignment_repetition_cap must be positive, all, or none.")
    return parsed


def _normalize_target_calibration_per_anchor(value: int | str | None) -> int:
    if value is None:
        return 1
    if isinstance(value, str):
        text = value.strip().lower()
        parsed = 1 if text in {"", "default"} else int(text)
    else:
        parsed = int(value)
    if parsed < 1:
        raise ValueError("alignment_target_calibration_per_anchor must be positive.")
    return parsed


def normalize_alignment_components(value: int | float | str | None) -> int | float:
    if value is None:
        return DEFAULT_ALIGNMENT_COMPONENTS
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "default"}:
            return DEFAULT_ALIGNMENT_COMPONENTS
        if text in {"inf", "infinity", "all", "full"}:
            return float("inf")
        parsed: int | float = float(text) if any(marker in text for marker in (".", "e")) else int(text)
    else:
        parsed = value
    if isinstance(parsed, float):
        if parsed == float("inf"):
            return parsed
        if not np.isfinite(parsed) or parsed <= 0:
            raise ValueError("alignment_components must be positive.")
        if parsed.is_integer():
            return int(parsed)
        raise ValueError("alignment_components must be an integer count or infinity.")
    parsed = int(parsed)
    if parsed < 1:
        raise ValueError("alignment_components must be positive.")
    return parsed


def source_alignment_config(
    *,
    method: str | None = None,
    anchor_mode: str | None = None,
    anchor_column: str | None = None,
    repetition_cap: int | str | None = DEFAULT_ALIGNMENT_REPETITION_CAP,
    components: int | float | str | None = DEFAULT_ALIGNMENT_COMPONENTS,
    times: Sequence[float] | str | None = None,
    target_projection: str | None = "group_projection",
    target_calibration_per_anchor: int | str | None = 1,
    target_calibration_seed: int = 13,
    hyperalignment_iterations: int = 10,
    mcca_regularization: float = 1e-6,
    mcca_subject_pca_components: int | float | str | None = None,
) -> SourceAlignmentConfig:
    mcca_subject_components = None
    if mcca_subject_pca_components not in {None, "", "none", "None"}:
        mcca_subject_components = normalize_alignment_components(mcca_subject_pca_components)
    parsed_times, same_decode_window = parse_alignment_times(times)
    config = SourceAlignmentConfig(
        method=normalize_source_alignment_method(method),
        anchor_mode=normalize_source_alignment_anchor_mode(anchor_mode),
        anchor_column="" if anchor_column is None else str(anchor_column).strip(),
        repetition_cap=normalize_alignment_repetition_cap(repetition_cap),
        components=normalize_alignment_components(components),
        times=parsed_times,
        same_decode_window=same_decode_window,
        target_projection=normalize_source_alignment_target_projection(target_projection),
        target_calibration_per_anchor=_normalize_target_calibration_per_anchor(target_calibration_per_anchor),
        target_calibration_seed=int(target_calibration_seed),
        hyperalignment_iterations=int(hyperalignment_iterations),
        mcca_regularization=float(mcca_regularization),
        mcca_subject_pca_components=mcca_subject_components,
    )
    if config.hyperalignment_iterations < 1:
        raise ValueError("alignment hyperalignment_iterations must be positive.")
    if config.mcca_regularization < 0 or not np.isfinite(config.mcca_regularization):
        raise ValueError("alignment mcca_regularization must be finite and non-negative.")
    if _uses_unlabeled_covariance_alignment(config.method) and config.fits_target_projection:
        raise ValueError("Unsupervised covariance alignment uses unlabeled target data and does not support target-calibrated projections.")
    return config


def align_train_test_features(
    *,
    train_features: Sequence[Sequence[float]] | np.ndarray,
    train_labels: Sequence[Any] | np.ndarray,
    train_subject_ids: Sequence[Hashable] | np.ndarray,
    test_features: Sequence[Sequence[float]] | np.ndarray,
    config: SourceAlignmentConfig,
    target_labels: Sequence[Any] | np.ndarray | None = None,
    train_anchor_values: Sequence[Any] | np.ndarray | None = None,
    target_anchor_values: Sequence[Any] | np.ndarray | None = None,
    target_calibration_features: Sequence[Sequence[float]] | np.ndarray | None = None,
    target_calibration_labels: Sequence[Any] | np.ndarray | None = None,
    target_calibration_anchor_values: Sequence[Any] | np.ndarray | None = None,
    compute_source_inner_diagnostics: bool = True,
) -> SourceAlignmentResult:
    """Fit source-only alignment and transform train/test feature rows.

    ``target_labels`` exists as a guardrail. Strict source-only alignment rejects
    it rather than silently accepting held-out labels. The
    ``target_calibrated_alignment`` protocol accepts labels or anchors only for
    separate target calibration rows. The explicit
    ``oracle_target_calibrated_alignment`` debug mode accepts scored held-out
    target labels and marks the result as an upper bound that is not valid for
    benchmark reporting. Metadata-derived ``*_anchor_values`` let alignment use
    source stimulus or event identifiers without changing the decoder labels.
    """

    unsupervised_alignment = _uses_unlabeled_covariance_alignment(config.method)
    target_calibration_args = (
        target_calibration_features,
        target_calibration_labels,
        target_calibration_anchor_values,
    )
    if target_labels is not None and not config.oracle_target_calibrated:
        raise ValueError("Strict source-only alignment does not accept target labels.")
    if target_anchor_values is not None and not config.oracle_target_calibrated:
        raise ValueError("Strict source-only alignment does not accept target anchor values.")
    if any(value is not None for value in target_calibration_args) and not (
        config.target_calibrated or config.pseudo_label_target_calibrated
    ):
        raise ValueError(
            "Target calibration rows are accepted only with target_calibrated_alignment "
            "or pseudo_label_target_calibrated_alignment."
        )
    class_anchor_mode = _uses_decoder_label_anchors(config.anchor_mode)
    if unsupervised_alignment:
        train_anchor_source = "unlabeled_covariance"
    elif not config.enabled and train_anchor_values is None:
        train_anchor_source = "decoder_labels"
        train_anchor_values = train_labels
    elif class_anchor_mode:
        train_anchor_source = "decoder_labels"
        if train_anchor_values is None:
            train_anchor_values = train_labels
        if config.oracle_target_calibrated and target_anchor_values is None:
            target_anchor_values = target_labels
        if (config.target_calibrated or config.pseudo_label_target_calibrated) and target_calibration_anchor_values is None:
            target_calibration_anchor_values = target_calibration_labels
    else:
        train_anchor_source = "metadata"
        if train_anchor_values is None:
            raise ValueError(f"{config.anchor_mode} alignment requires train_anchor_values derived from metadata.")
    if config.oracle_target_calibrated and target_anchor_values is None:
        raise ValueError(
            "oracle_target_calibrated_alignment requires held-out target labels or anchor values and is not valid for benchmark reporting."
        )
    if (config.target_calibrated or config.pseudo_label_target_calibrated) and target_calibration_anchor_values is None:
        projection_name = (
            "pseudo_label_target_calibrated_alignment"
            if config.pseudo_label_target_calibrated
            else "target_calibrated_alignment"
        )
        raise ValueError(f"{projection_name} requires target calibration labels or anchor values.")

    train_matrix = _feature_matrix(train_features, name="train_features")
    test_matrix = _feature_matrix(test_features, name="test_features")
    target_calibration_matrix = (
        None
        if target_calibration_features is None
        else _feature_matrix(target_calibration_features, name="target_calibration_features")
    )
    train_vector = np.asarray(train_labels).reshape(-1)
    subject_vector = np.asarray(train_subject_ids, dtype=object).reshape(-1)
    target_vector = None if target_labels is None else np.asarray(target_labels).reshape(-1)
    train_anchor_vector = (
        np.empty(train_matrix.shape[0], dtype=object)
        if unsupervised_alignment
        else _anchor_vector(
            train_anchor_values,
            expected_length=train_matrix.shape[0],
            name="train_anchor_values",
        )
    )
    target_anchor_vector = (
        None
        if target_anchor_values is None
        else _anchor_vector(
            target_anchor_values,
            expected_length=test_matrix.shape[0],
            name="target_anchor_values",
        )
    )
    target_calibration_label_vector = (
        None if target_calibration_labels is None else np.asarray(target_calibration_labels).reshape(-1)
    )
    target_calibration_anchor_vector = (
        None
        if target_calibration_anchor_values is None
        else _anchor_vector(
            target_calibration_anchor_values,
            expected_length=0 if target_calibration_matrix is None else target_calibration_matrix.shape[0],
            name="target_calibration_anchor_values",
        )
    )
    if train_matrix.shape[0] != train_vector.shape[0]:
        raise ValueError("train_features and train_labels must have the same row count.")
    if train_matrix.shape[0] != subject_vector.shape[0]:
        raise ValueError("train_features and train_subject_ids must have the same row count.")
    if target_vector is not None and test_matrix.shape[0] != target_vector.shape[0]:
        raise ValueError("test_features and target_labels must have the same row count.")
    if (config.target_calibrated or config.pseudo_label_target_calibrated) and target_calibration_matrix is None:
        projection_name = (
            "pseudo_label_target_calibrated_alignment"
            if config.pseudo_label_target_calibrated
            else "target_calibrated_alignment"
        )
        raise ValueError(f"{projection_name} requires target_calibration_features.")
    if target_calibration_matrix is not None and target_calibration_label_vector is not None:
        if target_calibration_matrix.shape[0] != target_calibration_label_vector.shape[0]:
            raise ValueError("target_calibration_features and target_calibration_labels must have the same row count.")
    if train_matrix.shape[1] != test_matrix.shape[1]:
        raise ValueError(
            "train_features and test_features must have the same feature width before alignment: "
            f"{train_matrix.shape[1]} != {test_matrix.shape[1]}."
        )
    if target_calibration_matrix is not None and train_matrix.shape[1] != target_calibration_matrix.shape[1]:
        raise ValueError(
            "train_features and target_calibration_features must have the same feature width before alignment: "
            f"{train_matrix.shape[1]} != {target_calibration_matrix.shape[1]}."
        )

    metadata = config.static_metadata()
    if not config.enabled:
        return SourceAlignmentResult(
            train_features=train_matrix,
            test_features=test_matrix,
            metadata={
                **metadata,
                "alignment_n_components": "",
                "alignment_n_source_subjects": "",
                "alignment_n_classes": "",
                "alignment_n_anchor_values": "",
                "alignment_anchor_value_source": train_anchor_source,
                "alignment_common_anchor_count": "",
                "alignment_anchor_rows_used": "",
                "alignment_anchor_rows_dropped": "",
                "alignment_repetitions_per_class": "",
                "alignment_target_alignment_rows": "",
                "alignment_target_projection_fit": "",
                "alignment_target_labels_used": False,
                "alignment_target_anchor_values_used": False,
            },
            diagnostics={},
        )

    subject_ids = tuple(dict.fromkeys(subject_vector.tolist()))
    if len(subject_ids) < 2:
        raise ValueError("Strict source-only alignment requires at least two source subjects.")

    features_by_subject = {subject_id: train_matrix[subject_vector == subject_id] for subject_id in subject_ids}
    labels_by_subject = {subject_id: train_vector[subject_vector == subject_id] for subject_id in subject_ids}
    anchors_by_subject = {subject_id: train_anchor_vector[subject_vector == subject_id] for subject_id in subject_ids}
    sample_mode = _alignment_sample_mode(config.anchor_mode)
    if unsupervised_alignment:
        train_aligned, test_aligned, covariance_metadata = _transform_unsupervised_covariance_alignment(
            features_by_subject,
            test_matrix,
            method=config.method,
        )
        source_inner_raw_ba = float("nan")
        source_inner_aligned_ba = float("nan")
        if compute_source_inner_diagnostics:
            source_inner_raw_ba, source_inner_aligned_ba = _source_inner_unsupervised_loso_scores(
                features_by_subject=features_by_subject,
                labels_by_subject=labels_by_subject,
                method=config.method,
            )
        source_inner_gain = (
            source_inner_aligned_ba - source_inner_raw_ba
            if np.isfinite(source_inner_aligned_ba) and np.isfinite(source_inner_raw_ba)
            else float("nan")
        )
        diagnostics = {
            "alignment_method": config.method,
            "sample_mode": "unlabeled_covariance",
            "n_source_subjects": len(subject_ids),
            "n_classes": "",
            "n_alignment_rows": 0,
            "n_repetitions_per_class": "",
            "requested_components": "unlabeled_covariance",
            "actual_components": int(train_matrix.shape[1]),
            "feature_dim": int(train_matrix.shape[1]),
            "decode_feature_dim": int(test_aligned.shape[1]),
            "uses_channel_projection_collapse": False,
            "alignment_dimensionality_reduction": bool(test_aligned.shape[1] < train_matrix.shape[1]),
            "anchor_row_correlation_before": "",
            "anchor_row_correlation_after": "",
            "source_inner_decoding_before_alignment": _finite_or_blank(source_inner_raw_ba),
            "source_inner_decoding_after_alignment": _finite_or_blank(source_inner_aligned_ba),
            "source_inner_raw_balanced_accuracy": _finite_or_blank(source_inner_raw_ba),
            "source_inner_aligned_balanced_accuracy": _finite_or_blank(source_inner_aligned_ba),
            "source_inner_aligned_minus_raw": _finite_or_blank(source_inner_gain),
            "source_inner_validation_type": (
                "strict_source_loso_nearest_centroid_unlabeled_target_covariance"
                if compute_source_inner_diagnostics
                else ""
            ),
            "uses_unlabeled_target_data": True,
            "covariance_alignment_estimator": covariance_metadata["covariance_alignment_estimator"],
            "target_transform_type": covariance_metadata["target_transform_type"],
        }
        return SourceAlignmentResult(
            train_features=train_aligned,
            test_features=test_aligned,
            metadata={
                **metadata,
                "alignment_n_components": int(train_matrix.shape[1]),
                "alignment_n_source_subjects": len(subject_ids),
                "alignment_n_classes": "",
                "alignment_n_anchor_values": "",
                "alignment_anchor_value_source": train_anchor_source,
                "alignment_common_anchor_count": "",
                "alignment_anchor_rows_used": 0,
                "alignment_anchor_rows_dropped": 0,
                "alignment_repetitions_per_class": "",
                "alignment_target_alignment_rows": int(test_matrix.shape[0]),
                "alignment_target_projection_fit": covariance_metadata["target_transform_type"],
                "alignment_target_labels_used": False,
                "alignment_target_anchor_values_used": False,
                **covariance_metadata,
            },
            diagnostics=diagnostics,
        )

    fit = _fit_source_alignment_model(
        features_by_subject,
        anchors_by_subject,
        config=config,
        sample_mode=sample_mode,
        external_anchor_mode=not class_anchor_mode,
    )
    model = fit.model
    alignment = fit.alignment
    transformed_by_subject = fit.transformed_by_subject
    transformed_train = np.empty((train_matrix.shape[0], fit.n_components), dtype=float)
    for subject_id in subject_ids:
        transformed_train[subject_vector == subject_id] = transformed_by_subject[subject_id]
    source_inner_raw_ba = float("nan")
    source_inner_aligned_ba = float("nan")
    if compute_source_inner_diagnostics:
        source_inner_raw_ba, source_inner_aligned_ba = _source_inner_strict_loso_scores(
            features_by_subject=features_by_subject,
            labels_by_subject=labels_by_subject,
            anchors_by_subject=anchors_by_subject,
            config=config,
            sample_mode=sample_mode,
            external_anchor_mode=not class_anchor_mode,
        )

    if config.method in {"procrustes", "hyperalignment"}:
        if config.fits_target_projection:
            projection_features = (
                target_calibration_matrix
                if config.target_calibrated or config.pseudo_label_target_calibrated
                else test_matrix
            )
            projection_anchors = (
                target_calibration_anchor_vector
                if config.target_calibrated or config.pseudo_label_target_calibrated
                else target_anchor_vector
            )
            target_anchors = _target_alignment_matrix(
                projection_features,
                projection_anchors,
                classes=alignment.classes,
                config=config,
                n_repetitions_per_class=alignment.n_repetitions_per_class,
            )
            target_projection = fit_projection_to_hyperalignment(target_anchors, template=model.template)
            transformed_test = transform_with_projection(test_matrix, target_projection)
            target_alignment_rows = int(target_projection.n_alignment_rows)
            target_projection_fit = (
                "pseudo_label_template_procrustes"
                if config.pseudo_label_target_calibrated
                else "target_calibrated_template_procrustes"
                if config.target_calibrated
                else "template_procrustes"
            )
        else:
            transformed_test = model.transform_group(test_matrix)
            target_alignment_rows = ""
            target_projection_fit = "source_group_projection"
    elif config.method == "mcca":
        if config.fits_target_projection:
            projection_features = (
                target_calibration_matrix
                if config.target_calibrated or config.pseudo_label_target_calibrated
                else test_matrix
            )
            projection_anchors = (
                target_calibration_anchor_vector
                if config.target_calibrated or config.pseudo_label_target_calibrated
                else target_anchor_vector
            )
            target_anchors = _target_alignment_matrix(
                projection_features,
                projection_anchors,
                classes=alignment.classes,
                config=config,
                n_repetitions_per_class=alignment.n_repetitions_per_class,
            )
            target_mean, target_projection = _fit_target_projection_to_template(
                target_anchors,
                model.component_scores,
                regularization=config.mcca_regularization,
            )
            transformed_test = (test_matrix - target_mean) @ target_projection
            target_alignment_rows = int(target_anchors.shape[0])
            target_projection_fit = (
                "pseudo_label_template_ridge_least_squares"
                if config.pseudo_label_target_calibrated
                else "target_calibrated_template_ridge_least_squares"
                if config.target_calibrated
                else "template_ridge_least_squares"
            )
        else:
            transformed_test = model.transform_group(test_matrix)
            target_alignment_rows = ""
            target_projection_fit = "source_group_projection"
    else:  # pragma: no cover - guarded by normalization
        raise ValueError(f"Unsupported source alignment method: {config.method}")
    n_alignment_rows = int(next(iter(fit.anchor_before.values())).shape[0])
    source_inner_gain = (
        source_inner_aligned_ba - source_inner_raw_ba
        if np.isfinite(source_inner_aligned_ba) and np.isfinite(source_inner_raw_ba)
        else float("nan")
    )
    diagnostics = {
        "alignment_method": config.method,
        "sample_mode": sample_mode,
        "n_source_subjects": len(subject_ids),
        "n_classes": len(alignment.classes),
        "n_alignment_rows": n_alignment_rows,
        "n_repetitions_per_class": "" if alignment.n_repetitions_per_class is None else int(alignment.n_repetitions_per_class),
        "requested_components": _component_label(config.components),
        "actual_components": int(fit.n_components),
        "feature_dim": int(train_matrix.shape[1]),
        "decode_feature_dim": int(transformed_test.shape[1]),
        "uses_channel_projection_collapse": False,
        "alignment_dimensionality_reduction": bool(transformed_test.shape[1] < train_matrix.shape[1]),
        "anchor_row_correlation_before": _finite_or_blank(_mean_pairwise_anchor_row_correlation(fit.anchor_before)),
        "anchor_row_correlation_after": _finite_or_blank(_mean_pairwise_anchor_row_correlation(fit.anchor_after)),
        "source_inner_decoding_before_alignment": _finite_or_blank(source_inner_raw_ba),
        "source_inner_decoding_after_alignment": _finite_or_blank(source_inner_aligned_ba),
        "source_inner_raw_balanced_accuracy": _finite_or_blank(source_inner_raw_ba),
        "source_inner_aligned_balanced_accuracy": _finite_or_blank(source_inner_aligned_ba),
        "source_inner_aligned_minus_raw": _finite_or_blank(source_inner_gain),
        "source_inner_validation_type": (
            "strict_source_loso_nearest_centroid_group_projection" if compute_source_inner_diagnostics else ""
        ),
        "uses_unlabeled_target_data": bool(config.pseudo_label_target_calibrated),
        "covariance_alignment_estimator": "",
        "target_transform_type": target_projection_fit,
    }

    return SourceAlignmentResult(
        train_features=transformed_train,
        test_features=transformed_test,
        metadata={
            **metadata,
            "alignment_n_components": int(fit.n_components),
            "alignment_n_source_subjects": len(subject_ids),
            "alignment_n_classes": len(alignment.classes),
            "alignment_n_anchor_values": int(len(alignment.classes)),
            "alignment_anchor_value_source": train_anchor_source,
            "alignment_common_anchor_count": int(len(alignment.classes)),
            **fit.anchor_filter_metadata,
            "alignment_repetitions_per_class": "" if alignment.n_repetitions_per_class is None else int(alignment.n_repetitions_per_class),
            "alignment_target_alignment_rows": target_alignment_rows,
            "alignment_target_projection_fit": target_projection_fit,
            "alignment_target_labels_used": bool(
                (config.oracle_target_calibrated and target_labels is not None)
                or (config.target_calibrated and target_calibration_labels is not None)
            ),
            "alignment_target_pseudo_labels_used": bool(
                config.pseudo_label_target_calibrated and target_calibration_labels is not None
            ),
            "alignment_target_anchor_values_used": bool(
                (config.oracle_target_calibrated and target_anchor_vector is not None)
                or (
                    (config.target_calibrated or config.pseudo_label_target_calibrated)
                    and target_calibration_anchor_vector is not None
                )
            ),
        },
        diagnostics=diagnostics,
    )


def source_alignment_anchor_availability(
    *,
    train_labels: Sequence[Any] | np.ndarray,
    train_subject_ids: Sequence[Hashable] | np.ndarray,
    config: SourceAlignmentConfig,
    train_anchor_values: Sequence[Any] | np.ndarray | None = None,
    target_labels: Sequence[Any] | np.ndarray | None = None,
    target_anchor_values: Sequence[Any] | np.ndarray | None = None,
    target_calibration_labels: Sequence[Any] | np.ndarray | None = None,
    target_calibration_anchor_values: Sequence[Any] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Summarize fold-local anchor coverage before fitting alignment.

    This is intentionally pre-fit and cheap: failed alignment folds should still
    leave an artifact row explaining whether the source intersection, target
    projection anchors, or M-CCA row count made the fit impossible.
    """
    row: dict[str, Any] = {column: "" for column in ALIGNMENT_ANCHOR_AVAILABILITY_COLUMNS}
    row.update({key: value for key, value in config.static_metadata().items() if key in row})
    row.update(
        {
            "alignment_method": config.method,
            "alignment_anchor_mode": config.anchor_mode,
            "alignment_anchor_column": config.anchor_column,
            "sample_mode": "unlabeled_covariance"
            if _uses_unlabeled_covariance_alignment(config.method)
            else _alignment_sample_mode(config.anchor_mode),
            "alignment_target_projection": config.target_projection,
        }
    )
    if not config.enabled:
        row["prefit_status"] = "alignment_disabled"
        return row

    train_vector = np.asarray(train_labels, dtype=object).reshape(-1)
    subject_vector = np.asarray(train_subject_ids, dtype=object).reshape(-1)
    if train_vector.shape[0] != subject_vector.shape[0]:
        raise ValueError("train_labels and train_subject_ids must have the same row count.")

    subject_ids = tuple(dict.fromkeys(subject_vector.tolist()))
    row.update(
        {
            "n_source_subjects": len(subject_ids),
            "n_source_rows": int(train_vector.shape[0]),
        }
    )
    if _uses_unlabeled_covariance_alignment(config.method):
        row["source_anchor_value_source"] = "unlabeled_covariance"
        row["prefit_status"] = "no_anchors_required"
        return row

    class_anchor_mode = _uses_decoder_label_anchors(config.anchor_mode)
    source_anchor_source = "decoder_labels" if class_anchor_mode else "metadata"
    if class_anchor_mode and train_anchor_values is None:
        train_anchor_values = train_vector
    row["source_anchor_value_source"] = source_anchor_source
    failures: list[str] = []
    if train_anchor_values is None:
        failures.append("missing_train_anchor_values")
        row["prefit_status"] = "likely_fit_failure"
        row["prefit_failure_reason"] = ";".join(failures)
        return row

    train_anchor_vector = _anchor_vector(
        train_anchor_values,
        expected_length=train_vector.shape[0],
        name="train_anchor_values",
    )
    if len(subject_ids) < 2:
        failures.append("strict_source_alignment_requires_at_least_two_source_subjects")
    anchors_by_subject = {subject_id: train_anchor_vector[subject_vector == subject_id] for subject_id in subject_ids}
    unique_counts = [int(np.unique(anchors).size) for anchors in anchors_by_subject.values()]
    common_anchors = _common_anchor_values(anchors_by_subject) if subject_ids else np.asarray([], dtype=object)
    common_set = set(common_anchors.tolist())
    common_counts = [
        int(np.sum(np.asarray([anchor in common_set for anchor in anchors], dtype=bool)))
        for anchors in anchors_by_subject.values()
    ]
    retained_rows = int(sum(common_counts))
    total_rows = int(train_anchor_vector.shape[0])
    row.update(
        {
            "n_source_anchor_values": int(np.unique(train_anchor_vector).size),
            "n_common_source_anchors": int(common_anchors.size),
            "common_source_anchor_values_preview": _preview_values(common_anchors),
            "source_anchor_rows_total": total_rows,
            "source_anchor_rows_retained": retained_rows,
            "source_anchor_rows_dropped": int(total_rows - retained_rows),
            "min_source_unique_anchors_per_subject": min(unique_counts) if unique_counts else "",
            "median_source_unique_anchors_per_subject": _median_int_or_blank(unique_counts),
            "max_source_unique_anchors_per_subject": max(unique_counts) if unique_counts else "",
            "min_source_common_anchor_rows_per_subject": min(common_counts) if common_counts else "",
            "median_source_common_anchor_rows_per_subject": _median_int_or_blank(common_counts),
            "max_source_common_anchor_rows_per_subject": max(common_counts) if common_counts else "",
        }
    )
    if common_anchors.size < 1:
        failures.append("no_common_source_alignment_anchors")
    estimated_rows, estimated_repetitions = _estimate_alignment_rows(
        anchors_by_subject,
        common_anchors=common_anchors,
        config=config,
        sample_mode=str(row["sample_mode"]),
    )
    row["estimated_alignment_rows"] = estimated_rows
    row["estimated_repetitions_per_anchor"] = estimated_repetitions
    if config.method == "mcca" and estimated_rows != "" and int(estimated_rows) < 2:
        failures.append("mcca_requires_at_least_two_aligned_rows")

    if config.fits_target_projection:
        projection_labels = (
            target_calibration_labels
            if config.target_calibrated or config.pseudo_label_target_calibrated
            else target_labels
        )
        projection_anchors = (
            target_calibration_anchor_values
            if config.target_calibrated or config.pseudo_label_target_calibrated
            else target_anchor_values
        )
        if class_anchor_mode and projection_anchors is None:
            projection_anchors = projection_labels
        prefix = "target_calibration" if config.target_calibrated or config.pseudo_label_target_calibrated else "target"
        _update_projection_anchor_availability(
            row,
            prefix=prefix,
            projection_anchors=projection_anchors,
            common_anchors=common_anchors,
            failures=failures,
        )

    row["prefit_status"] = "likely_fit_failure" if failures else "ok"
    row["prefit_failure_reason"] = ";".join(dict.fromkeys(failures))
    return row


def _target_alignment_matrix(
    features: np.ndarray,
    labels: np.ndarray | None,
    *,
    classes: np.ndarray,
    config: SourceAlignmentConfig,
    n_repetitions_per_class: int | None,
) -> np.ndarray:
    if labels is None:  # guarded earlier, but this keeps type-checkers honest
        raise ValueError("Target projection alignment requires target labels or anchor values.")
    labels = np.asarray(labels).reshape(-1)
    if labels.shape[0] != features.shape[0]:
        raise ValueError("target alignment anchors must have the same row count as target features.")
    missing = [class_label for class_label in classes if not np.any(labels == class_label)]
    if missing:
        raise ValueError(f"Target subject is missing alignment anchors: {missing!r}.")
    if _alignment_sample_mode(config.anchor_mode) == "class_mean":
        return np.vstack([np.mean(features[labels == class_label], axis=0) for class_label in classes])

    if n_repetitions_per_class is None:
        raise ValueError("class_repetition oracle target alignment requires a repetition count.")
    rows = []
    for class_position, class_label in enumerate(classes):
        class_features = features[labels == class_label]
        if class_features.shape[0] < n_repetitions_per_class:
            raise ValueError(
                f"Target class {class_label!r} has only {class_features.shape[0]} repetitions, "
                f"need {n_repetitions_per_class}."
            )
        selected = select_class_limited_indices(
            np.zeros(class_features.shape[0], dtype=int),
            int(n_repetitions_per_class),
            selection=DEFAULT_CLASS_LIMIT_SELECTION,
            seed=DEFAULT_CLASS_LIMIT_SEED,
            seed_context=class_position,
        )
        rows.extend(class_features[selected])
    return np.vstack(rows)


def _fit_target_projection_to_template(
    features: np.ndarray,
    template: np.ndarray,
    *,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a target-subject linear projection to an existing common template.

    This intentionally uses target labels/anchors and is therefore only suitable
    for the oracle debug upper bound.
    """

    matrix = _feature_matrix(features, name="target_alignment_features")
    template_matrix = _feature_matrix(template, name="target_alignment_template")
    if matrix.shape[0] != template_matrix.shape[0]:
        raise ValueError(
            "target alignment features and template must have the same row count: "
            f"{matrix.shape[0]} != {template_matrix.shape[0]}."
        )
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    ridge = float(regularization)
    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("target projection regularization must be finite and non-negative.")
    dual = centered @ centered.T
    if ridge > 0:
        dual = dual + ridge * np.eye(dual.shape[0])
    try:
        coefficients = np.linalg.solve(dual, template_matrix)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.pinv(dual) @ template_matrix
    projection = centered.T @ coefficients
    return mean, projection


def _fit_source_alignment_model(
    features_by_subject: Mapping[Hashable, np.ndarray],
    anchors_by_subject: Mapping[Hashable, np.ndarray],
    *,
    config: SourceAlignmentConfig,
    sample_mode: str,
    external_anchor_mode: bool,
) -> _SourceAlignmentFit:
    subject_ids = tuple(features_by_subject)
    fit_features_by_subject, fit_anchors_by_subject, anchor_filter_metadata = _source_alignment_fit_inputs(
        features_by_subject,
        anchors_by_subject,
        external_anchor_mode=external_anchor_mode,
    )
    n_repetitions = _effective_repetitions_per_class(fit_anchors_by_subject, sample_mode, config)

    if config.method in {"procrustes", "hyperalignment"}:
        iterations = 1 if config.method == "procrustes" else config.hyperalignment_iterations
        model, alignment = fit_class_hyperalignment(
            fit_features_by_subject,
            fit_anchors_by_subject,
            sample_mode=sample_mode,
            n_repetitions_per_class=n_repetitions,
            n_components=config.components,
            n_iterations=iterations,
            initialization="mean" if config.method == "procrustes" else "pca",
        )
    elif config.method == "mcca":
        model, alignment = fit_class_mcca(
            fit_features_by_subject,
            fit_anchors_by_subject,
            sample_mode=sample_mode,
            n_repetitions_per_class=n_repetitions,
            n_components=config.components,
            regularization=config.mcca_regularization,
            subject_pca_components=config.mcca_subject_pca_components,
        )
    else:  # pragma: no cover - guarded by normalization
        raise ValueError(f"Unsupported source alignment method: {config.method}")

    transformed_by_subject = {subject_id: model.transform(subject_id, features_by_subject[subject_id]) for subject_id in subject_ids}
    anchor_before = alignment.aligned_by_subject
    anchor_after = {subject_id: model.transform(subject_id, anchor_before[subject_id]) for subject_id in subject_ids}
    return _SourceAlignmentFit(
        model=model,
        alignment=alignment,
        transformed_by_subject=transformed_by_subject,
        anchor_filter_metadata=anchor_filter_metadata,
        anchor_before=anchor_before,
        anchor_after=anchor_after,
        n_components=int(model.n_components),
    )


def _transform_unsupervised_covariance_alignment(
    features_by_subject: Mapping[Hashable, np.ndarray],
    test_features: np.ndarray,
    *,
    method: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    subject_ids = tuple(features_by_subject)
    target_stats = _covariance_stats(test_features)
    pooled_source = np.vstack([features_by_subject[subject_id] for subject_id in subject_ids])
    source_stats = _covariance_stats(pooled_source)
    estimators = {target_stats.estimator, source_stats.estimator}
    transformed_by_subject: list[np.ndarray] = []

    if method in {"euclidean", "subject_sensor_covariance"}:
        for subject_id in subject_ids:
            stats = _covariance_stats(features_by_subject[subject_id])
            estimators.add(stats.estimator)
            transformed_by_subject.append(_covariance_whiten(features_by_subject[subject_id], stats))
        transformed_test = _covariance_whiten(test_features, target_stats)
        target_transform_type = "unlabeled_target_covariance_whitening"
    elif method == "coral":
        for subject_id in subject_ids:
            stats = _covariance_stats(features_by_subject[subject_id])
            estimators.add(stats.estimator)
            transformed_by_subject.append(_covariance_recolor(_covariance_whiten(features_by_subject[subject_id], stats), target_stats))
        transformed_test = np.asarray(test_features, dtype=float)
        target_transform_type = "unlabeled_target_covariance_recoloring"
    elif method == "target_baseline_covariance":
        transformed_by_subject = [
            _covariance_recolor(_covariance_whiten(features_by_subject[subject_id], source_stats), target_stats)
            for subject_id in subject_ids
        ]
        transformed_test = np.asarray(test_features, dtype=float)
        target_transform_type = "unlabeled_target_pooled_source_to_target_covariance"
    else:  # pragma: no cover - guarded by normalization
        raise ValueError(f"Unsupported unsupervised covariance alignment method: {method}.")

    return (
        np.vstack(transformed_by_subject),
        transformed_test,
        {
            "alignment_uses_unlabeled_target_data": True,
            "alignment_covariance_method": method,
            "covariance_alignment_estimator": "|".join(sorted(estimators)),
            "target_transform_type": target_transform_type,
        },
    )


def _covariance_stats(features: np.ndarray) -> _CovarianceStats:
    matrix = _feature_matrix(features, name="covariance_alignment_features")
    mean = np.mean(matrix, axis=0)
    centered = matrix - mean
    n_rows, n_features = centered.shape
    if n_features <= MAX_FULL_COVARIANCE_FEATURES and n_rows > 1:
        covariance = centered.T @ centered / max(1, n_rows - 1)
        scale = float(np.trace(covariance) / max(1, n_features))
        floor = MIN_COVARIANCE_EIGENVALUE * max(scale, 1.0)
        values, vectors = np.linalg.eigh(covariance)
        values = np.maximum(values, floor)
        sqrt = (vectors * np.sqrt(values)) @ vectors.T
        inv_sqrt = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
        return _CovarianceStats(mean=mean, sqrt=sqrt, inv_sqrt=inv_sqrt, estimator="full")

    variance = np.var(centered, axis=0, ddof=1 if n_rows > 1 else 0)
    scale = float(np.mean(variance)) if variance.size else 1.0
    floor = MIN_COVARIANCE_EIGENVALUE * max(scale, 1.0)
    std = np.sqrt(np.maximum(variance, floor))
    return _CovarianceStats(mean=mean, sqrt=std, inv_sqrt=1.0 / std, estimator="diagonal")


def _covariance_whiten(features: np.ndarray, stats: _CovarianceStats) -> np.ndarray:
    centered = np.asarray(features, dtype=float) - stats.mean
    if stats.estimator == "full":
        return centered @ stats.inv_sqrt
    return centered * stats.inv_sqrt


def _covariance_recolor(whitened_features: np.ndarray, target_stats: _CovarianceStats) -> np.ndarray:
    if target_stats.estimator == "full":
        return whitened_features @ target_stats.sqrt + target_stats.mean
    return whitened_features * target_stats.sqrt + target_stats.mean


def _effective_repetitions_per_class(
    labels_by_subject: Mapping[Hashable, np.ndarray],
    sample_mode: str,
    config: SourceAlignmentConfig,
) -> int | None:
    if sample_mode != "class_repetition":
        return None
    subject_ids = tuple(labels_by_subject)
    first_classes = np.unique(labels_by_subject[subject_ids[0]])
    counts = []
    for subject_id in subject_ids:
        classes = np.unique(labels_by_subject[subject_id])
        if not np.array_equal(first_classes, classes):
            raise ValueError(f"Subject {subject_id!r} does not contain the common alignment classes.")
        counts.extend(int(np.sum(labels_by_subject[subject_id] == class_label)) for class_label in first_classes)
    available = min(counts)
    if available < 1:
        raise ValueError("Every source subject must have at least one sample per alignment class.")
    if config.repetition_cap is None:
        return int(available)
    return int(min(available, int(config.repetition_cap)))


def _source_alignment_fit_inputs(
    features_by_subject: Mapping[Hashable, np.ndarray],
    anchors_by_subject: Mapping[Hashable, np.ndarray],
    *,
    external_anchor_mode: bool,
) -> tuple[dict[Hashable, np.ndarray], dict[Hashable, np.ndarray], dict[str, Any]]:
    total_rows = int(sum(features.shape[0] for features in features_by_subject.values()))
    if not external_anchor_mode:
        return (
            dict(features_by_subject),
            dict(anchors_by_subject),
            {
                "alignment_anchor_rows_used": total_rows,
                "alignment_anchor_rows_dropped": 0,
            },
        )

    common_anchors = _common_anchor_values(anchors_by_subject)
    if len(common_anchors) < 1:
        raise ValueError("No common source alignment anchors are shared by every source subject.")
    common = set(common_anchors.tolist())
    fit_features: dict[Hashable, np.ndarray] = {}
    fit_anchors: dict[Hashable, np.ndarray] = {}
    used_rows = 0
    for subject_id, anchors in anchors_by_subject.items():
        mask = np.asarray([anchor in common for anchor in anchors], dtype=bool)
        if not np.any(mask):
            raise ValueError(f"Subject {subject_id!r} has no rows after common alignment-anchor filtering.")
        fit_features[subject_id] = features_by_subject[subject_id][mask]
        fit_anchors[subject_id] = anchors[mask]
        used_rows += int(np.sum(mask))
    return (
        fit_features,
        fit_anchors,
        {
            "alignment_anchor_rows_used": int(used_rows),
            "alignment_anchor_rows_dropped": int(total_rows - used_rows),
        },
    )


def _common_anchor_values(anchors_by_subject: Mapping[Hashable, np.ndarray]) -> np.ndarray:
    subject_ids = tuple(anchors_by_subject)
    if not subject_ids:
        raise ValueError("At least one source subject is required for anchor intersection.")
    first = np.unique(anchors_by_subject[subject_ids[0]])
    common = set(first.tolist())
    for subject_id in subject_ids[1:]:
        common &= set(np.unique(anchors_by_subject[subject_id]).tolist())
    return np.asarray([anchor for anchor in first if anchor in common], dtype=object)


def _uses_decoder_label_anchors(anchor_mode: str) -> bool:
    return anchor_mode in SOURCE_ALIGNMENT_CLASS_ANCHOR_MODES


def _uses_unlabeled_covariance_alignment(method: str) -> bool:
    return method in SOURCE_ALIGNMENT_UNSUPERVISED_METHODS


def _alignment_sample_mode(anchor_mode: str) -> str:
    if anchor_mode in {"class_repetition", "stimulus_id_repetition"}:
        return "class_repetition"
    return "class_mean"


def _anchor_vector(values: Sequence[Any] | np.ndarray | None, *, expected_length: int, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required for this alignment mode.")
    vector = np.asarray(values, dtype=object).reshape(-1)
    if vector.shape[0] != expected_length:
        raise ValueError(f"{name} must have the same row count as the corresponding feature matrix.")
    if any(value is None or (isinstance(value, float) and np.isnan(value)) for value in vector):
        raise ValueError(f"{name} contains missing values.")
    return vector


def _component_label(value: int | float) -> int | str:
    return "inf" if value == float("inf") else int(value)


def _finite_or_blank(value: float) -> float | str:
    return float(value) if np.isfinite(value) else ""


def _median_int_or_blank(values: Sequence[int]) -> int | float | str:
    if not values:
        return ""
    median = float(np.median(np.asarray(values, dtype=float)))
    return int(median) if median.is_integer() else median


def _preview_values(values: Sequence[Any] | np.ndarray, *, limit: int = 8) -> str:
    vector = np.asarray(values, dtype=object).reshape(-1)
    preview = [str(value) for value in vector[:limit]]
    if vector.size > limit:
        preview.append("...")
    return "|".join(preview)


def _estimate_alignment_rows(
    anchors_by_subject: Mapping[Hashable, np.ndarray],
    *,
    common_anchors: np.ndarray,
    config: SourceAlignmentConfig,
    sample_mode: str,
) -> tuple[int | str, int | str]:
    if common_anchors.size < 1:
        return 0, ""
    if sample_mode != "class_repetition":
        return int(common_anchors.size), ""
    common_set = set(common_anchors.tolist())
    filtered = {
        subject_id: np.asarray([anchor for anchor in anchors if anchor in common_set], dtype=object)
        for subject_id, anchors in anchors_by_subject.items()
    }
    try:
        repetitions = _effective_repetitions_per_class(filtered, sample_mode, config)
    except ValueError:
        return "", ""
    if repetitions is None:
        return int(common_anchors.size), ""
    return int(common_anchors.size * repetitions), int(repetitions)


def _update_projection_anchor_availability(
    row: dict[str, Any],
    *,
    prefix: str,
    projection_anchors: Sequence[Any] | np.ndarray | None,
    common_anchors: np.ndarray,
    failures: list[str],
) -> None:
    used_key = f"{prefix}_anchor_values_used"
    rows_key = f"n_{prefix}_rows"
    values_key = f"n_{prefix}_anchor_values"
    missing_count_key = f"{prefix}_missing_common_anchor_count"
    missing_preview_key = f"{prefix}_missing_common_anchor_values_preview"
    row[used_key] = projection_anchors is not None
    if projection_anchors is None:
        failures.append(f"{prefix}_projection_missing_anchor_values")
        return
    vector = np.asarray(projection_anchors, dtype=object).reshape(-1)
    row[rows_key] = int(vector.shape[0])
    row[values_key] = int(np.unique(vector).size)
    available = set(np.unique(vector).tolist())
    missing = np.asarray([anchor for anchor in common_anchors if anchor not in available], dtype=object)
    row[missing_count_key] = int(missing.size)
    row[missing_preview_key] = _preview_values(missing)
    if missing.size:
        failures.append(f"{prefix}_subject_missing_alignment_anchors")


def _mean_pairwise_anchor_row_correlation(matrices: Mapping[Hashable, np.ndarray]) -> float:
    subject_ids = tuple(matrices)
    if len(subject_ids) < 2:
        return float("nan")
    values: list[float] = []
    for left_index, left_subject in enumerate(subject_ids):
        left = np.asarray(matrices[left_subject], dtype=float)
        if left.ndim != 2 or left.shape[1] < 2:
            continue
        left_centered = left - np.mean(left, axis=1, keepdims=True)
        left_norm = np.linalg.norm(left_centered, axis=1)
        for right_subject in subject_ids[left_index + 1 :]:
            right = np.asarray(matrices[right_subject], dtype=float)
            if right.shape != left.shape:
                continue
            right_centered = right - np.mean(right, axis=1, keepdims=True)
            right_norm = np.linalg.norm(right_centered, axis=1)
            denom = left_norm * right_norm
            valid = denom > 1e-12
            if not np.any(valid):
                continue
            correlations = np.sum(left_centered[valid] * right_centered[valid], axis=1) / denom[valid]
            values.extend(float(value) for value in correlations if np.isfinite(value))
    return float(np.mean(values)) if values else float("nan")


def _source_inner_strict_loso_scores(
    *,
    features_by_subject: Mapping[Hashable, np.ndarray],
    labels_by_subject: Mapping[Hashable, np.ndarray],
    anchors_by_subject: Mapping[Hashable, np.ndarray],
    config: SourceAlignmentConfig,
    sample_mode: str,
    external_anchor_mode: bool,
) -> tuple[float, float]:
    """Compare raw vs aligned source-inner held-out-subject decoding.

    The aligned side treats each source subject as target-like: the alignment
    model is fit from the remaining source subjects and the held-out source is
    transformed only with the source-fitted group projection.
    """

    subject_ids = tuple(features_by_subject)
    if len(subject_ids) < 2:
        return float("nan"), float("nan")
    raw_predictions: list[Any] = []
    raw_truth: list[Any] = []
    aligned_predictions: list[Any] = []
    aligned_truth: list[Any] = []
    for test_subject in subject_ids:
        train_subjects = tuple(subject_id for subject_id in subject_ids if subject_id != test_subject)
        train_features = np.vstack([features_by_subject[subject_id] for subject_id in train_subjects])
        train_labels = np.concatenate([labels_by_subject[subject_id] for subject_id in train_subjects])
        test_features = features_by_subject[test_subject]
        test_labels = labels_by_subject[test_subject]

        predicted_raw = _nearest_centroid_predict(train_features, train_labels, test_features)
        if predicted_raw.size:
            raw_predictions.extend(predicted_raw.tolist())
            raw_truth.extend(test_labels.tolist())

        if len(train_subjects) < 2:
            continue
        try:
            inner_fit = _fit_source_alignment_model(
                {subject_id: features_by_subject[subject_id] for subject_id in train_subjects},
                {subject_id: anchors_by_subject[subject_id] for subject_id in train_subjects},
                config=config,
                sample_mode=sample_mode,
                external_anchor_mode=external_anchor_mode,
            )
            aligned_train_features = np.vstack([inner_fit.transformed_by_subject[subject_id] for subject_id in train_subjects])
            aligned_test_features = inner_fit.model.transform_group(test_features)
        except (ValueError, np.linalg.LinAlgError):
            continue
        predicted_aligned = _nearest_centroid_predict(aligned_train_features, train_labels, aligned_test_features)
        if predicted_aligned.size:
            aligned_predictions.extend(predicted_aligned.tolist())
            aligned_truth.extend(test_labels.tolist())

    raw_score = (
        _balanced_accuracy(np.asarray(raw_truth, dtype=object), np.asarray(raw_predictions, dtype=object))
        if raw_truth
        else float("nan")
    )
    aligned_score = (
        _balanced_accuracy(np.asarray(aligned_truth, dtype=object), np.asarray(aligned_predictions, dtype=object))
        if aligned_truth
        else float("nan")
    )
    return raw_score, aligned_score


def _source_inner_unsupervised_loso_scores(
    *,
    features_by_subject: Mapping[Hashable, np.ndarray],
    labels_by_subject: Mapping[Hashable, np.ndarray],
    method: str,
) -> tuple[float, float]:
    subject_ids = tuple(features_by_subject)
    if len(subject_ids) < 2:
        return float("nan"), float("nan")
    raw_predictions: list[Any] = []
    raw_truth: list[Any] = []
    aligned_predictions: list[Any] = []
    aligned_truth: list[Any] = []
    for test_subject in subject_ids:
        train_subjects = tuple(subject_id for subject_id in subject_ids if subject_id != test_subject)
        train_features = np.vstack([features_by_subject[subject_id] for subject_id in train_subjects])
        train_labels = np.concatenate([labels_by_subject[subject_id] for subject_id in train_subjects])
        test_features = features_by_subject[test_subject]
        test_labels = labels_by_subject[test_subject]

        predicted_raw = _nearest_centroid_predict(train_features, train_labels, test_features)
        if predicted_raw.size:
            raw_predictions.extend(predicted_raw.tolist())
            raw_truth.extend(test_labels.tolist())

        aligned_train_features, aligned_test_features, _metadata = _transform_unsupervised_covariance_alignment(
            {subject_id: features_by_subject[subject_id] for subject_id in train_subjects},
            test_features,
            method=method,
        )
        predicted_aligned = _nearest_centroid_predict(aligned_train_features, train_labels, aligned_test_features)
        if predicted_aligned.size:
            aligned_predictions.extend(predicted_aligned.tolist())
            aligned_truth.extend(test_labels.tolist())

    raw_score = (
        _balanced_accuracy(np.asarray(raw_truth, dtype=object), np.asarray(raw_predictions, dtype=object))
        if raw_truth
        else float("nan")
    )
    aligned_score = (
        _balanced_accuracy(np.asarray(aligned_truth, dtype=object), np.asarray(aligned_predictions, dtype=object))
        if aligned_truth
        else float("nan")
    )
    return raw_score, aligned_score


def _nearest_centroid_predict(train_features: np.ndarray, train_labels: np.ndarray, test_features: np.ndarray) -> np.ndarray:
    labels = np.asarray(train_labels).reshape(-1)
    classes = np.unique(labels)
    if len(classes) < 2:
        return np.asarray([], dtype=object)
    centroids = np.vstack([np.mean(train_features[labels == class_label], axis=0) for class_label in classes])
    distances = np.sum((test_features[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return np.asarray(classes, dtype=object)[distances.argmin(axis=1)]


def _balanced_accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    recalls: list[float] = []
    for class_label in np.unique(labels):
        mask = labels == class_label
        if not np.any(mask):
            continue
        recalls.append(float(np.mean(predictions[mask] == class_label)))
    return float(np.mean(recalls)) if recalls else float("nan")


def _feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must have at least one row and one column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values.")
    return matrix


__all__ = [
    "ALIGNMENT_ANCHOR_AVAILABILITY_COLUMNS",
    "ALIGNMENT_DIAGNOSTIC_COLUMNS",
    "DEFAULT_ALIGNMENT_COMPONENTS",
    "DEFAULT_ALIGNMENT_REPETITION_CAP",
    "DEFAULT_ALIGNMENT_TIMES",
    "ORACLE_TARGET_CALIBRATED_ALIGNMENT",
    "PSEUDO_LABEL_TARGET_CALIBRATED_ALIGNMENT",
    "SAME_DECODE_WINDOW_ALIGNMENT",
    "SOURCE_ALIGNMENT_ANCHOR_MODES",
    "SOURCE_ALIGNMENT_CLASS_ANCHORED_METHODS",
    "SOURCE_ALIGNMENT_METHODS",
    "SOURCE_ALIGNMENT_TARGET_PROJECTIONS",
    "SOURCE_ALIGNMENT_UNSUPERVISED_METHODS",
    "SourceAlignmentConfig",
    "SourceAlignmentResult",
    "TARGET_CALIBRATED_ALIGNMENT",
    "align_train_test_features",
    "normalize_source_alignment_anchor_mode",
    "normalize_source_alignment_method",
    "source_alignment_anchor_availability",
    "source_alignment_config",
]

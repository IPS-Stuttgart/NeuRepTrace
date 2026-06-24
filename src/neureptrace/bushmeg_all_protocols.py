"""Unified BUSH-MEG all-protocol evaluation runner.

This module is an orchestration layer over the existing BUSH-MEG LOSO
implementations. It keeps protocol metadata explicit so benchmark-valid
source-only, unlabeled transductive, same-subject calibration, and oracle/debug
results cannot be mixed accidentally.
"""

from __future__ import annotations

import argparse
import hashlib
import copy
import importlib.util
import inspect
import json
import os
import signal
import shutil
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss

from neureptrace.dataset_config import load_config, parse_participant_ids
from neureptrace.metrics import brier_score_multiclass, expected_calibration_error

RESPONSE_WINDOW_C = (0.088, 0.136, 0.184, 0.232, 0.280)
DEFAULT_OUT_DIR = Path("results/bush_meg/all_protocols")
DEFAULT_CONFIG = Path("configs/bush_meg/all_protocols.yml")

_PROTOCOL3_SUBJECT_CACHE: dict[str, tuple[Any, Any]] = {}

PROGRESS_STAGES = {
    "configured",
    "checking_requirements",
    "loading_subjects",
    "loaded_subject",
    "fold_start",
    "feature_start",
    "fit_start",
    "predict_start",
    "fold_done",
    "method_done",
    "method_failed",
    "method_skipped",
}
AGGREGATE_PROGRESS_STAGES = {"fold_done", "method_done", "method_failed", "method_skipped"}


class RunTimeoutError(TimeoutError):
    """Raised when a method or fold exceeds its configured timeout."""

    def __init__(self, *, kind: str, seconds: float, context: Mapping[str, Any] | None = None) -> None:
        self.kind = kind
        self.seconds = float(seconds)
        self.context = dict(context or {})
        context_text = ", ".join(f"{key}={value}" for key, value in sorted(self.context.items()))
        message = f"{kind} timeout exceeded {self.seconds:g} seconds"
        if context_text:
            message = f"{message} ({context_text})"
        super().__init__(message)

REGISTRY_AUDIT_COLUMNS = [
    "method",
    "method_family",
    "protocol_category",
    "protocol_name",
    "uses_source_data",
    "uses_source_labels",
    "uses_target_data",
    "uses_target_labels_for_fitting",
    "calibration_rows_disjoint_from_evaluation",
    "valid_for_strict_source_only",
    "valid_for_zero_calibration",
    "debug_upper_bound",
    "uses_target_labels_for_scoring_only",
    "target_data_use",
    "target_label_use",
    "runner",
    "required_modules",
    "required_config_any",
    "requires_torch",
    "inventory_only",
    "implementation_status",
    "skip_reason",
    "requested",
    "missing_required_modules",
]

SUMMARY_COLUMNS = [
    "method",
    "method_family",
    "protocol_category",
    "protocol_name",
    "uses_source_data",
    "uses_source_labels",
    "uses_target_data",
    "uses_target_labels_for_fitting",
    "calibration_rows_disjoint_from_evaluation",
    "valid_for_strict_source_only",
    "valid_for_zero_calibration",
    "debug_upper_bound",
    "uses_target_labels_for_scoring_only",
    "target_data_use",
    "target_label_use",
    "outer_test_subject",
    "n_source_subjects",
    "n_source_trials",
    "n_target_trials",
    "n_calibration_trials",
    "target_calibration_per_class",
    "k_per_class",
    "n_target_calibration_trials",
    "n_target_evaluation_trials",
    "target_calibration_seed",
    "feature_kind",
    "window_centers",
    "window_size",
    "temporal_bins",
    "balanced_accuracy",
    "accuracy",
    "top2_accuracy",
    "top3_accuracy",
    "log_loss",
    "brier",
    "ece",
]


@dataclass(frozen=True, slots=True)
class ProtocolSpec:
    """Protocol metadata written into every all-protocol artifact."""

    category: int
    name: str
    uses_source_data: bool
    uses_source_labels: bool
    uses_target_data: bool
    uses_target_labels_for_fitting: bool
    uses_target_labels_for_scoring_only: bool
    target_features_for_fitting_allowed: bool
    calibration_rows_disjoint_from_evaluation: bool
    valid_for_strict_source_only: bool
    valid_for_zero_calibration: bool
    debug_upper_bound: bool
    target_data_use: str
    target_label_use: str

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_category": int(self.category),
            "protocol_name": self.name,
            "uses_source_data": bool(self.uses_source_data),
            "uses_source_labels": bool(self.uses_source_labels),
            "uses_target_data": bool(self.uses_target_data),
            "uses_target_labels_for_fitting": bool(self.uses_target_labels_for_fitting),
            "calibration_rows_disjoint_from_evaluation": bool(self.calibration_rows_disjoint_from_evaluation),
            "valid_for_strict_source_only": bool(self.valid_for_strict_source_only),
            "valid_for_zero_calibration": bool(self.valid_for_zero_calibration),
            "debug_upper_bound": bool(self.debug_upper_bound),
            "uses_target_labels_for_scoring_only": bool(self.uses_target_labels_for_scoring_only),
            "target_data_use": self.target_data_use,
            "target_label_use": self.target_label_use,
        }


PROTOCOLS: dict[int, ProtocolSpec] = {
    1: ProtocolSpec(
        category=1,
        name="strict_source_only",
        uses_source_data=True,
        uses_source_labels=True,
        uses_target_data=False,
        uses_target_labels_for_fitting=False,
        uses_target_labels_for_scoring_only=True,
        target_features_for_fitting_allowed=False,
        calibration_rows_disjoint_from_evaluation=True,
        valid_for_strict_source_only=True,
        valid_for_zero_calibration=True,
        debug_upper_bound=False,
        target_data_use="inference_only",
        target_label_use="scoring_only",
    ),
    2: ProtocolSpec(
        category=2,
        name="unlabeled_transductive_adaptation",
        uses_source_data=True,
        uses_source_labels=True,
        uses_target_data=True,
        uses_target_labels_for_fitting=False,
        uses_target_labels_for_scoring_only=True,
        target_features_for_fitting_allowed=True,
        calibration_rows_disjoint_from_evaluation=True,
        valid_for_strict_source_only=False,
        valid_for_zero_calibration=True,
        debug_upper_bound=False,
        target_data_use="unlabeled_adaptation_and_inference",
        target_label_use="scoring_only",
    ),
    3: ProtocolSpec(
        category=3,
        name="same_subject_calibration_split",
        uses_source_data=True,
        uses_source_labels=True,
        uses_target_data=True,
        uses_target_labels_for_fitting=True,
        uses_target_labels_for_scoring_only=True,
        target_features_for_fitting_allowed=True,
        calibration_rows_disjoint_from_evaluation=True,
        valid_for_strict_source_only=False,
        valid_for_zero_calibration=False,
        debug_upper_bound=False,
        target_data_use="disjoint_calibration_and_evaluation",
        target_label_use="calibration_labels_plus_scoring_labels",
    ),
    4: ProtocolSpec(
        category=4,
        name="oracle_target_calibrated_debug",
        uses_source_data=True,
        uses_source_labels=True,
        uses_target_data=True,
        uses_target_labels_for_fitting=True,
        uses_target_labels_for_scoring_only=True,
        target_features_for_fitting_allowed=True,
        calibration_rows_disjoint_from_evaluation=False,
        valid_for_strict_source_only=False,
        valid_for_zero_calibration=False,
        debug_upper_bound=True,
        target_data_use="oracle_target_calibration_and_inference",
        target_label_use="oracle_fitting_and_scoring",
    ),
}


@dataclass(frozen=True, slots=True)
class MethodSpec:
    """One runnable or inventory-only all-protocol method."""

    method: str
    method_family: str
    protocol_category: int
    runner: str
    config_updates: Mapping[str, Any] = field(default_factory=dict)
    runnable: bool = True
    blocked_reason: str = ""
    required_modules: tuple[str, ...] = ()
    requires_torch: bool = False
    required_config_any: tuple[str, ...] = ()

    @property
    def protocol(self) -> ProtocolSpec:
        return PROTOCOLS[self.protocol_category]

    def metadata(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "method_family": self.method_family,
            "runner": self.runner,
            "runnable": bool(self.runnable),
            "blocked_reason": self.blocked_reason,
            "inventory_only": bool((not self.runnable) or self.runner == "unavailable"),
            "requires_torch": bool(self.requires_torch),
            "required_modules": "|".join(self.required_modules),
            "required_config_any": "|".join(self.required_config_any),
            "status": "runnable" if self.runnable else "skipped",
            "skip_reason": "" if self.runnable else self.blocked_reason,
            **self.protocol.metadata(),
        }


@dataclass(frozen=True, slots=True)
class AllProtocolsResult:
    summary_csv: Path
    predictions_csv: Path
    method_metadata_csv: Path
    provenance_json: Path
    summary: pd.DataFrame
    predictions: pd.DataFrame
    method_metadata: pd.DataFrame


@dataclass(frozen=True, slots=True)
class BushmegTargetCalibrationSplit:
    """Protocol 3 target calibration/evaluation row split."""

    calibration_indices: np.ndarray
    evaluation_indices: np.ndarray
    per_class: int
    seed: int
    min_evaluation_per_class: int
    context: tuple[str, ...] = ()
    effective_seed: int | None = None
    skipped: bool = False
    skip_reason: str = ""
    skip_reason_code: str = ""
    n_classes: int = 0

    @property
    def n_target_calibration_trials(self) -> int:
        return int(self.calibration_indices.size)

    @property
    def n_target_evaluation_trials(self) -> int:
        return int(self.evaluation_indices.size)

    @property
    def calibration_rows_disjoint_from_evaluation(self) -> bool:
        return bool(np.intersect1d(self.calibration_indices, self.evaluation_indices).size == 0)

    def metadata(self) -> dict[str, Any]:
        return {
            "protocol_category": 3,
            "uses_target_data": True,
            "uses_target_labels_for_fitting": True,
            "valid_for_zero_calibration": False,
            "valid_for_strict_source_only": False,
            "target_calibration_per_class": int(self.per_class),
            "n_target_calibration_trials": self.n_target_calibration_trials,
            "n_target_evaluation_trials": self.n_target_evaluation_trials,
            "calibration_rows_disjoint_from_evaluation": self.calibration_rows_disjoint_from_evaluation,
            "target_calibration_seed": int(self.seed),
            "target_calibration_effective_seed": self.effective_seed,
            "target_calibration_context": "|".join(self.context),
            "target_calibration_skipped": bool(self.skipped),
            "target_calibration_skip_reason": self.skip_reason,
            "target_calibration_skip_reason_code": self.skip_reason_code,
            "n_target_classes": int(self.n_classes),
        }


@dataclass(frozen=True, slots=True)
class Protocol3FoldAdapterResult:
    """Fold-level Protocol 3 result emitted by the generic split adapter."""

    summary: pd.DataFrame
    predictions: pd.DataFrame
    split: BushmegTargetCalibrationSplit
    skipped: bool = False
    skip_reason: str = ""


def _protocol_spec(protocol: int | ProtocolSpec) -> ProtocolSpec:
    if isinstance(protocol, ProtocolSpec):
        return protocol
    return PROTOCOLS[int(protocol)]


def validate_disjoint_calibration_evaluation(
    calibration_indices: Sequence[int] | np.ndarray,
    evaluation_indices: Sequence[int] | np.ndarray,
) -> None:
    """Reject Protocol 3 calibration/evaluation overlap."""

    calibration = np.asarray(calibration_indices, dtype=int)
    evaluation = np.asarray(evaluation_indices, dtype=int)
    overlap = np.intersect1d(calibration, evaluation)
    if overlap.size:
        preview = ",".join(map(str, overlap[:10]))
        raise ValueError(f"Protocol 3 calibration/evaluation rows must be disjoint; overlapping row(s): {preview}.")


def validate_protocol_input_use(
    protocol: int | ProtocolSpec,
    *,
    target_features_for_fitting: bool = False,
    target_labels_for_fitting: bool = False,
    target_class_prototypes_for_fitting: bool = False,
    target_accuracy_for_model_selection: bool = False,
    calibration_indices: Sequence[int] | np.ndarray | None = None,
    evaluation_indices: Sequence[int] | np.ndarray | None = None,
    include_oracle: bool = False,
    require_protocol3_split: bool = True,
) -> None:
    """Validate target-data/target-label use against the four-protocol taxonomy."""

    spec = _protocol_spec(protocol)
    if spec.category == 4 and not include_oracle:
        raise ValueError("Protocol 4 oracle/debug methods require --include-oracle.")

    if target_features_for_fitting and not spec.target_features_for_fitting_allowed:
        raise ValueError(
            f"Protocol {spec.category} ({spec.name}) must not receive held-out target features for fitting/adaptation."
        )

    if target_labels_for_fitting and not spec.uses_target_labels_for_fitting:
        raise ValueError(
            f"Protocol {spec.category} ({spec.name}) must not use held-out target labels for fitting/adaptation."
        )

    if spec.category in {1, 2}:
        if target_class_prototypes_for_fitting:
            raise ValueError(
                f"Protocol {spec.category} ({spec.name}) must not use target class prototypes for fitting/adaptation."
            )
        if target_accuracy_for_model_selection:
            raise ValueError(
                f"Protocol {spec.category} ({spec.name}) must not use target accuracy for fitting/adaptation/model selection."
            )

    if spec.category == 3 and require_protocol3_split:
        if calibration_indices is None or evaluation_indices is None:
            raise ValueError("Protocol 3 methods must provide disjoint calibration_indices and evaluation_indices.")
        validate_disjoint_calibration_evaluation(calibration_indices, evaluation_indices)


def validate_target_label_policy(
    protocol: int | ProtocolSpec,
    *,
    uses_target_labels_for_fitting: bool,
    include_oracle: bool = False,
) -> None:
    """Reject target-label fitting in protocols where it is leakage."""

    validate_protocol_input_use(
        protocol,
        target_labels_for_fitting=uses_target_labels_for_fitting,
        include_oracle=include_oracle,
        require_protocol3_split=False,
    )


def _empty_bushmeg_target_calibration_split(
    *,
    per_class: int,
    seed: int,
    min_evaluation_per_class: int,
    context: Sequence[Any],
    skip_reason: str,
    skip_reason_code: str,
    n_classes: int = 0,
) -> BushmegTargetCalibrationSplit:
    return BushmegTargetCalibrationSplit(
        calibration_indices=np.asarray([], dtype=int),
        evaluation_indices=np.asarray([], dtype=int),
        per_class=int(per_class),
        seed=int(seed),
        min_evaluation_per_class=int(min_evaluation_per_class),
        context=tuple(str(item) for item in context),
        effective_seed=None,
        skipped=True,
        skip_reason=skip_reason,
        skip_reason_code=skip_reason_code,
        n_classes=int(n_classes),
    )


def _stable_target_calibration_seed(seed: int, *, per_class: int, context: Sequence[Any]) -> int:
    payload = json.dumps(
        {
            "seed": int(seed),
            "per_class": int(per_class),
            "context": [str(item) for item in context],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**32)


def _readable_label_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def select_bushmeg_target_calibration_split(
    target_labels: Sequence[Any] | np.ndarray,
    *,
    per_class: int,
    seed: int,
    min_evaluation_per_class: int = 1,
    context: Sequence[Any] = (),
) -> BushmegTargetCalibrationSplit:
    """Select deterministic Protocol 3 calibration rows and leave the rest for scoring.

    The function returns a structured skipped result for infeasible folds rather
    than raising, so a sweep can mark the method/fold as skipped and continue.
    """

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
        return _empty_bushmeg_target_calibration_split(
            per_class=per_class_count,
            seed=seed_value,
            min_evaluation_per_class=min_eval_count,
            context=context_tuple,
            skip_reason="Protocol 3 target calibration requires per_class >= 1.",
            skip_reason_code="invalid_per_class",
        )
    if min_eval_count < 1:
        return _empty_bushmeg_target_calibration_split(
            per_class=per_class_count,
            seed=seed_value,
            min_evaluation_per_class=min_eval_count,
            context=context_tuple,
            skip_reason="Protocol 3 target calibration requires min_evaluation_per_class >= 1.",
            skip_reason_code="invalid_min_evaluation_per_class",
        )

    labels_array = np.asarray(target_labels)
    if labels_array.ndim != 1:
        return _empty_bushmeg_target_calibration_split(
            per_class=per_class_count,
            seed=seed_value,
            min_evaluation_per_class=min_eval_count,
            context=context_tuple,
            skip_reason="Protocol 3 target labels must be one-dimensional.",
            skip_reason_code="labels_not_one_dimensional",
        )
    if labels_array.size == 0:
        return _empty_bushmeg_target_calibration_split(
            per_class=per_class_count,
            seed=seed_value,
            min_evaluation_per_class=min_eval_count,
            context=context_tuple,
            skip_reason="Protocol 3 target labels must not be empty.",
            skip_reason_code="empty_labels",
        )

    classes = np.unique(labels_array)
    required_per_class = per_class_count + min_eval_count
    for class_value in classes:
        class_count = int(np.count_nonzero(labels_array == class_value))
        if class_count < required_per_class:
            readable_class = _readable_label_value(class_value)
            return _empty_bushmeg_target_calibration_split(
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

    effective_seed = _stable_target_calibration_seed(seed_value, per_class=per_class_count, context=context_tuple)
    rng = np.random.default_rng(effective_seed)
    calibration: list[int] = []
    evaluation_mask = np.ones(labels_array.shape[0], dtype=bool)
    for class_value in classes:
        class_indices = np.flatnonzero(labels_array == class_value)
        selected = rng.choice(class_indices, size=per_class_count, replace=False)
        calibration.extend(int(index) for index in selected)
        evaluation_mask[selected] = False

    calibration_indices = np.asarray(sorted(calibration), dtype=int)
    evaluation_indices = np.flatnonzero(evaluation_mask).astype(int, copy=False)
    if np.intersect1d(calibration_indices, evaluation_indices).size:
        return _empty_bushmeg_target_calibration_split(
            per_class=per_class_count,
            seed=seed_value,
            min_evaluation_per_class=min_eval_count,
            context=context_tuple,
            skip_reason="Protocol 3 target calibration/evaluation rows overlap after selection.",
            skip_reason_code="overlapping_rows",
            n_classes=int(classes.size),
        )

    for class_value in classes:
        evaluation_count = int(np.count_nonzero(labels_array[evaluation_indices] == class_value))
        if evaluation_count < min_eval_count:
            readable_class = _readable_label_value(class_value)
            return _empty_bushmeg_target_calibration_split(
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

    return BushmegTargetCalibrationSplit(
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


def category3_calibration_evaluation_split(
    labels: Sequence[Any] | np.ndarray,
    *,
    calibration_per_class: int = 1,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disjoint calibration/evaluation row indices for Protocol 3."""

    split = select_bushmeg_target_calibration_split(
        labels,
        per_class=calibration_per_class,
        seed=seed,
        min_evaluation_per_class=1,
    )
    if split.skipped:
        raise ValueError(split.skip_reason)
    validate_disjoint_calibration_evaluation(split.calibration_indices, split.evaluation_indices)
    return split.calibration_indices, split.evaluation_indices


def _alignment_update(method: str, *, anchor_mode: str = "class_mean", target_projection: str = "group_projection") -> dict[str, Any]:
    return {
        "source_loso": {
            "alignment_method": method,
            "alignment_anchor_mode": anchor_mode,
            "alignment_times": "same_decode_window",
            "alignment_target_projection": target_projection,
        }
    }


def _source_loso_decoder_update(decoders: Sequence[str]) -> dict[str, Any]:
    return {
        "source_loso": {
            "alignment_method": "none",
            "alignment_target_projection": "group_projection",
            "skip_inner_selection_when_single_candidate": True,
            "candidate_grid": {"decoders": list(decoders)},
        }
    }


def _source_loso_calibration_update(mode: str) -> dict[str, Any]:
    class_bias = "none" if mode == "none" else "balanced_accuracy"
    return {
        "source_loso": {
            "alignment_method": "none",
            "alignment_target_projection": "group_projection",
            "candidate_grid": {"class_bias_modes": [class_bias]},
        }
    }


def _few_shot_protocol3_update(k_per_class: int) -> dict[str, Any]:
    return {
        "protocol3": {
            "target_calibration_per_class": int(k_per_class),
            "target_calibration_seed": 13,
            "min_evaluation_per_class": 1,
            "target_repeats": 1,
        }
    }


def _source_plus_target_protocol3_update(decoder: str, k_per_class: int) -> dict[str, Any]:
    update = _few_shot_protocol3_update(k_per_class)
    update["source_loso"] = {
        "skip_inner_selection_when_single_candidate": True,
        "candidate_grid": {"decoders": [decoder]},
    }
    return update


def _target_calibrated_alignment_protocol3_update(method: str, k_per_class: int) -> dict[str, Any]:
    update = _few_shot_protocol3_update(k_per_class)
    update["protocol3"].update({"alignment_method": method})
    return update


def _target_calibrated_gaussian_protocol3_update(k_per_class: int) -> dict[str, Any]:
    update = _few_shot_protocol3_update(k_per_class)
    update["protocol3"].update({
        "augmentation_method": "target_calibrated_gaussian",
        "synthetic_per_class": 8,
        "noise_scale": 1.0,
        "covariance_shrinkage": 1.0,
        "target_calibration_weight": 0.5,
    })
    return update


def _unavailable_method(
    method: str,
    family: str,
    protocol: int,
    *,
    required_modules: Sequence[str] = (),
    requires_torch: bool = False,
    required_config_any: Sequence[str] = (),
    reason: str = "No BUSH-MEG all-protocol fold adapter is implemented for this family in this checkout.",
) -> MethodSpec:
    return MethodSpec(
        method,
        family,
        protocol,
        "unavailable",
        runnable=False,
        blocked_reason=reason,
        required_modules=tuple(required_modules),
        requires_torch=requires_torch,
        required_config_any=tuple(required_config_any),
    )


def method_registry(*, include_inventory_blocked: bool = True) -> dict[str, MethodSpec]:
    """Return known BUSH-MEG method inventory for the unified runner."""

    registry: dict[str, MethodSpec] = {}

    def add(spec: MethodSpec) -> None:
        registry[spec.method] = spec

    add(MethodSpec("source_loso_raw", "source_loso", 1, "source_loso", _alignment_update("none")))
    add(MethodSpec("source_loso_logistic", "source_loso_decoder", 1, "source_loso", _source_loso_decoder_update(["logistic"])))
    add(MethodSpec("source_loso_linear_svm", "source_loso_decoder", 1, "source_loso", _source_loso_decoder_update(["linear_svm"])))
    add(
        MethodSpec(
            "source_loso_correlation_prototype",
            "source_loso_decoder",
            1,
            "source_loso",
            _source_loso_decoder_update(["correlation-prototype"]),
        )
    )
    add(MethodSpec("source_loso_decoder_ensemble", "source_loso_decoder_ensemble", 1, "loso_decode"))
    add(
        MethodSpec(
            "source_loso_response_window_c",
            "source_loso",
            1,
            "source_loso",
            {
                "source_loso": {
                    "alignment_method": "none",
                    "alignment_target_projection": "group_projection",
                    "candidate_grid": {
                        "window_sets": [
                            {
                                "name": "response_window_c",
                                "centers": list(RESPONSE_WINDOW_C),
                                "window_size": 0.100,
                            }
                        ],
                        "window_combines": ["log_probability_mean"],
                    },
                }
            },
        )
    )
    add(MethodSpec("memory_bounded_loso_decode", "windowed_loso_decode", 1, "loso_decode"))
    add(MethodSpec("covariance_loso", "covariance_loso", 1, "covariance_loso"))
    add(MethodSpec("supervised_lowrank_loso", "supervised_lowrank_loso", 1, "supervised_lowrank_loso"))

    add(MethodSpec("source_probability_calibration_none", "source_probability_calibration", 1, "source_loso", _source_loso_calibration_update("none")))
    add(MethodSpec("source_probability_calibration_class_bias", "source_probability_calibration", 1, "source_loso", _source_loso_calibration_update("class_bias")))
    for calibration in ("temperature", "temperature_plus_class_bias", "confusion_correction_l2"):
        add(
            _unavailable_method(
                f"source_probability_calibration_{calibration}",
                "source_probability_calibration",
                1,
                reason=(
                    "Source probability calibration mode exists in the generic MNE time decoder, "
                    "but it is not wired into the current BUSH source_loso runner."
                ),
            )
        )

    for method in ("procrustes", "hyperalignment", "mcca"):
        add(
            MethodSpec(
                f"source_alignment_{method}_group_projection",
                "strict_source_alignment",
                1,
                "source_loso",
                _alignment_update(method, anchor_mode="class_mean", target_projection="group_projection"),
                required_modules=("neureptrace.decoding.source_alignment",),
            )
        )
        for anchor in ("class_mean", "class_repetition"):
            add(
                MethodSpec(
                    f"strict_{method}_{anchor}",
                    "strict_source_alignment",
                    1,
                    "source_loso",
                    _alignment_update(method, anchor_mode=anchor, target_projection="group_projection"),
                    required_modules=("neureptrace.decoding.source_alignment",),
                )
            )
            add(
                MethodSpec(
                    f"oracle_{method}_{anchor}",
                    "oracle_target_calibrated_alignment",
                    4,
                    "source_loso",
                    _alignment_update(method, anchor_mode=anchor, target_projection="oracle_target_calibrated_alignment"),
                    required_modules=("neureptrace.decoding.source_alignment",),
                )
            )
        add(
            MethodSpec(
                f"target_calibrated_{method}_class_mean",
                "target_calibrated_alignment",
                3,
                "unavailable",
                _alignment_update(method, anchor_mode="class_mean", target_projection="target_calibrated_alignment"),
                runnable=False,
                blocked_reason=(
                    "source_alignment supports disjoint target calibration rows, but the current "
                    "BUSH source_loso path does not pass target calibration features into the fold evaluator"
                ),
                required_modules=("neureptrace.decoding.source_alignment",),
            )
        )
        add(
            MethodSpec(
                f"target_calibrated_{method}",
                "target_calibrated_alignment",
                3,
                "unavailable",
                _alignment_update(method, anchor_mode="class_mean", target_projection="target_calibrated_alignment"),
                runnable=False,
                blocked_reason=(
                    "source_alignment supports disjoint target calibration rows, but the current "
                    "BUSH source_loso path does not pass target calibration features into the fold evaluator"
                ),
                required_modules=("neureptrace.decoding.source_alignment",),
            )
        )
        add(
            MethodSpec(
                f"oracle_target_calibrated_{method}",
                "oracle_target_calibrated_alignment",
                4,
                "source_loso",
                _alignment_update(method, anchor_mode="class_mean", target_projection="oracle_target_calibrated_alignment"),
                required_modules=("neureptrace.decoding.source_alignment",),
            )
        )

    for method in ("euclidean", "coral", "target_baseline_covariance", "subject_sensor_covariance"):
        add(
            MethodSpec(
                f"unlabeled_{method}",
                "unlabeled_covariance_alignment",
                2,
                "source_loso",
                _alignment_update(method, anchor_mode="class_mean", target_projection="group_projection"),
                required_modules=("neureptrace.decoding.source_alignment",),
            )
        )
    add(MethodSpec("euclidean_alignment", "unlabeled_covariance_alignment", 2, "source_loso", _alignment_update("euclidean"), required_modules=("neureptrace.decoding.source_alignment",)))
    add(MethodSpec("coral_alignment", "unlabeled_covariance_alignment", 2, "source_loso", _alignment_update("coral"), required_modules=("neureptrace.decoding.source_alignment",)))
    add(MethodSpec("target_baseline_covariance", "unlabeled_covariance_alignment", 2, "source_loso", _alignment_update("target_baseline_covariance"), required_modules=("neureptrace.decoding.source_alignment",)))
    add(MethodSpec("subject_sensor_covariance", "unlabeled_covariance_alignment", 2, "source_loso", _alignment_update("subject_sensor_covariance"), required_modules=("neureptrace.decoding.source_alignment",)))

    for k in (1, 2, 4, 8, 16):
        add(
            MethodSpec(
                f"few_shot_target_calibrated_decoder_k{k}",
                "few_shot_target_calibration",
                3,
                "protocol3_few_shot",
                _few_shot_protocol3_update(k),
                required_modules=("neureptrace.decoding.few_shot",),
            )
        )
        for decoder in ("logistic", "linear_svm"):
            add(
                MethodSpec(
                    f"source_plus_target_calibration_{decoder}_k{k}",
                    "source_plus_target_calibration",
                    3,
                    "protocol3_source_plus_target",
                    _source_plus_target_protocol3_update(decoder, k),
                )
            )
        for alignment_method in ("procrustes", "hyperalignment", "mcca"):
            add(
                MethodSpec(
                    f"target_calibrated_{alignment_method}_k{k}",
                    "target_calibrated_alignment",
                    3,
                    "protocol3_target_calibrated_alignment",
                    _target_calibrated_alignment_protocol3_update(alignment_method, k),
                    required_modules=("neureptrace.decoding.source_alignment",),
                )
            )
        add(
            MethodSpec(
                f"target_calibrated_gaussian_k{k}",
                "generative_augmentation",
                3,
                "protocol3_target_calibrated_gaussian",
                _target_calibrated_gaussian_protocol3_update(k),
                required_modules=("neureptrace.decoding.generative_augmentation",),
            )
        )
        add(
            MethodSpec(
                f"semi_supervised_lora_few_shot_k{k}",
                "semi_supervised_lora_few_shot",
                3,
                "protocol3_lora_few_shot",
                _few_shot_protocol3_update(k),
                required_modules=("neureptrace.decoding.semi_supervised_lora_few_shot",),
                requires_torch=True,
            )
        )

    if include_inventory_blocked:
        autoencoder_available = importlib.util.find_spec("neureptrace.bushmeg_category2_autoencoder_loso") is not None
        add(
            MethodSpec(
                "category2_autoencoder_loso",
                "category2_autoencoder",
                2,
                "unavailable",
                runnable=False,
                blocked_reason=(
                    "Category 2 autoencoder implementation exists, but no BUSH-MEG all-protocol fold adapter "
                    "is implemented for this family in this checkout."
                    if autoencoder_available
                    else "neureptrace.bushmeg_category2_autoencoder_loso is not present in this checkout"
                ),
                required_modules=("neureptrace.bushmeg_category2_autoencoder_loso",),
            )
        )
        # Protocol 1 inventory-only families.
        add(_unavailable_method("contrastive_group_projection", "contrastive_alignment", 1, required_modules=("neureptrace.decoding.unlabeled_calibration_alignment",)))
        add(_unavailable_method("source_domain_generalization_erm", "source_domain_generalization", 1, required_modules=("neureptrace.decoding.source_domain_generalization",)))
        add(_unavailable_method("source_domain_generalization_subject_adversarial", "source_domain_generalization", 1, required_modules=("neureptrace.decoding.source_domain_generalization",), requires_torch=True))
        add(_unavailable_method("source_domain_generalization_group_dro", "source_domain_generalization", 1, required_modules=("neureptrace.decoding.source_domain_generalization",)))
        add(_unavailable_method("reconstruction_source_only", "reconstruction_encoder", 1, required_modules=("neureptrace.decoding.reconstruction_encoder",), requires_torch=True))
        add(_unavailable_method("generative_source_gaussian", "generative_augmentation", 1, required_modules=("neureptrace.decoding.generative_augmentation",)))
        add(_unavailable_method("generative_source_gan", "generative_augmentation", 1, required_modules=("neureptrace.decoding.generative_augmentation",), requires_torch=True))
        add(_unavailable_method("generative_source_diffusion", "generative_augmentation", 1, required_modules=("neureptrace.decoding.generative_augmentation",), requires_torch=True))
        add(
            _unavailable_method(
                "foundation_frozen_linear_probe",
                "foundation",
                1,
                required_modules=("neureptrace.decoding.foundation",),
                required_config_any=("foundation.model_path", "foundation.config", "foundation_model_path", "decoding.foundation_model_path"),
                reason="Requires a foundation model path/config and a BUSH-MEG foundation-model adapter.",
            )
        )

        # Protocol 2 inventory-only families.
        protocol2_blocked: tuple[tuple[str, str, tuple[str, ...], bool, tuple[str, ...]], ...] = (
            ("sinkhorn_transport", "optimal_transport", ("neureptrace.decoding.mekt",), False, ()),
            ("group_projection_target_centered", "target_centered_group_projection", ("neureptrace.decoding.unlabeled_calibration_alignment",), False, ()),
            ("pseudo_label_target_calibrated_alignment", "pseudo_label_alignment", ("neureptrace.decoding.source_alignment",), False, ()),
            ("pseudo_label_self_training", "pseudo_label_self_training", (), False, ()),
            ("riemannian_tangent_transfer", "riemannian", ("neureptrace.decoding.riemannian",), False, ()),
            ("riemannian_procrustes_no_rotation", "riemannian", ("neureptrace.decoding.riemannian",), False, ()),
            (
                "riemannian_procrustes_paired_unlabeled",
                "riemannian",
                ("neureptrace.decoding.riemannian",),
                False,
                ("unlabeled_calibration.anchors", "unlabeled_calibration.anchor_column", "decoding.unlabeled_calibration_anchor_column"),
            ),
            ("mekt", "mekt", ("neureptrace.decoding.mekt",), False, ()),
            ("optimal_transport_sinkhorn", "optimal_transport", ("neureptrace.decoding.mekt",), False, ()),
            ("source_weighting_target_similarity", "source_weighting", ("neureptrace.decoding.source_weighting",), False, ()),
            ("source_weighting_hybrid", "source_weighting", ("neureptrace.decoding.source_weighting",), False, ()),
            ("dann", "domain_adversarial", ("neureptrace.decoding.dann",), True, ()),
            ("cdan", "conditional_domain_adversarial", ("neureptrace.decoding.cdan",), True, ()),
            ("cdan_mmd", "conditional_domain_adversarial", ("neureptrace.decoding.cdan",), True, ()),
            ("cdan_conditional_mmd", "conditional_domain_adversarial", ("neureptrace.decoding.cdan",), True, ()),
            ("ttime_after_predict", "test_time_adaptation", ("neureptrace.decoding.test_time_adaptation",), True, ()),
            ("ttime_before_predict", "test_time_adaptation", ("neureptrace.decoding.test_time_adaptation",), True, ()),
            ("source_free_adaptation", "source_free", ("neureptrace.decoding.source_free",), True, ()),
            ("reconstruction_source_plus_target", "reconstruction_encoder", ("neureptrace.decoding.reconstruction_encoder",), True, ()),
            ("unlabeled_calibration_hyperalignment", "unlabeled_calibration_alignment", ("neureptrace.decoding.unlabeled_calibration_alignment",), False, ()),
            ("unlabeled_calibration_mcca", "unlabeled_calibration_alignment", ("neureptrace.decoding.unlabeled_calibration_alignment",), False, ()),
            ("unlabeled_calibration_procrustes", "unlabeled_calibration_alignment", ("neureptrace.decoding.unlabeled_calibration_alignment",), False, ()),
            ("target_style_gaussian", "generative_augmentation", ("neureptrace.decoding.generative_augmentation",), False, ()),
            ("target_style_gan", "generative_augmentation", ("neureptrace.decoding.generative_augmentation",), True, ()),
            ("target_style_diffusion", "generative_augmentation", ("neureptrace.decoding.generative_augmentation",), True, ()),
            (
                "weak_label_proportion_calibration",
                "label_proportions",
                ("neureptrace.decoding.label_proportions",),
                False,
                ("label_proportions.target_proportions", "target_proportions", "decoding.target_proportions"),
            ),
        )
        for method, family, modules, requires_torch, required_config_any in protocol2_blocked:
            add(
                _unavailable_method(
                    method,
                    family,
                    2,
                    required_modules=modules,
                    requires_torch=requires_torch,
                    required_config_any=required_config_any,
                )
            )

        # Protocol 3 supervised/calibrated target inventory.
        add(_unavailable_method("contrastive_target_calibrated", "contrastive_alignment", 3, required_modules=("neureptrace.decoding.unlabeled_calibration_alignment",)))
        add(_unavailable_method("few_shot_target_calibrated_decoder", "few_shot", 3, required_modules=("neureptrace.decoding.few_shot",)))
        add(_unavailable_method("semi_supervised_lora_few_shot", "semi_supervised_lora_few_shot", 3, required_modules=("neureptrace.decoding.semi_supervised_lora_few_shot",), requires_torch=True))
        add(_unavailable_method("target_calibrated_gaussian", "generative_augmentation", 3, required_modules=("neureptrace.decoding.generative_augmentation",)))
        add(_unavailable_method("target_calibrated_gan", "generative_augmentation", 3, required_modules=("neureptrace.decoding.generative_augmentation",), requires_torch=True))
        add(_unavailable_method("target_calibrated_diffusion", "generative_augmentation", 3, required_modules=("neureptrace.decoding.generative_augmentation",), requires_torch=True))
    return registry


def _split_csv(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [token.strip() for token in value.split(",") if token.strip()]
    return [str(token).strip() for token in value if str(token).strip()]


def _parse_protocols(value: str | Sequence[str | int] | None) -> set[int]:
    tokens = _split_csv(value)
    if not tokens:
        return {1, 2, 3}
    protocols = {int(token) for token in tokens}
    unknown = protocols.difference(PROTOCOLS)
    if unknown:
        raise ValueError(f"Unknown protocol category/categories: {sorted(unknown)}.")
    return protocols


def _deep_merge(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _config_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {}) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section {name!r} must be a mapping.")
    return dict(value)


def _set_nested(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    cursor = config
    parts = [part for part in dotted_path.split(".") if part]
    for part in parts[:-1]:
        child = cursor.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"Cannot set {dotted_path!r}: {part!r} is not a mapping.")
        cursor = child
    cursor[parts[-1]] = value


def _yaml_safe_dump(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")


def _validate_timeout_seconds(name: str, value: float | int | None) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when provided.")
    return parsed


def _signal_timeouts_supported() -> bool:
    return (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )


class MethodProgress:
    """Write method-local progress artifacts for long BUSH-MEG runs."""

    def __init__(
        self,
        method_dir: Path,
        *,
        method: str,
        aggregate_callback: Callable[[], None] | None = None,
        method_timeout_seconds: float | None = None,
        fold_timeout_seconds: float | None = None,
    ) -> None:
        self.method_dir = method_dir
        self.method = method
        self.aggregate_callback = aggregate_callback
        self.method_timeout_seconds = _validate_timeout_seconds("method_timeout_seconds", method_timeout_seconds)
        self.fold_timeout_seconds = _validate_timeout_seconds("fold_timeout_seconds", fold_timeout_seconds)
        self.status_path = method_dir / "status.json"
        self.log_path = method_dir / "run.log"
        self.summary_partial_path = method_dir / "summary.partial.csv"
        self.predictions_partial_path = method_dir / "predictions.partial.csv"
        self.inner_partial_path = method_dir / "inner_cv.partial.csv"
        self._event_count = 0
        self._method_deadline: float | None = None
        self._fold_deadline: float | None = None
        self._fold_context: dict[str, Any] = {}
        self._previous_signal_handler: Any = None
        self._previous_timer: tuple[float, float] | None = None
        self._signal_installed = False

    def initialize_artifacts(self) -> None:
        self.method_dir.mkdir(parents=True, exist_ok=True)
        for path in (self.summary_partial_path, self.predictions_partial_path):
            if not path.exists():
                path.write_text("", encoding="utf-8")
        if not self.log_path.exists():
            self.log_path.write_text("", encoding="utf-8")

    def update(self, stage: str, **fields: Any) -> None:
        if stage not in PROGRESS_STAGES:
            raise ValueError(f"Unknown progress stage {stage!r}.")
        self.initialize_artifacts()
        if stage == "fold_done":
            self._fold_deadline = None
            self._fold_context = {}
            self._refresh_signal_timer()
        if stage in {"method_done", "method_failed", "method_skipped"}:
            self.clear_timeouts()
        elif stage != "fold_done":
            self._raise_if_timeout_elapsed()
        self._event_count += 1
        now = datetime.now(UTC).isoformat()
        event = {
            "event_index": self._event_count,
            "updated_at_utc": now,
            "method": self.method,
            "stage": stage,
            **fields,
        }
        status = {
            **event,
            "status_json": str(self.status_path),
            "run_log": str(self.log_path),
            "summary_partial_csv": str(self.summary_partial_path),
            "predictions_partial_csv": str(self.predictions_partial_path),
        }
        tmp_path = self.status_path.with_suffix(f".json.{os.getpid()}.{threading.get_ident()}.{self._event_count}.tmp")
        tmp_path.write_text(json.dumps(status, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        _replace_with_retries(tmp_path, self.status_path)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")
        if stage in AGGREGATE_PROGRESS_STAGES and self.aggregate_callback is not None:
            self.aggregate_callback()
        if stage == "fold_start" and self.fold_timeout_seconds is not None:
            self._fold_context = {key: value for key, value in fields.items() if key in {"outer_test_subject", "fold_index", "n_folds"}}
            self._fold_deadline = time.monotonic() + self.fold_timeout_seconds
            self._refresh_signal_timer()

    def __call__(self, stage: str, **fields: Any) -> None:
        self.update(stage, **fields)

    def start_method_timeout(self) -> None:
        if self.method_timeout_seconds is None:
            return
        self._method_deadline = time.monotonic() + self.method_timeout_seconds
        self._install_signal_handler()
        self._refresh_signal_timer()

    def clear_timeouts(self) -> None:
        self._method_deadline = None
        self._fold_deadline = None
        self._fold_context = {}
        self._clear_signal_timer()

    def _install_signal_handler(self) -> None:
        if self._signal_installed or not _signal_timeouts_supported():
            return
        self._previous_signal_handler = signal.getsignal(signal.SIGALRM)
        self._previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self._handle_signal_timeout)
        self._signal_installed = True

    def _clear_signal_timer(self) -> None:
        if not self._signal_installed or not _signal_timeouts_supported():
            return
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        if self._previous_signal_handler is not None:
            signal.signal(signal.SIGALRM, self._previous_signal_handler)
        if self._previous_timer is not None and self._previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, self._previous_timer[0], self._previous_timer[1])
        self._previous_signal_handler = None
        self._previous_timer = None
        self._signal_installed = False

    def _refresh_signal_timer(self) -> None:
        if not _signal_timeouts_supported():
            return
        if self._method_deadline is not None or self._fold_deadline is not None:
            self._install_signal_handler()
        if not self._signal_installed:
            return
        deadline = self._next_deadline()
        if deadline is None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            return
        delay = max(0.001, deadline - time.monotonic())
        signal.setitimer(signal.ITIMER_REAL, delay)

    def _next_deadline(self) -> float | None:
        deadlines = [deadline for deadline in (self._method_deadline, self._fold_deadline) if deadline is not None]
        return min(deadlines) if deadlines else None

    def _current_timeout(self) -> tuple[str, float, dict[str, Any]]:
        now = time.monotonic()
        fold_over = self._fold_deadline is not None and now >= self._fold_deadline
        method_over = self._method_deadline is not None and now >= self._method_deadline
        if fold_over or (self._fold_deadline is not None and self._method_deadline is not None and self._fold_deadline <= self._method_deadline):
            return "fold", float(self.fold_timeout_seconds or 0.0), dict(self._fold_context)
        if method_over or self._method_deadline is not None:
            return "method", float(self.method_timeout_seconds or 0.0), {"method": self.method}
        return "method", 0.0, {"method": self.method}

    def _raise_if_timeout_elapsed(self) -> None:
        now = time.monotonic()
        if self._fold_deadline is not None and now >= self._fold_deadline:
            raise RunTimeoutError(kind="fold", seconds=float(self.fold_timeout_seconds or 0.0), context=self._fold_context)
        if self._method_deadline is not None and now >= self._method_deadline:
            raise RunTimeoutError(kind="method", seconds=float(self.method_timeout_seconds or 0.0), context={"method": self.method})

    def _handle_signal_timeout(self, signum: int, frame: Any) -> None:
        kind, seconds, context = self._current_timeout()
        raise RunTimeoutError(kind=kind, seconds=seconds, context=context)


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def _read_csv_if_nonempty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_with_retries(tmp_path: Path, target_path: Path, *, attempts: int = 25, delay_seconds: float = 0.08) -> None:
    """Replace a file, tolerating transient Windows locks from readers."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for attempt in range(attempts):
        try:
            tmp_path.replace(target_path)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(delay_seconds * (1.0 + attempt / 5.0))
    try:
        tmp_path.unlink(missing_ok=True)
    finally:
        if last_error is not None:
            raise last_error


def _write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.{os.getpid()}.{threading.get_ident()}.{time.monotonic_ns()}.tmp")
    frame.to_csv(tmp_path, index=False)
    _replace_with_retries(tmp_path, path)


def _append_csv_rows(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.empty:
        if not path.exists():
            path.write_text("", encoding="utf-8")
        return
    write_header = not path.exists() or path.stat().st_size == 0
    frame.to_csv(path, mode="a", header=write_header, index=False)


def _load_runner_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    raw = load_config(config_path)
    all_protocols = dict(raw.get("all_protocols", {}))
    base_config_value = all_protocols.get("base_config", raw.get("base_config"))
    if base_config_value:
        base_path = Path(str(base_config_value))
        if not base_path.is_absolute():
            base_path = config_path.parent / base_path
        return load_config(base_path), all_protocols, base_path
    return raw, all_protocols, config_path


def _resolve_data_dir(config: Mapping[str, Any], data_dir: str | Path | None) -> Path | None:
    if data_dir is not None:
        return Path(data_dir)
    env = os.environ.get("BUSH_MEG_DATA_DIR")
    if env:
        return Path(env)
    dataset = config.get("dataset", {})
    if isinstance(dataset, Mapping) and dataset.get("root"):
        expanded = os.path.expanduser(os.path.expandvars(str(dataset["root"])))
        if "$" not in expanded and "{" not in expanded:
            return Path(expanded)
    return None


def _apply_common_config_updates(
    config: dict[str, Any],
    *,
    data_dir: str | Path | None,
    participants: str | Sequence[str] | None,
    participant_limit: int | None = None,
    smoke_participants: str | Sequence[str] | None = None,
    window_limit: int | None = None,
) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    resolved_data_dir = _resolve_data_dir(updated, data_dir)
    if resolved_data_dir is not None:
        _set_nested(updated, "dataset.root", str(resolved_data_dir))
    configured_participants = participants
    if configured_participants is None:
        participant_block = updated.get("participants", {})
        if isinstance(participant_block, Mapping):
            configured_participants = participant_block.get("ids")
    limited_participants = _limited_participant_ids(
        configured_participants,
        participant_limit=participant_limit,
        smoke_participants=smoke_participants,
    )
    if limited_participants is not None:
        _set_nested(updated, "participants.ids", limited_participants)
    _apply_window_limit(updated, window_limit)
    return updated


def _validate_positive_limit(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return parsed


def _participant_ids_to_config_value(ids: Sequence[int | str]) -> str:
    return ",".join(str(item) for item in ids)


def _limited_participant_ids(
    participants: str | Sequence[str] | Sequence[int] | None,
    *,
    participant_limit: int | None = None,
    smoke_participants: str | Sequence[str] | Sequence[int] | None = None,
) -> str | None:
    participant_limit = _validate_positive_limit("participant_limit", participant_limit)
    selected = smoke_participants if smoke_participants is not None else participants
    if selected is None:
        if participant_limit is None:
            return None
        raise ValueError("--participant-limit requires configured participants, --participants, or --smoke-participants.")
    parsed = parse_participant_ids(selected)
    if participant_limit is not None:
        parsed = parsed[:participant_limit]
    if not parsed:
        raise ValueError("Participant selection is empty after applying tiny-run limits.")
    return _participant_ids_to_config_value(parsed)


def _limit_sequence(value: Any, limit: int) -> Any:
    if isinstance(value, list):
        return value[:limit]
    if isinstance(value, tuple):
        return list(value[:limit])
    return value


def _apply_window_limit(config: dict[str, Any], window_limit: int | None) -> None:
    window_limit = _validate_positive_limit("window_limit", window_limit)
    if window_limit is None:
        return

    preprocessing = config.get("preprocessing")
    if isinstance(preprocessing, dict):
        if "window_centers" in preprocessing:
            preprocessing["window_centers"] = _limit_sequence(preprocessing["window_centers"], window_limit)
        if "diagnostics_times" in preprocessing:
            preprocessing["diagnostics_times"] = _limit_sequence(preprocessing["diagnostics_times"], window_limit)

    source_loso = config.get("source_loso")
    if not isinstance(source_loso, dict):
        return
    if isinstance(source_loso.get("alignment_times"), (list, tuple)):
        source_loso["alignment_times"] = _limit_sequence(source_loso["alignment_times"], window_limit)
    candidate_grid = source_loso.get("candidate_grid")
    if not isinstance(candidate_grid, dict):
        return
    window_sets = candidate_grid.get("window_sets")
    if isinstance(window_sets, list):
        for window_set in window_sets:
            if isinstance(window_set, dict) and "centers" in window_set:
                window_set["centers"] = _limit_sequence(window_set["centers"], window_limit)


def _participant_ids_from_config(config: Mapping[str, Any]) -> str | None:
    participants = config.get("participants", {})
    if not isinstance(participants, Mapping):
        return None
    ids = participants.get("ids")
    if ids is None:
        return None
    if isinstance(ids, str):
        return ids
    return _participant_ids_to_config_value(parse_participant_ids(ids))


def _method_config(
    base_config: Mapping[str, Any],
    all_protocols: Mapping[str, Any],
    spec: MethodSpec,
    *,
    data_dir: str | Path | None,
    participants: str | Sequence[str] | None,
    max_folds: int | None = None,
    include_heavy: bool = False,
    participant_limit: int | None = None,
    smoke_participants: str | Sequence[str] | None = None,
    window_limit: int | None = None,
) -> dict[str, Any]:
    config = _apply_common_config_updates(
        dict(base_config),
        data_dir=data_dir,
        participants=participants,
        participant_limit=participant_limit,
        smoke_participants=smoke_participants,
        window_limit=window_limit,
    )
    config = _deep_merge(config, spec.config_updates)
    method_overrides = (all_protocols.get("method_overrides", {}) or {}).get(spec.method, {})
    if method_overrides:
        if not isinstance(method_overrides, Mapping):
            raise ValueError(f"all_protocols.method_overrides.{spec.method} must be a mapping.")
        config = _deep_merge(config, method_overrides)
    settings = _method_settings(all_protocols, spec.method)
    if max_folds is not None and bool(settings.get("smoke_enabled", False)) and not include_heavy:
        smoke_overrides = settings.get("smoke_overrides", {}) or {}
        if not isinstance(smoke_overrides, Mapping):
            raise ValueError(f"all_protocols.method_settings.{spec.method}.smoke_overrides must be a mapping.")
        config = _deep_merge(config, smoke_overrides)
    _apply_window_limit(config, window_limit)
    return config


def _call_with_supported_kwargs(function: Callable[..., Any], **kwargs: Any) -> Any:
    parameters = inspect.signature(function).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return function(**kwargs)
    return function(**{key: value for key, value in kwargs.items() if key in parameters})


def _participant_count_from_config(config: Mapping[str, Any]) -> int | None:
    participants = config.get("participants", {})
    if not isinstance(participants, Mapping):
        return None
    ids = participants.get("ids")
    if ids is None:
        return None
    try:
        return len(parse_participant_ids(ids))
    except ValueError:
        return None


def _run_source_loso_method(
    config_path: Path,
    *,
    summary_path: Path,
    inner_path: Path,
    predictions_path: Path,
    max_folds: int | None,
    progress_callback: Callable[..., None] | None = None,
) -> pd.DataFrame:
    from neureptrace.bushmeg_source_loso import run_bushmeg_source_loso

    return _call_with_supported_kwargs(
        run_bushmeg_source_loso,
        config_path=config_path,
        out_path=summary_path,
        inner_cv_out_path=inner_path,
        predictions_out_path=predictions_path,
        max_folds=max_folds,
        progress_callback=progress_callback,
    )


def _shape_list(array: Any) -> list[int]:
    return [int(dim) for dim in getattr(array, "shape", ())]


def _memory_mib(*arrays: Any) -> float:
    return float(sum(int(getattr(array, "nbytes", 0)) for array in arrays) / (1024.0 * 1024.0))


def _class_count_dict(labels: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(np.asarray(labels), return_counts=True)
    return {str(value): int(count) for value, count in zip(values, counts, strict=True)}


def _profile_base_config(
    *,
    config_path: str | Path,
    data_dir: str | Path | None,
    participants: str | Sequence[str] | None,
    participant_limit: int | None,
    smoke_participants: str | Sequence[str] | None = None,
    window_limit: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config_path = Path(config_path)
    base_config, all_protocols, base_config_path = _load_runner_config(config_path)
    config = _apply_common_config_updates(
        base_config,
        data_dir=data_dir,
        participants=participants,
        participant_limit=participant_limit,
        smoke_participants=smoke_participants,
        window_limit=window_limit,
    )
    return config, all_protocols, base_config_path


def profile_bushmeg_load_only(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    data_dir: str | Path | None = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    participants: str | Sequence[str] | None = None,
    participant_limit: int | None = None,
    smoke_participants: str | Sequence[str] | None = None,
    window_limit: int | None = None,
) -> Path:
    """Load requested BUSH-MEG participants, write per-subject load profile, and exit."""

    from neureptrace.bushmeg_source_loso import FeatureCache, _load_subjects_from_config

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config, _all_protocols, base_config_path = _profile_base_config(
        config_path=config_path,
        data_dir=data_dir,
        participants=participants,
        participant_limit=participant_limit,
        smoke_participants=smoke_participants,
        window_limit=window_limit,
    )
    _yaml_safe_dump(output_dir / "profile_config.yml", config)
    load_starts: dict[int, float] = {}
    load_paths: dict[int, str] = {}
    load_participants: dict[int, str] = {}
    load_seconds: dict[str, float] = {}

    def progress(stage: str, **fields: Any) -> None:
        if stage == "loading_subjects" and "subject_index" in fields:
            subject_index = int(fields["subject_index"])
            load_starts[subject_index] = time.perf_counter()
            load_paths[subject_index] = str(fields.get("path", ""))
            load_participants[subject_index] = str(fields.get("participant", ""))
        elif stage == "loaded_subject" and "subject_index" in fields:
            subject_index = int(fields["subject_index"])
            started = load_starts.get(subject_index)
            if started is not None:
                load_seconds[str(fields.get("subject", load_participants.get(subject_index, subject_index)))] = time.perf_counter() - started

    started_total = time.perf_counter()
    subjects, encoder = _load_subjects_from_config(config, config_dir=base_config_path.parent, progress_callback=progress)
    FeatureCache(subjects)
    rows: list[dict[str, Any]] = []
    for subject_id in sorted(subjects):
        subject = subjects[subject_id]
        times = np.asarray(subject.times)
        labels = np.asarray(subject.labels)
        data = np.asarray(subject.data)
        subject_index = list(sorted(subjects)).index(subject_id) + 1
        rows.append(
            {
                "subject": subject_id,
                "participant": subject_id,
                "path": load_paths.get(subject_index, ""),
                "load_seconds": load_seconds.get(subject_id, np.nan),
                "data_shape": "x".join(map(str, data.shape)),
                "labels_shape": "x".join(map(str, labels.shape)),
                "times_shape": "x".join(map(str, times.shape)),
                "n_trials": int(data.shape[0]),
                "n_channels": int(data.shape[1]) if data.ndim >= 2 else 0,
                "n_times": int(data.shape[2]) if data.ndim >= 3 else int(times.size),
                "time_start": float(times[0]) if times.size else np.nan,
                "time_stop": float(times[-1]) if times.size else np.nan,
                "time_step_median": float(np.median(np.diff(times))) if times.size > 1 else np.nan,
                "data_dtype": str(data.dtype),
                "labels_dtype": str(labels.dtype),
                "memory_mib": _memory_mib(data, labels, times),
                "n_classes": int(len(np.unique(labels))),
                "class_counts": json.dumps(_class_count_dict(labels), sort_keys=True),
            }
        )
    profile = pd.DataFrame(rows)
    profile_path = output_dir / "load_profile.csv"
    _write_csv_atomic(profile, profile_path)
    print(f"Loaded {len(subjects)} subject(s) in {time.perf_counter() - started_total:.3f}s.")
    print(f"Classes: {list(map(str, encoder.classes_))}")
    print(profile.to_string(index=False))
    print(f"Wrote load profile: {profile_path}")
    return profile_path


def profile_bushmeg_one_fold(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    data_dir: str | Path | None = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    participants: str | Sequence[str] | None = None,
    methods: str | Sequence[str] | None = None,
    protocols: str | Sequence[str | int] | None = None,
    max_folds: int | None = None,
    participant_limit: int | None = None,
    smoke_participants: str | Sequence[str] | None = None,
    fold_limit: int | None = None,
    window_limit: int | None = None,
    include_oracle: bool = False,
    include_heavy: bool = False,
    non_oracle: bool = False,
) -> Path:
    """Load a tiny BUSH-MEG fold, build one feature matrix, fit one logistic model, and exit."""

    from neureptrace.bushmeg_source_loso import (
        FeatureCache,
        _candidate_grid,
        _candidate_model,
        _load_subjects_from_config,
        _prepare_window_train_test_features,
        _stack_subject_labels,
    )

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config, all_protocols, base_config_path = _load_runner_config(Path(config_path))
    selected = _selected_methods(
        all_protocols=all_protocols,
        methods=methods,
        protocols=protocols,
        include_oracle=include_oracle,
        non_oracle=non_oracle,
    )
    if not selected:
        raise ValueError("--profile-one-fold requires at least one selected method.")
    spec = selected[0]
    if spec.runner != "source_loso":
        raise ValueError(f"--profile-one-fold currently supports source_loso methods only; {spec.method} uses runner {spec.runner!r}.")
    effective_max_folds = _effective_fold_limit(max_folds, fold_limit)
    method_config = _method_config(
        base_config,
        all_protocols,
        spec,
        data_dir=data_dir,
        participants=participants,
        max_folds=effective_max_folds,
        include_heavy=include_heavy,
        participant_limit=participant_limit,
        smoke_participants=smoke_participants,
        window_limit=window_limit,
    )
    _yaml_safe_dump(output_dir / "profile_config.yml", method_config)
    load_start = time.perf_counter()
    subjects, encoder = _load_subjects_from_config(method_config, config_dir=base_config_path.parent)
    load_seconds = time.perf_counter() - load_start
    outer_subjects = sorted(subjects)
    if effective_max_folds is not None:
        outer_subjects = outer_subjects[: max(1, int(effective_max_folds))]
    test_subject = outer_subjects[0]
    train_subjects = [subject for subject in sorted(subjects) if subject != test_subject]
    if not train_subjects:
        raise ValueError("--profile-one-fold requires at least two participants after limits.")
    candidates = _candidate_grid(method_config)
    if not candidates:
        raise ValueError("No source_loso candidates are available for --profile-one-fold.")
    candidate = candidates[0]
    if not candidate.windows:
        raise ValueError(f"Candidate {candidate.name!r} has no windows.")
    window = candidate.windows[0]
    cache = FeatureCache(subjects)
    n_classes = int(len(encoder.classes_))

    feature_start = time.perf_counter()
    train_features, test_features = _prepare_window_train_test_features(
        subjects=subjects,
        cache=cache,
        candidate=candidate,
        train_subjects=train_subjects,
        test_subject=test_subject,
        window=window,
        n_classes=n_classes,
    )
    train_labels = _stack_subject_labels(subjects, train_subjects)
    test_labels = subjects[test_subject].labels
    feature_seconds = time.perf_counter() - feature_start

    fit_start = time.perf_counter()
    decoding = _config_section(method_config, "decoding") or _config_section(method_config, "workflow")
    model = _candidate_model(
        candidate,
        max_iter=int(decoding.get("max_iter", 1000)),
        n_features=int(train_features.shape[1]),
        n_samples=int(train_features.shape[0]),
    )
    model.fit(train_features, train_labels)
    fit_seconds = time.perf_counter() - fit_start
    predict_seconds = np.nan
    test_accuracy = np.nan
    if hasattr(model, "predict"):
        predict_start = time.perf_counter()
        predicted = model.predict(test_features)
        predict_seconds = time.perf_counter() - predict_start
        test_accuracy = float(accuracy_score(test_labels, predicted))

    profile = {
        "method": spec.method,
        "method_family": spec.method_family,
        "protocol_category": int(spec.protocol_category),
        "outer_test_subject": test_subject,
        "train_subjects": train_subjects,
        "n_subjects_loaded": len(subjects),
        "load_seconds": float(load_seconds),
        "feature_seconds": float(feature_seconds),
        "fit_seconds": float(fit_seconds),
        "predict_seconds": None if pd.isna(predict_seconds) else float(predict_seconds),
        "candidate": candidate.name,
        "feature_kind": candidate.feature_kind,
        "feature_family": candidate.feature_family,
        "window_center": float(window.center),
        "window_size": float(window.width),
        "temporal_bins": int(candidate.temporal_bins),
        "train_feature_shape": _shape_list(train_features),
        "test_feature_shape": _shape_list(test_features),
        "feature_dtype": str(train_features.dtype),
        "feature_memory_mib": _memory_mib(train_features, test_features),
        "train_label_shape": _shape_list(train_labels),
        "test_label_shape": _shape_list(test_labels),
        "train_class_counts": _class_count_dict(train_labels),
        "test_class_counts": _class_count_dict(test_labels),
        "n_classes": n_classes,
        "class_names": list(map(str, encoder.classes_)),
        "model_type": type(model).__name__,
        "test_accuracy": None if pd.isna(test_accuracy) else float(test_accuracy),
    }
    profile_path = output_dir / "one_fold_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(json.dumps(profile, indent=2, sort_keys=True, default=str))
    print(f"Wrote one-fold profile: {profile_path}")
    return profile_path


def _run_covariance_method(
    config_path: Path,
    *,
    summary_path: Path,
    inner_path: Path,
    predictions_path: Path,
    max_folds: int | None,
    progress_callback: Callable[..., None] | None = None,
) -> pd.DataFrame:
    from neureptrace.bushmeg_covariance_loso import run_bushmeg_covariance_loso

    return _call_with_supported_kwargs(
        run_bushmeg_covariance_loso,
        config_path=config_path,
        out_path=summary_path,
        inner_cv_out_path=inner_path,
        predictions_out_path=predictions_path,
        max_folds=max_folds,
        progress_callback=progress_callback,
    )


def _run_supervised_lowrank_method(
    config_path: Path,
    *,
    summary_path: Path,
    inner_path: Path,
    predictions_path: Path,
    max_folds: int | None,
    progress_callback: Callable[..., None] | None = None,
) -> pd.DataFrame:
    from neureptrace.bushmeg_supervised_lowrank_loso import run_supervised_lowrank_loso

    return _call_with_supported_kwargs(
        run_supervised_lowrank_loso,
        config_path=config_path,
        out_path=summary_path,
        inner_cv_out_path=inner_path,
        predictions_out_path=predictions_path,
        max_folds=max_folds,
        progress_callback=progress_callback,
    )


def _run_memory_bounded_decode(
    config: Mapping[str, Any],
    *,
    summary_path: Path,
    data_dir: str | Path | None,
    participants: str | Sequence[str] | None,
    max_folds: int | None,
    resume: bool,
    progress_callback: Callable[..., None] | None = None,
) -> pd.DataFrame:
    from neureptrace.bushmeg_loso_decode import run_bushmeg_loso_decode

    dataset = config.get("dataset", {}) if isinstance(config.get("dataset"), Mapping) else {}
    decoding = config.get("decoding", {}) if isinstance(config.get("decoding"), Mapping) else {}
    preprocessing = config.get("preprocessing", {}) if isinstance(config.get("preprocessing"), Mapping) else {}
    data_root = _resolve_data_dir(config, data_dir)
    if data_root is None:
        raise ValueError("BUSH-MEG data directory is required via --data-dir, BUSH_MEG_DATA_DIR, or dataset.root.")
    participant_ids = participants
    if participant_ids is None:
        configured_participants = config.get("participants", {})
        if isinstance(configured_participants, Mapping):
            participant_ids = configured_participants.get("ids")
    return run_bushmeg_loso_decode(
        data_dir=Path(data_root),
        out_path=summary_path,
        participants=participant_ids,
        file_template=str(dataset.get("participant_file", "Part{participant}Data.mat")),
        label_column=str(decoding.get("label_column", "stimulus_class")),
        tmin=preprocessing.get("tmin", -0.35),
        tmax=preprocessing.get("tmax", 0.25),
        window_ms=float(preprocessing.get("window_size", 0.100)) * 1000.0,
        step_ms=float(preprocessing.get("step_size", 0.025)) * 1000.0,
        decode_window=tuple(preprocessing.get("decode_window", (0.134, 0.234))),
        normalization=str(preprocessing.get("normalization", "subject_baseline_whiten")),
        baseline_window=tuple(preprocessing.get("baseline_window", (-0.35, -0.05))),
        max_folds=max_folds,
        resume=resume,
        progress_callback=progress_callback,
    )


def _probability_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if str(column).startswith("prob_class_")]
    return sorted(columns, key=lambda column: int(str(column).rsplit("_", 1)[-1]))


def _top_k_accuracy(probabilities: np.ndarray, labels: np.ndarray, *, k: int) -> float:
    if probabilities.size == 0:
        return float("nan")
    k = min(int(k), probabilities.shape[1])
    top = np.argsort(probabilities, axis=1)[:, -k:]
    return float(np.mean([label in row for label, row in zip(labels, top, strict=True)]))


def _prediction_metric_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    prob_columns = _probability_columns(predictions)
    if predictions.empty or "true_label" not in predictions.columns or not prob_columns:
        return pd.DataFrame()
    subject_column = "outer_test_subject" if "outer_test_subject" in predictions.columns else "heldout_subject"
    rows: list[dict[str, Any]] = []
    for subject, group in predictions.groupby(subject_column, sort=False):
        labels = group["true_label"].astype(int).to_numpy()
        probabilities = group[prob_columns].astype(float).to_numpy()
        predicted = probabilities.argmax(axis=1)
        rows.append(
            {
                "outer_test_subject": str(subject),
                "accuracy": float(accuracy_score(labels, predicted)),
                "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
                "top2_accuracy": _top_k_accuracy(probabilities, labels, k=2),
                "top3_accuracy": _top_k_accuracy(probabilities, labels, k=3),
                "log_loss": float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))),
                "brier": float(brier_score_multiclass(probabilities, labels)),
                "ece": float(expected_calibration_error(probabilities, labels)),
            }
        )
    return pd.DataFrame(rows)


def _as_2d_feature_matrix(features: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(features)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    return matrix


def _as_1d_vector(values: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(values)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    return vector


def _encode_labels_for_classes(labels: Sequence[Any] | np.ndarray, classes: Sequence[Any] | np.ndarray, *, name: str) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=object).reshape(-1)
    classes_array = np.asarray(classes, dtype=object).reshape(-1)
    encoded: list[int] = []
    for value in labels_array:
        matches = np.flatnonzero(classes_array == value)
        if matches.size == 0:
            raise ValueError(f"{name} contains label {value!r}, which is absent from classes.")
        encoded.append(int(matches[0]))
    return np.asarray(encoded, dtype=int)


def _coerce_protocol3_prediction_output(
    output: Any,
    *,
    n_evaluation_rows: int,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities: Any = None
    predicted_labels: Any = None
    if isinstance(output, Mapping):
        probabilities = output.get("probabilities")
        predicted_labels = output.get("predicted_labels")
    elif isinstance(output, tuple) and len(output) == 2:
        probabilities, predicted_labels = output
    else:
        probabilities = output

    n_classes = int(classes.size)
    if probabilities is not None:
        probability_matrix = np.asarray(probabilities, dtype=float)
        if probability_matrix.shape != (n_evaluation_rows, n_classes):
            raise ValueError(
                "Protocol 3 fit/predict returned probabilities with shape "
                f"{probability_matrix.shape}; expected {(n_evaluation_rows, n_classes)}."
            )
        row_sums = probability_matrix.sum(axis=1, keepdims=True)
        if np.any(~np.isfinite(probability_matrix)) or np.any(row_sums <= 0):
            raise ValueError("Protocol 3 fit/predict returned invalid probability rows.")
        probability_matrix = probability_matrix / row_sums
    elif predicted_labels is not None:
        predicted_indices = _encode_or_index_predictions(predicted_labels, classes, n_evaluation_rows=n_evaluation_rows)
        probability_matrix = np.zeros((n_evaluation_rows, n_classes), dtype=float)
        probability_matrix[np.arange(n_evaluation_rows), predicted_indices] = 1.0
    else:
        raise ValueError("Protocol 3 fit/predict must return probabilities or predicted_labels.")

    if predicted_labels is None:
        predicted_indices = probability_matrix.argmax(axis=1).astype(int, copy=False)
    else:
        predicted_indices = _encode_or_index_predictions(predicted_labels, classes, n_evaluation_rows=n_evaluation_rows)
    predicted_values = classes[predicted_indices]
    return probability_matrix, predicted_indices, predicted_values


def _encode_or_index_predictions(
    predicted_labels: Sequence[Any] | np.ndarray,
    classes: np.ndarray,
    *,
    n_evaluation_rows: int,
) -> np.ndarray:
    predicted = np.asarray(predicted_labels)
    if predicted.ndim != 1 or predicted.size != n_evaluation_rows:
        raise ValueError(
            "Protocol 3 fit/predict returned predicted_labels with shape "
            f"{predicted.shape}; expected {(n_evaluation_rows,)}."
        )
    try:
        return _encode_labels_for_classes(predicted, classes, name="predicted_labels")
    except ValueError:
        if np.issubdtype(predicted.dtype, np.integer):
            indices = predicted.astype(int, copy=False)
            if np.all((0 <= indices) & (indices < classes.size)):
                return indices
        raise


def _safe_log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    return float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1])))


def run_bushmeg_protocol3_fold_adapter(
    *,
    source_features: Sequence[Sequence[float]] | np.ndarray,
    source_labels: Sequence[Any] | np.ndarray,
    source_subject_ids: Sequence[Any] | np.ndarray,
    target_features: Sequence[Sequence[float]] | np.ndarray,
    target_labels: Sequence[Any] | np.ndarray,
    classes: Sequence[Any] | np.ndarray,
    method_spec: MethodSpec,
    k_per_class: int,
    method_dir: str | Path,
    fit_predict: Callable[..., Any],
    outer_test_subject: str | int = "target",
    fold_index: int = 1,
    n_folds: int | None = None,
    seed: int = 13,
    min_evaluation_per_class: int = 1,
    summary_metadata: Mapping[str, Any] | None = None,
) -> Protocol3FoldAdapterResult:
    """Run one Protocol 3 fold with a shared target calibration/evaluation split.

    ``fit_predict`` receives source data, target calibration features/labels,
    target evaluation features, classes, and method metadata. It deliberately
    does not receive target evaluation labels.
    """

    if int(method_spec.protocol_category) != 3:
        raise ValueError("run_bushmeg_protocol3_fold_adapter requires a Protocol 3 MethodSpec.")

    source_matrix = _as_2d_feature_matrix(source_features, name="source_features")
    target_matrix = _as_2d_feature_matrix(target_features, name="target_features")
    if source_matrix.shape[1] != target_matrix.shape[1]:
        raise ValueError(
            "source_features and target_features must have the same feature width: "
            f"{source_matrix.shape[1]} != {target_matrix.shape[1]}."
        )
    source_label_vector = _as_1d_vector(source_labels, name="source_labels")
    target_label_vector = _as_1d_vector(target_labels, name="target_labels")
    source_subject_vector = _as_1d_vector(source_subject_ids, name="source_subject_ids")
    class_values = _as_1d_vector(classes, name="classes").astype(object, copy=False)
    if source_matrix.shape[0] != source_label_vector.size:
        raise ValueError("source_features and source_labels must have the same row count.")
    if source_matrix.shape[0] != source_subject_vector.size:
        raise ValueError("source_features and source_subject_ids must have the same row count.")
    if target_matrix.shape[0] != target_label_vector.size:
        raise ValueError("target_features and target_labels must have the same row count.")
    if class_values.size == 0:
        raise ValueError("classes must not be empty.")

    method_dir = Path(method_dir)
    progress = MethodProgress(method_dir, method=method_spec.method)
    progress.initialize_artifacts()
    progress.update(
        "configured",
        method_family=method_spec.method_family,
        protocol_category=3,
        runner=method_spec.runner,
        outer_test_subject=str(outer_test_subject),
        fold_index=int(fold_index),
        n_folds=n_folds,
        k_per_class=int(k_per_class),
    )
    progress.update("fold_start", outer_test_subject=str(outer_test_subject), fold_index=int(fold_index), n_folds=n_folds)

    split = select_bushmeg_target_calibration_split(
        target_label_vector,
        per_class=k_per_class,
        seed=seed,
        min_evaluation_per_class=min_evaluation_per_class,
        context=(outer_test_subject, method_spec.method, k_per_class),
    )
    common_row: dict[str, Any] = {
        "method": method_spec.method,
        "method_family": method_spec.method_family,
        **method_spec.protocol.metadata(),
        **split.metadata(),
        **dict(summary_metadata or {}),
        "outer_test_subject": str(outer_test_subject),
        "fold_index": int(fold_index),
        "n_folds": n_folds,
        "k_per_class": int(k_per_class),
        "n_source_subjects": int(np.unique(source_subject_vector).size),
        "n_source_trials": int(source_label_vector.size),
        "n_train_subjects": int(np.unique(source_subject_vector).size),
        "n_train": int(source_label_vector.size),
        "n_target_trials": int(target_label_vector.size),
        "n_test_trials": int(split.n_target_evaluation_trials),
        "n_calibration_trials": int(split.n_target_calibration_trials),
        "n_classes": int(class_values.size),
        "class_names": "|".join(map(str, class_values.tolist())),
    }

    if split.skipped:
        summary = pd.DataFrame(
            [
                {
                    **common_row,
                    "balanced_accuracy": np.nan,
                    "accuracy": np.nan,
                    "top2_accuracy": np.nan,
                    "top3_accuracy": np.nan,
                    "log_loss": np.nan,
                    "brier": np.nan,
                    "ece": np.nan,
                    "skip_reason": split.skip_reason,
                    "skip_reason_code": split.skip_reason_code,
                }
            ]
        )
        predictions = pd.DataFrame()
        _append_csv_rows(summary, progress.summary_partial_path)
        _append_csv_rows(predictions, progress.predictions_partial_path)
        progress.update(
            "fold_done",
            outer_test_subject=str(outer_test_subject),
            fold_index=int(fold_index),
            skipped=True,
            skip_reason=split.skip_reason,
            skip_reason_code=split.skip_reason_code,
        )
        return Protocol3FoldAdapterResult(summary=summary, predictions=predictions, split=split, skipped=True, skip_reason=split.skip_reason)

    validate_protocol_input_use(
        3,
        target_features_for_fitting=True,
        target_labels_for_fitting=True,
        calibration_indices=split.calibration_indices,
        evaluation_indices=split.evaluation_indices,
    )
    target_calibration_features = target_matrix[split.calibration_indices]
    target_calibration_labels = target_label_vector[split.calibration_indices]
    target_evaluation_features = target_matrix[split.evaluation_indices]
    target_evaluation_labels = target_label_vector[split.evaluation_indices]
    evaluation_label_indices = _encode_labels_for_classes(target_evaluation_labels, class_values, name="target evaluation labels")

    progress.update(
        "fit_start",
        outer_test_subject=str(outer_test_subject),
        fold_index=int(fold_index),
        n_calibration_trials=int(target_calibration_labels.size),
        n_evaluation_trials=int(target_evaluation_labels.size),
    )
    method_output = _call_with_supported_kwargs(
        fit_predict,
        source_features=source_matrix,
        source_labels=source_label_vector,
        source_subject_ids=source_subject_vector,
        target_calibration_features=target_calibration_features,
        target_calibration_labels=target_calibration_labels,
        target_evaluation_features=target_evaluation_features,
        classes=class_values,
        method_spec=method_spec,
        k_per_class=int(k_per_class),
        calibration_indices=split.calibration_indices.copy(),
        evaluation_indices=split.evaluation_indices.copy(),
    )
    progress.update(
        "predict_start",
        outer_test_subject=str(outer_test_subject),
        fold_index=int(fold_index),
        n_evaluation_trials=int(target_evaluation_labels.size),
    )
    method_metadata = dict(method_output.get("metadata", {})) if isinstance(method_output, Mapping) else {}
    probabilities, predicted_indices, predicted_values = _coerce_protocol3_prediction_output(
        method_output,
        n_evaluation_rows=int(target_evaluation_labels.size),
        classes=class_values,
    )

    summary = pd.DataFrame(
        [
            {
                **common_row,
                **method_metadata,
                "balanced_accuracy": float(balanced_accuracy_score(evaluation_label_indices, predicted_indices)),
                "accuracy": float(accuracy_score(evaluation_label_indices, predicted_indices)),
                "top2_accuracy": _top_k_accuracy(probabilities, evaluation_label_indices, k=2),
                "top3_accuracy": _top_k_accuracy(probabilities, evaluation_label_indices, k=3),
                "log_loss": _safe_log_loss(evaluation_label_indices, probabilities),
                "brier": float(brier_score_multiclass(probabilities, evaluation_label_indices)),
                "ece": float(expected_calibration_error(probabilities, evaluation_label_indices)),
            }
        ]
    )
    prediction_rows: list[dict[str, Any]] = []
    for row_index, true_label, true_index, predicted_label, predicted_index, probability_row in zip(
        split.evaluation_indices,
        target_evaluation_labels,
        evaluation_label_indices,
        predicted_values,
        predicted_indices,
        probabilities,
        strict=True,
    ):
        prediction_row: dict[str, Any] = {
            "method": method_spec.method,
            "method_family": method_spec.method_family,
            **method_spec.protocol.metadata(),
            "outer_test_subject": str(outer_test_subject),
            "fold_index": int(fold_index),
            "trial_index": int(row_index),
            "target_row_index": int(row_index),
            "is_calibration_row": False,
            "k_per_class": int(k_per_class),
            "target_calibration_per_class": int(split.per_class),
            "n_target_calibration_trials": int(split.n_target_calibration_trials),
            "n_target_evaluation_trials": int(split.n_target_evaluation_trials),
            "true_label": _readable_label_value(true_label),
            "true_label_index": int(true_index),
            "predicted_label": _readable_label_value(predicted_label),
            "predicted_label_index": int(predicted_index),
        }
        for class_index, probability in enumerate(probability_row):
            prediction_row[f"prob_class_{class_index}"] = float(probability)
        prediction_rows.append(prediction_row)
    predictions = pd.DataFrame(prediction_rows)

    _append_csv_rows(summary, progress.summary_partial_path)
    _append_csv_rows(predictions, progress.predictions_partial_path)
    progress.update(
        "fold_done",
        outer_test_subject=str(outer_test_subject),
        fold_index=int(fold_index),
        skipped=False,
        n_summary_rows=int(len(summary)),
        n_prediction_rows=int(len(predictions)),
        k_per_class=int(k_per_class),
        n_target_calibration_trials=int(split.n_target_calibration_trials),
        n_target_evaluation_trials=int(split.n_target_evaluation_trials),
    )
    return Protocol3FoldAdapterResult(summary=summary, predictions=predictions, split=split)


def _few_shot_target_calibrated_fit_predict(
    *,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_calibration_features: np.ndarray,
    target_calibration_labels: np.ndarray,
    target_evaluation_features: np.ndarray,
    classes: np.ndarray,
    method_spec: MethodSpec,
    k_per_class: int,
    candidate: Any,
    decoding: Mapping[str, Any],
    protocol3: Mapping[str, Any],
) -> Mapping[str, np.ndarray]:
    from neureptrace.decoding.few_shot import (
        FewShotTargetCalibrationSplit,
        fit_few_shot_target_calibrated_decoder,
    )

    calibration_features = _as_2d_feature_matrix(target_calibration_features, name="target_calibration_features")
    evaluation_features = _as_2d_feature_matrix(target_evaluation_features, name="target_evaluation_features")
    calibration_labels = _as_1d_vector(target_calibration_labels, name="target_calibration_labels")
    if calibration_features.shape[0] != calibration_labels.size:
        raise ValueError("target_calibration_features and target_calibration_labels must have the same row count.")
    if calibration_labels.size == 0:
        raise ValueError("few-shot target-calibrated decoder requires at least one calibration label.")

    target_features = np.vstack([calibration_features, evaluation_features])
    source_label_tokens = np.asarray([str(value) for value in np.asarray(source_labels).reshape(-1)], dtype=str)
    calibration_label_tokens = np.asarray([str(value) for value in calibration_labels.reshape(-1)], dtype=str)
    class_tokens = np.asarray([str(value) for value in np.asarray(classes).reshape(-1)], dtype=str)
    # Dummy evaluation labels satisfy the helper's shape checks without leaking
    # true evaluation labels into method-specific fitting/adaptation code.
    dummy_evaluation_labels = np.full(evaluation_features.shape[0], calibration_label_tokens[0], dtype=str)
    target_labels_for_helper = np.concatenate([calibration_label_tokens, dummy_evaluation_labels])
    split = FewShotTargetCalibrationSplit(
        calibration_indices=np.arange(calibration_features.shape[0], dtype=int),
        evaluation_indices=np.arange(calibration_features.shape[0], target_features.shape[0], dtype=int),
    )
    result = fit_few_shot_target_calibrated_decoder(
        source_features=source_features,
        source_labels=source_label_tokens,
        target_features=target_features,
        target_labels=target_labels_for_helper,
        classes=class_tokens,
        split=split,
        per_class=int(k_per_class),
        seed=int(protocol3.get("target_calibration_seed", 13)),
        min_evaluation_per_class=int(protocol3.get("min_evaluation_per_class", 1)),
        target_repeats=int(protocol3.get("target_repeats", 1)),
        decoder_name=getattr(candidate, "decoder", decoding.get("classifier", "logistic")),
        emission_mode=getattr(candidate, "emission_mode", decoding.get("emission_mode", "uncalibrated")),
        max_iter=int(decoding.get("max_iter", 1000)),
        feature_preprocessor=getattr(candidate, "feature_preprocessor", "none"),
        pca_components=getattr(candidate, "pca_components", None),
        tune_hyperparameters=False,
        decoder_kwargs={"classifier_param": getattr(candidate, "classifier_param", None)},
    )
    return {"probabilities": result.probabilities, "metadata": result.metadata}


def _fit_decoder_probability_output(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    test_features: np.ndarray,
    classes: np.ndarray,
    candidate: Any,
    decoding: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    from neureptrace.decoding import make_decoder, predict_emission_probabilities

    model = make_decoder(
        getattr(candidate, "decoder", decoding.get("classifier", "logistic")),
        max_iter=int(decoding.get("max_iter", 1000)),
        emission_mode=getattr(candidate, "emission_mode", decoding.get("emission_mode", "uncalibrated")),
        feature_preprocessor=getattr(candidate, "feature_preprocessor", "none"),
        pca_components=getattr(candidate, "pca_components", None),
        classifier_param=getattr(candidate, "classifier_param", None),
    )
    model.fit(train_features, train_labels)
    raw = predict_emission_probabilities(model, test_features, emission_mode=getattr(candidate, "emission_mode", decoding.get("emission_mode", "uncalibrated")))
    model_classes = np.asarray(getattr(model, "classes_", classes), dtype=object).reshape(-1)
    class_values = np.asarray(classes, dtype=object).reshape(-1)
    aligned = np.zeros((raw.shape[0], class_values.size), dtype=float)
    for source_column, class_label in enumerate(model_classes.tolist()):
        matches = np.flatnonzero(class_values == class_label)
        if matches.size:
            aligned[:, int(matches[0])] = raw[:, source_column]
    row_sums = aligned.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("Classifier emitted no probability mass for at least one Protocol 3 evaluation row.")
    return {"probabilities": aligned / row_sums, "metadata": dict(metadata or {})}


def _source_plus_target_calibrated_fit_predict(
    *,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_calibration_features: np.ndarray,
    target_calibration_labels: np.ndarray,
    target_evaluation_features: np.ndarray,
    classes: np.ndarray,
    method_spec: MethodSpec,
    k_per_class: int,
    candidate: Any,
    decoding: Mapping[str, Any],
    protocol3: Mapping[str, Any],
) -> Mapping[str, Any]:
    fit_features = np.vstack([source_features, target_calibration_features])
    fit_labels = np.concatenate([np.asarray(source_labels).reshape(-1), np.asarray(target_calibration_labels).reshape(-1)])
    return _fit_decoder_probability_output(
        train_features=fit_features,
        train_labels=fit_labels,
        test_features=target_evaluation_features,
        classes=classes,
        candidate=candidate,
        decoding=decoding,
        metadata={
            "source_plus_target_calibration": True,
            "source_plus_target_classifier_training": True,
            "source_plus_target_decoder": getattr(candidate, "decoder", decoding.get("classifier", "logistic")),
            "target_calibration_per_class": int(k_per_class),
        },
    )


def _target_calibrated_alignment_fit_predict(
    *,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    source_subject_ids: np.ndarray,
    target_calibration_features: np.ndarray,
    target_calibration_labels: np.ndarray,
    target_evaluation_features: np.ndarray,
    classes: np.ndarray,
    method_spec: MethodSpec,
    k_per_class: int,
    candidate: Any,
    decoding: Mapping[str, Any],
    protocol3: Mapping[str, Any],
) -> Mapping[str, Any]:
    from neureptrace.decoding.source_alignment import align_train_test_features, source_alignment_config

    alignment_method = str(protocol3.get("alignment_method", "procrustes"))
    config = source_alignment_config(
        method=alignment_method,
        anchor_mode="class_mean",
        target_projection="target_calibrated_alignment",
        target_calibration_per_anchor=int(k_per_class),
        target_calibration_seed=int(protocol3.get("target_calibration_seed", 13)),
    )
    aligned = align_train_test_features(
        train_features=source_features,
        train_labels=source_labels,
        train_subject_ids=source_subject_ids,
        test_features=target_evaluation_features,
        config=config,
        target_calibration_features=target_calibration_features,
        target_calibration_labels=target_calibration_labels,
    )
    return _fit_decoder_probability_output(
        train_features=aligned.train_features,
        train_labels=source_labels,
        test_features=aligned.test_features,
        classes=classes,
        candidate=candidate,
        decoding=decoding,
        metadata={**aligned.metadata, "target_calibrated_alignment_method": alignment_method},
    )


def _target_calibrated_gaussian_fit_predict(
    *,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    target_calibration_features: np.ndarray,
    target_calibration_labels: np.ndarray,
    target_evaluation_features: np.ndarray,
    classes: np.ndarray,
    method_spec: MethodSpec,
    k_per_class: int,
    candidate: Any,
    decoding: Mapping[str, Any],
    protocol3: Mapping[str, Any],
) -> Mapping[str, Any]:
    from neureptrace.decoding.generative_augmentation import augment_training_features, generative_augmentation_config

    config = generative_augmentation_config(
        method="target_calibrated_gaussian",
        synthetic_per_class=int(protocol3.get("synthetic_per_class", 8)),
        noise_scale=protocol3.get("noise_scale", 1.0),
        covariance_shrinkage=protocol3.get("covariance_shrinkage", 1.0),
        target_calibration_weight=protocol3.get("target_calibration_weight", 0.5),
        random_state=int(protocol3.get("target_calibration_seed", 13)),
    )
    augmented = augment_training_features(
        source_features,
        source_labels,
        config=config,
        target_calibration_features=target_calibration_features,
        target_calibration_labels=target_calibration_labels,
    )
    metadata = {
        **augmented.metadata,
        "augmentation_method": "target_calibrated_gaussian",
        "target_calibrated_gaussian": True,
        "synthetic_rows_marked": bool(np.any(augmented.synthetic_mask)),
        "n_synthetic_rows": int(np.sum(augmented.synthetic_mask)),
    }
    return _fit_decoder_probability_output(
        train_features=augmented.features,
        train_labels=augmented.labels,
        test_features=target_evaluation_features,
        classes=classes,
        candidate=candidate,
        decoding=decoding,
        metadata=metadata,
    )


def _semi_supervised_lora_few_shot_fit_predict(
    *,
    source_features: np.ndarray,
    source_labels: np.ndarray,
    source_subject_ids: np.ndarray,
    target_calibration_features: np.ndarray,
    target_calibration_labels: np.ndarray,
    target_evaluation_features: np.ndarray,
    classes: np.ndarray,
    method_spec: MethodSpec,
    k_per_class: int,
    candidate: Any,
    decoding: Mapping[str, Any],
    protocol3: Mapping[str, Any],
) -> Mapping[str, Any]:
    from neureptrace.decoding.few_shot import FewShotTargetCalibrationSplit
    from neureptrace.decoding.semi_supervised_lora_few_shot import fit_semi_supervised_lora_few_shot_decoder

    calibration_labels = np.asarray(target_calibration_labels).reshape(-1)
    dummy_eval = np.full(target_evaluation_features.shape[0], calibration_labels[0], dtype=object)
    target_features = np.vstack([target_calibration_features, target_evaluation_features])
    target_labels_for_helper = np.concatenate([calibration_labels, dummy_eval])
    split = FewShotTargetCalibrationSplit(
        calibration_indices=np.arange(target_calibration_features.shape[0], dtype=int),
        evaluation_indices=np.arange(target_calibration_features.shape[0], target_features.shape[0], dtype=int),
    )
    result = fit_semi_supervised_lora_few_shot_decoder(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        target_labels=target_labels_for_helper,
        source_groups=source_subject_ids,
        classes=classes,
        split=split,
        per_class=int(k_per_class),
        seed=int(protocol3.get("target_calibration_seed", 13)),
        min_evaluation_per_class=int(protocol3.get("min_evaluation_per_class", 1)),
        use_evaluation_features_unlabeled=True,
        source_pretrain_epochs=int(protocol3.get("source_pretrain_epochs", protocol3.get("max_epochs", 5))),
        target_adaptation_steps=int(protocol3.get("target_adaptation_steps", 5)),
        hidden_dim=int(protocol3.get("hidden_dim", 32)),
        lora_rank=int(protocol3.get("lora_rank", 4)),
    )
    return {"probabilities": result.probabilities, "metadata": result.metadata}


def _protocol3_subject_cache_key(config: Mapping[str, Any], *, config_dir: Path) -> str:
    payload: dict[str, Any] = {}
    for section_name in ("dataset", "metadata", "matlab", "preprocessing", "participants"):
        value = config.get(section_name)
        if value is not None:
            payload[section_name] = value
    decoding = config.get("decoding") if isinstance(config.get("decoding"), Mapping) else {}
    source_loso = config.get("source_loso") if isinstance(config.get("source_loso"), Mapping) else {}
    payload["label_column"] = decoding.get("label_column", "stimulus_class")
    payload["group_column"] = source_loso.get("group_column", decoding.get("group_column", "participant"))
    dataset = payload.get("dataset")
    if isinstance(dataset, Mapping):
        root = dataset.get("root")
        if root is not None:
            root_path = Path(str(root))
            if not root_path.is_absolute():
                dataset = dict(dataset)
                dataset["root"] = str((config_dir / root_path).resolve())
                payload["dataset"] = dataset
    return json.dumps(payload, sort_keys=True, default=str)


def _load_protocol3_subjects_cached(
    config: Mapping[str, Any],
    *,
    config_dir: Path,
    progress_callback: Callable[..., None] | None,
) -> tuple[Any, Any]:
    from neureptrace.bushmeg_source_loso import _load_subjects_from_config

    key = _protocol3_subject_cache_key(config, config_dir=config_dir)
    if key in _PROTOCOL3_SUBJECT_CACHE:
        subjects, encoder = _PROTOCOL3_SUBJECT_CACHE[key]
        if progress_callback is not None:
            progress_callback("loading_subjects", cache_hit=True, n_subject_files=len(subjects))
        return subjects, encoder
    subjects, encoder = _load_subjects_from_config(config, config_dir=config_dir, progress_callback=progress_callback)
    _PROTOCOL3_SUBJECT_CACHE[key] = (subjects, encoder)
    return subjects, encoder


def _candidate_window_concatenated_features(
    *,
    subjects: Mapping[str, Any],
    cache: Any,
    candidate: Any,
    train_subjects: Sequence[str],
    test_subject: str,
    n_classes: int,
    train_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not getattr(candidate, "windows", ()):
        raise ValueError(f"Candidate {getattr(candidate, 'name', '<unknown>')!r} has no windows.")
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    from neureptrace.bushmeg_source_loso import _prepare_window_train_test_features

    for window in candidate.windows:
        train_features, test_features = _prepare_window_train_test_features(
            subjects=subjects,
            cache=cache,
            candidate=candidate,
            train_subjects=train_subjects,
            test_subject=test_subject,
            window=window,
            n_classes=n_classes,
            train_labels=train_labels,
        )
        train_parts.append(np.asarray(train_features))
        test_parts.append(np.asarray(test_features))
    if len(train_parts) == 1:
        return train_parts[0], test_parts[0]
    return np.hstack(train_parts), np.hstack(test_parts)


def _run_protocol3_few_shot_method(
    config_path: Path,
    *,
    summary_path: Path,
    inner_path: Path,
    predictions_path: Path,
    max_folds: int | None,
    progress_callback: Callable[..., None] | None = None,
    method_spec: MethodSpec,
) -> pd.DataFrame:
    from neureptrace.bushmeg_source_loso import (
        FeatureCache,
        _candidate_grid,
        _candidate_rowspec,
        _stack_subject_ids,
        _stack_subject_labels,
    )

    config_path = Path(config_path)
    config = load_config(config_path)
    decoding = _config_section(config, "decoding") or _config_section(config, "workflow")
    protocol3 = _config_section(config, "protocol3")
    k_per_class = int(protocol3.get("target_calibration_per_class", 1))
    seed = int(protocol3.get("target_calibration_seed", 13))
    min_evaluation_per_class = int(protocol3.get("min_evaluation_per_class", 1))

    subjects, encoder = _load_protocol3_subjects_cached(config, config_dir=config_path.parent, progress_callback=progress_callback)
    candidates = _candidate_grid(config)
    if not candidates:
        raise ValueError("No source_loso candidates are available for Protocol 3 few-shot evaluation.")
    candidate = candidates[0]
    cache = FeatureCache(subjects)
    n_classes = int(len(encoder.classes_))
    class_values = np.arange(n_classes, dtype=int)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    inner_path.parent.mkdir(parents=True, exist_ok=True)
    if not inner_path.exists():
        inner_path.write_text("", encoding="utf-8")

    outer_subjects = sorted(subjects)
    if max_folds is not None:
        outer_subjects = outer_subjects[: max(0, int(max_folds))]
    method_dir = summary_path.parent

    for fold_index, outer_test_subject in enumerate(outer_subjects, start=1):
        train_subjects = [subject for subject in sorted(subjects) if subject != outer_test_subject]
        if not train_subjects:
            raise ValueError("Protocol 3 few-shot evaluation requires at least one source subject.")
        if progress_callback is not None:
            progress_callback(
                "feature_start",
                outer_test_subject=outer_test_subject,
                fold_index=fold_index,
                n_folds=len(outer_subjects),
                selected_candidate=getattr(candidate, "name", ""),
            )
        train_labels = _stack_subject_labels(subjects, train_subjects)
        source_features, target_features = _candidate_window_concatenated_features(
            subjects=subjects,
            cache=cache,
            candidate=candidate,
            train_subjects=train_subjects,
            test_subject=outer_test_subject,
            n_classes=n_classes,
            train_labels=train_labels,
        )
        source_subject_ids = _stack_subject_ids(subjects, train_subjects)

        fit_predict_impl: Callable[..., Mapping[str, Any]]
        if method_spec.runner == "protocol3_source_plus_target":
            fit_predict_impl = _source_plus_target_calibrated_fit_predict
        elif method_spec.runner == "protocol3_target_calibrated_alignment":
            fit_predict_impl = _target_calibrated_alignment_fit_predict
        elif method_spec.runner == "protocol3_target_calibrated_gaussian":
            fit_predict_impl = _target_calibrated_gaussian_fit_predict
        elif method_spec.runner == "protocol3_lora_few_shot":
            fit_predict_impl = _semi_supervised_lora_few_shot_fit_predict
        else:
            fit_predict_impl = _few_shot_target_calibrated_fit_predict

        def fit_predict(**kwargs: Any) -> Mapping[str, Any]:
            return _call_with_supported_kwargs(
                fit_predict_impl,
                **kwargs,
                candidate=candidate,
                decoding=decoding,
                protocol3=protocol3,
            )

        result = run_bushmeg_protocol3_fold_adapter(
            source_features=source_features,
            source_labels=train_labels,
            source_subject_ids=source_subject_ids,
            target_features=target_features,
            target_labels=subjects[outer_test_subject].labels,
            classes=class_values,
            method_spec=method_spec,
            k_per_class=k_per_class,
            method_dir=method_dir,
            fit_predict=fit_predict,
            outer_test_subject=outer_test_subject,
            fold_index=fold_index,
            n_folds=len(outer_subjects),
            seed=seed,
            min_evaluation_per_class=min_evaluation_per_class,
            summary_metadata=_candidate_rowspec(candidate),
        )

    return _read_csv_if_nonempty(summary_path)


def _first_existing(row: pd.Series, names: Sequence[str], default: Any = pd.NA) -> Any:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def _window_size_from_row(row: pd.Series, config: Mapping[str, Any]) -> Any:
    if "window_widths" in row.index and pd.notna(row["window_widths"]):
        return row["window_widths"]
    if "window_size" in row.index and pd.notna(row["window_size"]):
        return row["window_size"]
    if "window_start" in row.index and "window_stop" in row.index and pd.notna(row["window_start"]) and pd.notna(row["window_stop"]):
        return float(row["window_stop"]) - float(row["window_start"])
    preprocessing = config.get("preprocessing", {}) if isinstance(config.get("preprocessing"), Mapping) else {}
    return preprocessing.get("window_size", pd.NA)


def _normalize_summary(
    raw_summary: pd.DataFrame,
    raw_predictions: pd.DataFrame,
    *,
    spec: MethodSpec,
    config: Mapping[str, Any],
) -> pd.DataFrame:
    if raw_summary.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    summary = raw_summary.copy()
    if "analysis" in summary.columns and (summary["analysis"] == "temporal_ensemble").any():
        summary = summary.loc[summary["analysis"] == "temporal_ensemble"].copy()
    prediction_metrics = _prediction_metric_frame(raw_predictions)
    if not prediction_metrics.empty:
        subject_column = "outer_test_subject" if "outer_test_subject" in summary.columns else "heldout_subject"
        summary[subject_column] = summary[subject_column].astype(str)
        prediction_metrics["outer_test_subject"] = prediction_metrics["outer_test_subject"].astype(str)
        summary = summary.merge(prediction_metrics, how="left", left_on=subject_column, right_on="outer_test_subject", suffixes=("", "_from_predictions"))
        if "outer_test_subject_y" in summary.columns:
            summary = summary.drop(columns=["outer_test_subject_y"])
        if "outer_test_subject_x" in summary.columns:
            summary = summary.rename(columns={"outer_test_subject_x": "outer_test_subject"})
        for metric in ("accuracy", "balanced_accuracy", "top2_accuracy", "top3_accuracy", "log_loss", "brier", "ece"):
            fallback = f"{metric}_from_predictions"
            if fallback in summary.columns:
                if metric not in summary.columns:
                    summary[metric] = summary[fallback]
                else:
                    summary[metric] = summary[metric].where(pd.notna(summary[metric]), summary[fallback])
                summary = summary.drop(columns=[fallback])
    metadata = spec.protocol.metadata()
    rows: list[dict[str, Any]] = []
    for _, row in summary.iterrows():
        n_target_trials = _first_existing(row, ("n_test_trials", "n_test", "n_target_trials"))
        protocol_category = int(spec.protocol_category)
        n_calibration_trials = (
            n_target_trials
            if protocol_category == 4 and pd.notna(n_target_trials)
            else _first_existing(row, ("n_calibration_trials", "n_target_calibration_trials"), 0)
        )
        target_calibration_per_class = _first_existing(row, ("target_calibration_per_class", "few_shot_target_calibration_per_class"))
        k_per_class = _first_existing(row, ("k_per_class", "target_calibration_per_class", "few_shot_target_calibration_per_class"))
        n_target_calibration_trials = _first_existing(row, ("n_target_calibration_trials", "n_calibration_trials"), n_calibration_trials)
        n_target_evaluation_trials = _first_existing(row, ("n_target_evaluation_trials", "n_evaluation_trials"))
        target_calibration_seed = _first_existing(row, ("target_calibration_seed", "few_shot_target_calibration_seed"))
        normalized = {
            "method": spec.method,
            "method_family": spec.method_family,
            **metadata,
            "calibration_rows_disjoint_from_evaluation": _first_existing(
                row,
                ("calibration_rows_disjoint_from_evaluation",),
                metadata["calibration_rows_disjoint_from_evaluation"],
            ),
            "outer_test_subject": str(_first_existing(row, ("outer_test_subject", "heldout_subject"))),
            "n_source_subjects": _first_existing(row, ("n_train_subjects", "n_source_subjects")),
            "n_source_trials": _first_existing(row, ("n_train", "n_source_trials")),
            "n_target_trials": n_target_trials,
            "n_calibration_trials": n_calibration_trials,
            "target_calibration_per_class": target_calibration_per_class,
            "k_per_class": k_per_class,
            "n_target_calibration_trials": n_target_calibration_trials,
            "n_target_evaluation_trials": n_target_evaluation_trials,
            "target_calibration_seed": target_calibration_seed,
            "feature_kind": _first_existing(
                row,
                ("feature_kind", "covariance_feature_mode", "window_feature_mode", "feature_preprocessor"),
                spec.method_family,
            ),
            "window_centers": _first_existing(row, ("window_centers", "time")),
            "window_size": _window_size_from_row(row, config),
            "temporal_bins": _first_existing(row, ("temporal_bins",), pd.NA),
            "balanced_accuracy": _first_existing(row, ("balanced_accuracy",)),
            "accuracy": _first_existing(row, ("accuracy",)),
            "top2_accuracy": _first_existing(row, ("top2_accuracy",)),
            "top3_accuracy": _first_existing(row, ("top3_accuracy",)),
            "log_loss": _first_existing(row, ("log_loss",)),
            "brier": _first_existing(row, ("brier",)),
            "ece": _first_existing(row, ("ece",)),
        }
        for column, value in row.items():
            normalized.setdefault(str(column), value)
        rows.append(normalized)
    frame = pd.DataFrame(rows)
    extra_columns = [column for column in frame.columns if column not in SUMMARY_COLUMNS]
    return frame[SUMMARY_COLUMNS + extra_columns]


def _normalize_predictions(raw_predictions: pd.DataFrame, *, spec: MethodSpec) -> pd.DataFrame:
    if raw_predictions.empty:
        columns = [
            "method",
            "method_family",
            "protocol_category",
            "protocol_name",
            "outer_test_subject",
            "trial_index",
            "true_label",
            "predicted_label",
        ]
        return pd.DataFrame(columns=columns)
    predictions = raw_predictions.copy()
    metadata = {"method": spec.method, "method_family": spec.method_family, **spec.protocol.metadata()}
    for key, value in metadata.items():
        predictions[key] = value
    leading = [key for key in metadata if key in predictions.columns]
    trailing = [column for column in predictions.columns if column not in leading]
    return predictions[leading + trailing]


def _selected_methods(
    *,
    all_protocols: Mapping[str, Any],
    methods: str | Sequence[str] | None,
    protocols: str | Sequence[str | int] | None,
    include_oracle: bool,
    non_oracle: bool = False,
) -> list[MethodSpec]:
    registry = method_registry()
    configured = _split_csv(methods) or _split_csv(all_protocols.get("methods"))
    if not configured:
        configured = [
            method
            for method, spec in registry.items()
            if spec.runnable and spec.protocol_category in {1, 2}
        ]
    groups = all_protocols.get("method_groups", {}) or {}
    if not isinstance(groups, Mapping):
        raise ValueError("all_protocols.method_groups must be a mapping.")
    expanded: list[str] = []
    for token in configured:
        if token.lower() == "all":
            expanded.extend(registry)
        elif token in groups:
            expanded.extend(_split_csv(groups[token]))
        else:
            expanded.append(token)
    configured = list(dict.fromkeys(expanded))
    unknown = sorted(set(configured).difference(registry))
    if unknown:
        raise ValueError(f"Unknown BUSH-MEG all-protocol method(s): {', '.join(unknown)}.")
    requested_protocols = _parse_protocols(protocols)
    selected: list[MethodSpec] = []
    for method in configured:
        spec = registry[method]
        if spec.protocol_category not in requested_protocols:
            continue
        if non_oracle and spec.protocol_category == 4:
            continue
        if spec.protocol_category == 4 and not include_oracle:
            raise ValueError(f"Method {method!r} is Protocol 4 oracle/debug and requires --include-oracle.")
        validate_target_label_policy(
            spec.protocol,
            uses_target_labels_for_fitting=spec.protocol.uses_target_labels_for_fitting,
            include_oracle=include_oracle,
        )
        selected.append(spec)
    return selected


def _configured_method_names(
    *,
    all_protocols: Mapping[str, Any],
    methods: str | Sequence[str] | None,
    protocols: str | Sequence[str | int] | None,
    include_oracle: bool,
    non_oracle: bool = False,
) -> set[str]:
    registry = method_registry()
    configured = _split_csv(methods) or _split_csv(all_protocols.get("methods"))
    if not configured:
        configured = [
            method
            for method, spec in registry.items()
            if spec.runnable and spec.protocol_category in {1, 2}
        ]
    groups = all_protocols.get("method_groups", {}) or {}
    if not isinstance(groups, Mapping):
        raise ValueError("all_protocols.method_groups must be a mapping.")
    expanded: list[str] = []
    for token in configured:
        if token.lower() == "all":
            expanded.extend(registry)
        elif token in groups:
            expanded.extend(_split_csv(groups[token]))
        else:
            expanded.append(token)
    unknown = sorted(set(expanded).difference(registry))
    if unknown:
        raise ValueError(f"Unknown BUSH-MEG all-protocol method(s): {', '.join(unknown)}.")
    requested_protocols = _parse_protocols(protocols)
    return {
        method
        for method in dict.fromkeys(expanded)
        if registry[method].protocol_category in requested_protocols
        and not (non_oracle and registry[method].protocol_category == 4)
        and (include_oracle or registry[method].protocol_category != 4)
    }


def _available_method_specs(
    specs: Sequence[MethodSpec],
    *,
    base_config: Mapping[str, Any],
    all_protocols: Mapping[str, Any],
    data_dir: str | Path | None,
    participants: str | Sequence[str] | None,
    max_folds: int | None,
    include_heavy: bool,
    participant_limit: int | None,
    smoke_participants: str | Sequence[str] | None,
    window_limit: int | None,
) -> list[MethodSpec]:
    """Return specs that are runnable in the current checkout/config."""

    available_specs: list[MethodSpec] = []
    for spec in specs:
        method_config = _method_config(
            base_config,
            all_protocols,
            spec,
            data_dir=data_dir,
            participants=participants,
            max_folds=max_folds,
            include_heavy=include_heavy,
            participant_limit=participant_limit,
            smoke_participants=smoke_participants,
            window_limit=window_limit,
        )
        settings = _method_settings(all_protocols, spec.method)
        available, _ = _method_availability(
            spec,
            method_config,
            settings=settings,
            include_heavy=include_heavy,
            max_folds=max_folds,
        )
        if available and not spec.metadata().get("inventory_only", False):
            available_specs.append(spec)
    return available_specs


def _config_has_any(config: Mapping[str, Any], paths: Sequence[str]) -> bool:
    def has_path(mapping: Mapping[str, Any], dotted_path: str) -> bool:
        cursor: Any = mapping
        for part in dotted_path.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                return False
            cursor = cursor[part]
        if cursor is None:
            return False
        if isinstance(cursor, str) and not cursor.strip():
            return False
        if isinstance(cursor, Sequence) and not isinstance(cursor, (str, bytes)) and len(cursor) == 0:
            return False
        return True

    return any(has_path(config, path) for path in paths)


def _module_available(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _method_settings(all_protocols: Mapping[str, Any], method: str) -> dict[str, Any]:
    settings = all_protocols.get("method_settings", {}) or {}
    if not isinstance(settings, Mapping):
        raise ValueError("all_protocols.method_settings must be a mapping.")
    value = settings.get(method, {}) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"all_protocols.method_settings.{method} must be a mapping.")
    return dict(value)


def _method_availability(
    spec: MethodSpec,
    config: Mapping[str, Any],
    *,
    settings: Mapping[str, Any] | None = None,
    include_heavy: bool = False,
    max_folds: int | None = None,
) -> tuple[bool, str]:
    reasons: list[str] = []
    settings = dict(settings or {})
    heavy = bool(settings.get("heavy", False))
    enabled = bool(settings.get("enabled", True))
    smoke_enabled = bool(settings.get("smoke_enabled", False))
    if not enabled and not include_heavy:
        if max_folds is None or not smoke_enabled:
            reasons.append("disabled for full-size run; pass --include-heavy or use --max-folds when smoke_enabled=true")
    if spec.required_config_any and not _config_has_any(config, spec.required_config_any):
        reasons.append(f"missing required config value; provide one of: {', '.join(spec.required_config_any)}")
    missing_modules = [module for module in spec.required_modules if not _module_available(module)]
    if missing_modules:
        reasons.append(f"missing required module(s): {', '.join(missing_modules)}")
    if spec.requires_torch and not _torch_available():
        reasons.append("optional torch extra is unavailable")
    if not spec.runnable:
        reasons.append(spec.blocked_reason or f"runner {spec.runner!r} is not runnable")
    return not reasons, "; ".join(reasons)


def _effective_fold_limit(max_folds: int | None, fold_limit: int | None) -> int | None:
    max_folds = _validate_positive_limit("max_folds", max_folds)
    fold_limit = _validate_positive_limit("fold_limit", fold_limit)
    return fold_limit if fold_limit is not None else max_folds


def _missing_required_modules(spec: MethodSpec) -> list[str]:
    return [module for module in spec.required_modules if not _module_available(module)]


def build_registry_audit(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    methods: str | Sequence[str] | None = None,
    protocols: str | Sequence[str | int] | None = None,
    data_dir: str | Path | None = None,
    participants: str | Sequence[str] | None = None,
    max_folds: int | None = None,
    participant_limit: int | None = None,
    smoke_participants: str | Sequence[str] | None = None,
    fold_limit: int | None = None,
    window_limit: int | None = None,
    include_oracle: bool = False,
    include_heavy: bool = False,
    non_oracle: bool = False,
    strict_available: bool = False,
) -> tuple[pd.DataFrame, Path, list[str]]:
    """Write a checkout-completeness audit for every registered all-protocol method."""

    config_path = Path(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_config, all_protocols, _ = _load_runner_config(config_path)
    effective_max_folds = _effective_fold_limit(max_folds, fold_limit)
    common_config = _apply_common_config_updates(
        base_config,
        data_dir=data_dir,
        participants=participants,
        participant_limit=participant_limit,
        smoke_participants=smoke_participants,
        window_limit=window_limit,
    )
    requested = _configured_method_names(
        all_protocols=all_protocols,
        methods=methods,
        protocols=protocols,
        include_oracle=include_oracle,
        non_oracle=non_oracle,
    )
    rows: list[dict[str, Any]] = []
    strict_failures: list[str] = []
    for spec in method_registry().values():
        method_config = _method_config(
            common_config,
            all_protocols,
            spec,
            data_dir=data_dir,
            participants=participants,
            max_folds=effective_max_folds,
            include_heavy=include_heavy,
            participant_limit=participant_limit,
            smoke_participants=smoke_participants,
            window_limit=window_limit,
        )
        settings = _method_settings(all_protocols, spec.method)
        available, skip_reason = _method_availability(
            spec,
            method_config,
            settings=settings,
            include_heavy=include_heavy,
            max_folds=effective_max_folds,
        )
        missing_modules = _missing_required_modules(spec)
        row = {
            "method": spec.method,
            "method_family": spec.method_family,
            "protocol_category": int(spec.protocol_category),
            **spec.protocol.metadata(),
            "runner": spec.runner,
            "required_modules": "|".join(spec.required_modules),
            "required_config_any": "|".join(spec.required_config_any),
            "requires_torch": bool(spec.requires_torch),
            "inventory_only": bool((not spec.runnable) or spec.runner == "unavailable"),
            "implementation_status": "available" if available else "skipped",
            "skip_reason": "" if available else skip_reason,
            "requested": spec.method in requested,
            "missing_required_modules": "|".join(missing_modules),
        }
        rows.append(row)
        if strict_available and spec.method in requested and missing_modules:
            strict_failures.append(f"{spec.method}: missing required module(s): {', '.join(missing_modules)}")
    audit = pd.DataFrame(rows)
    for column in REGISTRY_AUDIT_COLUMNS:
        if column not in audit.columns:
            audit[column] = pd.NA
    extra_columns = [column for column in audit.columns if column not in REGISTRY_AUDIT_COLUMNS]
    audit = audit[REGISTRY_AUDIT_COLUMNS + extra_columns]
    out_path = out_dir / "registry_audit.csv"
    audit.to_csv(out_path, index=False)
    return audit, out_path, strict_failures


def _method_specs_from_manifest(output_dir: Path) -> list[MethodSpec]:
    manifest = _read_json_if_exists(output_dir / "run_manifest.json")
    registry = method_registry()
    method_names = manifest.get("methods")
    if isinstance(method_names, list) and method_names:
        return [registry[str(method)] for method in method_names if str(method) in registry]
    methods_dir = output_dir / "methods"
    if not methods_dir.exists():
        return []
    return [registry[path.name] for path in sorted(methods_dir.iterdir()) if path.is_dir() and path.name in registry]


def _metadata_status_from_stage(stage: str, *, runnable: bool) -> str:
    if stage == "method_skipped":
        return "skipped"
    if stage == "method_failed":
        return "failed"
    if stage == "method_done":
        return "runnable" if runnable else "skipped"
    if not stage:
        return "pending"
    return "running"


def _metadata_from_method_partial(
    method_dir: Path,
    spec: MethodSpec,
    all_protocols: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    status = _read_json_if_exists(method_dir / "status.json")
    settings = _method_settings(all_protocols, spec.method)
    runnable = status.get("stage") not in {"method_skipped", "method_failed"}
    row = {
        **spec.metadata(),
        "method_dir": str(method_dir),
        "method_config": str(method_dir / "config.yml"),
        "raw_summary_csv": str(method_dir / "summary.csv"),
        "raw_predictions_csv": str(method_dir / "predictions.csv"),
        "raw_inner_cv_csv": str(method_dir / "inner_cv.csv"),
        "status_json": str(method_dir / "status.json"),
        "run_log": str(method_dir / "run.log"),
        "summary_partial_csv": str(method_dir / "summary.partial.csv"),
        "predictions_partial_csv": str(method_dir / "predictions.partial.csv"),
        "n_configured_participants": _participant_count_from_config(config),
        "heavy": bool(settings.get("heavy", False)),
        "enabled": bool(settings.get("enabled", True)),
        "smoke_enabled": bool(settings.get("smoke_enabled", False)),
        "runnable": bool(runnable),
        "status": _metadata_status_from_stage(str(status.get("stage", "")), runnable=bool(runnable)),
        "current_stage": status.get("stage", ""),
        "updated_at_utc": status.get("updated_at_utc", ""),
        "skip_reason": status.get("skip_reason", ""),
        "blocked_reason": status.get("skip_reason", status.get("error", "")),
        "resumed": bool(status.get("resumed", False)),
        "timeout_kind": status.get("timeout_kind", ""),
        "timeout_seconds": status.get("timeout_seconds", ""),
    }
    if status.get("stage") == "method_failed":
        row["error_type"] = status.get("error_type", "")
        row["error"] = status.get("error", "")
    return row


def rebuild_all_protocol_outputs_from_partials(
    output_dir: str | Path,
    *,
    all_protocols: Mapping[str, Any] | None = None,
    method_specs: Sequence[MethodSpec] | None = None,
) -> AllProtocolsResult:
    """Rebuild top-level all-protocol CSVs from method-local partial artifacts."""

    output_dir = Path(output_dir)
    all_protocols = all_protocols or {}
    specs = list(method_specs) if method_specs is not None else _method_specs_from_manifest(output_dir)
    summary_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []
    for spec in specs:
        method_dir = output_dir / "methods" / spec.method
        config_path = method_dir / "config.yml"
        config = load_config(config_path) if config_path.exists() else {}
        raw_summary = _read_csv_if_nonempty(method_dir / "summary.partial.csv")
        raw_predictions = _read_csv_if_nonempty(method_dir / "predictions.partial.csv")
        metadata_rows.append(_metadata_from_method_partial(method_dir, spec, all_protocols, config))
        if raw_summary.empty:
            continue
        summary_frames.append(_normalize_summary(raw_summary, raw_predictions, spec=spec, config=config))
        if not raw_predictions.empty:
            prediction_frames.append(_normalize_predictions(raw_predictions, spec=spec))

    summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(columns=SUMMARY_COLUMNS)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    method_metadata = pd.DataFrame(metadata_rows)

    summary_csv = output_dir / "summary.csv"
    predictions_csv = output_dir / "predictions.csv"
    method_metadata_csv = output_dir / "method_metadata.csv"
    provenance_json = output_dir / "provenance.json"
    _write_csv_atomic(summary, summary_csv)
    _write_csv_atomic(predictions, predictions_csv)
    _write_csv_atomic(method_metadata, method_metadata_csv)
    return AllProtocolsResult(
        summary_csv=summary_csv,
        predictions_csv=predictions_csv,
        method_metadata_csv=method_metadata_csv,
        provenance_json=provenance_json,
        summary=summary,
        predictions=predictions,
        method_metadata=method_metadata,
    )


def _run_method(
    spec: MethodSpec,
    *,
    config: Mapping[str, Any],
    all_protocols: Mapping[str, Any],
    method_dir: Path,
    data_dir: str | Path | None,
    participants: str | Sequence[str] | None,
    max_folds: int | None,
    resume: bool,
    include_heavy: bool,
    aggregate_callback: Callable[[], None] | None = None,
    method_timeout_seconds: float | None = None,
    fold_timeout_seconds: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    config = copy.deepcopy(config)
    if spec.runner == "source_loso":
        source_loso_config = config.setdefault("source_loso", {})
        if isinstance(source_loso_config, dict):
            source_loso_config.setdefault("skip_inner_selection_when_single_candidate", True)
    elif spec.runner == "covariance_loso":
        covariance_loso_config = config.setdefault("covariance_loso", {})
        if isinstance(covariance_loso_config, dict):
            covariance_loso_config.setdefault("skip_inner_selection_when_single_candidate", True)
            covariance_grid = covariance_loso_config.setdefault("candidate_grid", {})
            if isinstance(covariance_grid, dict):
                decoding_config = config.get("decoding", {}) if isinstance(config.get("decoding", {}), Mapping) else {}
                preprocessing_config = config.get("preprocessing", {}) if isinstance(config.get("preprocessing", {}), Mapping) else {}
                covariance_grid.setdefault("time_windows", [{"name": "cov_050_300ms", "start": 0.05, "stop": 0.30}])
                covariance_grid.setdefault("feature_modes", ["logeuclidean_covariance"])
                covariance_grid.setdefault("covariance_shrinkages", [0.1])
                covariance_grid.setdefault("covariance_epsilons", [1.0e-6])
                covariance_grid.setdefault("covariance_max_channels", [64])
                covariance_grid.setdefault("decoders", [decoding_config.get("classifier", "multinomial-logistic")])
                covariance_grid.setdefault("emission_modes", [decoding_config.get("emission_mode", "uncalibrated")])
                covariance_grid.setdefault("feature_preprocessors", [preprocessing_config.get("feature_preprocessor", "pca")])
                covariance_grid.setdefault("pca_components", [preprocessing_config.get("pca_components", 64)])
                covariance_grid.setdefault("c_grid", [1.0])
    method_dir.mkdir(parents=True, exist_ok=True)
    summary_path = method_dir / "summary.csv"
    inner_path = method_dir / "inner_cv.csv"
    predictions_path = method_dir / "predictions.csv"
    config_path = method_dir / "config.yml"
    previous_status = _read_json_if_exists(method_dir / "status.json") if resume else {}
    previous_stage = str(previous_status.get("stage", ""))
    if not resume:
        for artifact in (
            summary_path,
            inner_path,
            predictions_path,
            config_path,
            method_dir / "status.json",
            method_dir / "run.log",
            method_dir / "summary.partial.csv",
            method_dir / "predictions.partial.csv",
            method_dir / "inner_cv.partial.csv",
        ):
            if artifact.exists():
                artifact.unlink()
    elif previous_stage and previous_stage not in {"method_done", "method_failed", "method_skipped"} and not summary_path.exists():
        for artifact in (
            method_dir / "status.json",
            method_dir / "run.log",
            method_dir / "summary.partial.csv",
            method_dir / "predictions.partial.csv",
            method_dir / "inner_cv.partial.csv",
        ):
            if artifact.exists():
                artifact.unlink()
        previous_status = {}
        previous_stage = ""
    progress = MethodProgress(
        method_dir,
        method=spec.method,
        aggregate_callback=aggregate_callback,
        method_timeout_seconds=method_timeout_seconds,
        fold_timeout_seconds=fold_timeout_seconds,
    )
    preserve_terminal_status = bool(resume and previous_stage in {"method_failed", "method_skipped"})
    if not preserve_terminal_status:
        progress.initialize_artifacts()
        _yaml_safe_dump(config_path, config)
        progress.update(
            "configured",
            method_family=spec.method_family,
            protocol_category=int(spec.protocol_category),
            runner=spec.runner,
            method_config=str(config_path),
            n_configured_participants=_participant_count_from_config(config),
            max_folds=max_folds,
            method_timeout_seconds=method_timeout_seconds,
            fold_timeout_seconds=fold_timeout_seconds,
        )

    settings = _method_settings(all_protocols, spec.method)
    metadata = {
        **spec.metadata(),
        "method_dir": str(method_dir),
        "method_config": str(config_path),
        "raw_summary_csv": str(summary_path),
        "raw_predictions_csv": str(predictions_path),
        "raw_inner_cv_csv": str(inner_path),
        "status_json": str(progress.status_path),
        "run_log": str(progress.log_path),
        "summary_partial_csv": str(progress.summary_partial_path),
        "predictions_partial_csv": str(progress.predictions_partial_path),
        "n_configured_participants": _participant_count_from_config(config),
        "heavy": bool(settings.get("heavy", False)),
        "enabled": bool(settings.get("enabled", True)),
        "smoke_enabled": bool(settings.get("smoke_enabled", False)),
        "method_timeout_seconds": method_timeout_seconds,
        "fold_timeout_seconds": fold_timeout_seconds,
    }
    if preserve_terminal_status:
        raw_summary = _read_csv_if_nonempty(progress.summary_partial_path)
        raw_predictions = _read_csv_if_nonempty(progress.predictions_partial_path)
        metadata["runnable"] = False
        metadata["status"] = "failed" if previous_stage == "method_failed" else "skipped"
        metadata["resumed"] = True
        metadata["skip_reason"] = previous_status.get("skip_reason", "")
        metadata["blocked_reason"] = previous_status.get("error", previous_status.get("skip_reason", ""))
        metadata["error_type"] = previous_status.get("error_type", "")
        metadata["error"] = previous_status.get("error", "")
        return raw_summary, raw_predictions, metadata
    progress.update("checking_requirements", include_heavy=bool(include_heavy))
    available, skip_reason = _method_availability(
        spec,
        config,
        settings=settings,
        include_heavy=include_heavy,
        max_folds=max_folds,
    )
    metadata["runnable"] = bool(available)
    metadata["status"] = "runnable" if available else "skipped"
    metadata["skip_reason"] = skip_reason
    if not available:
        metadata["blocked_reason"] = skip_reason
        progress.update("method_skipped", skip_reason=skip_reason)
        return pd.DataFrame(), pd.DataFrame(), metadata
    if resume and summary_path.exists():
        raw_summary = _read_csv_if_nonempty(summary_path)
        raw_predictions = _read_csv_if_nonempty(predictions_path)
        _copy_if_exists(summary_path, progress.summary_partial_path)
        _copy_if_exists(predictions_path, progress.predictions_partial_path)
        metadata["resumed"] = True
        progress.update("method_done", resumed=True, n_summary_rows=len(raw_summary), n_prediction_rows=len(raw_predictions))
        return raw_summary, raw_predictions, metadata

    progress.start_method_timeout()
    progress.update("loading_subjects")
    try:
        if spec.runner == "source_loso":
            raw_summary = _run_source_loso_method(
                config_path,
                summary_path=progress.summary_partial_path,
                inner_path=progress.inner_partial_path,
                predictions_path=progress.predictions_partial_path,
                max_folds=max_folds,
                progress_callback=progress,
            )
        elif spec.runner == "covariance_loso":
            raw_summary = _run_covariance_method(
                config_path,
                summary_path=progress.summary_partial_path,
                inner_path=progress.inner_partial_path,
                predictions_path=progress.predictions_partial_path,
                max_folds=max_folds,
                progress_callback=progress,
            )
        elif spec.runner == "supervised_lowrank_loso":
            raw_summary = _run_supervised_lowrank_method(
                config_path,
                summary_path=progress.summary_partial_path,
                inner_path=progress.inner_partial_path,
                predictions_path=progress.predictions_partial_path,
                max_folds=max_folds,
                progress_callback=progress,
            )
        elif spec.runner in {"protocol3_few_shot", "protocol3_source_plus_target", "protocol3_target_calibrated_alignment", "protocol3_target_calibrated_gaussian", "protocol3_lora_few_shot"}:
            raw_summary = _run_protocol3_few_shot_method(
                config_path,
                summary_path=progress.summary_partial_path,
                inner_path=progress.inner_partial_path,
                predictions_path=progress.predictions_partial_path,
                max_folds=max_folds,
                progress_callback=progress,
                method_spec=spec,
            )
        elif spec.runner == "loso_decode":
            raw_summary = _run_memory_bounded_decode(
                config,
                summary_path=progress.summary_partial_path,
                data_dir=data_dir,
                participants=participants,
                max_folds=max_folds,
                resume=resume,
                progress_callback=progress,
            )
        else:
            metadata["runnable"] = False
            metadata["blocked_reason"] = metadata.get("blocked_reason") or f"Unsupported runner {spec.runner!r}."
            progress.update("method_skipped", skip_reason=metadata["blocked_reason"])
            return pd.DataFrame(), pd.DataFrame(), metadata
    except RunTimeoutError as exc:
        metadata["status"] = "failed"
        metadata["runnable"] = False
        metadata["timeout_kind"] = exc.kind
        metadata["timeout_seconds"] = exc.seconds
        metadata["blocked_reason"] = str(exc)
        metadata["skip_reason"] = str(exc)
        progress.update("method_failed", error_type=type(exc).__name__, error=str(exc), timeout_kind=exc.kind, timeout_seconds=exc.seconds)
        return pd.DataFrame(), pd.DataFrame(), metadata
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["runnable"] = False
        metadata["blocked_reason"] = str(exc)
        progress.update("method_failed", error_type=type(exc).__name__, error=str(exc))
        return pd.DataFrame(), pd.DataFrame(), metadata

    raw_predictions = _read_csv_if_nonempty(progress.predictions_partial_path)
    _copy_if_exists(progress.summary_partial_path, summary_path)
    _copy_if_exists(progress.inner_partial_path, inner_path)
    _copy_if_exists(progress.predictions_partial_path, predictions_path)
    metadata["resumed"] = False
    progress.update("method_done", resumed=False, n_summary_rows=len(raw_summary), n_prediction_rows=len(raw_predictions))
    return raw_summary, raw_predictions, metadata


def run_bushmeg_all_protocols(
    *,
    config_path: str | Path = DEFAULT_CONFIG,
    data_dir: str | Path | None = None,
    out_dir: str | Path = DEFAULT_OUT_DIR,
    participants: str | Sequence[str] | None = None,
    methods: str | Sequence[str] | None = None,
    protocols: str | Sequence[str | int] | None = None,
    max_folds: int | None = None,
    participant_limit: int | None = None,
    smoke_participants: str | Sequence[str] | None = None,
    fold_limit: int | None = None,
    window_limit: int | None = None,
    resume: bool = True,
    n_jobs: int = 1,
    include_oracle: bool = False,
    include_heavy: bool = False,
    available_only: bool = False,
    non_oracle: bool = False,
    method_timeout_seconds: float | None = None,
    fold_timeout_seconds: float | None = None,
) -> AllProtocolsResult:
    """Run selected BUSH-MEG protocols and write unified artifacts."""

    config_path = Path(config_path)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_config, all_protocols, base_config_path = _load_runner_config(config_path)
    effective_max_folds = _effective_fold_limit(max_folds, fold_limit)
    selected = _selected_methods(
        all_protocols=all_protocols,
        methods=methods,
        protocols=protocols,
        include_oracle=include_oracle,
        non_oracle=non_oracle,
    )
    if available_only:
        selected = _available_method_specs(
            selected,
            base_config=base_config,
            all_protocols=all_protocols,
            data_dir=data_dir,
            participants=participants,
            max_folds=effective_max_folds,
            include_heavy=include_heavy,
            participant_limit=participant_limit,
            smoke_participants=smoke_participants,
            window_limit=window_limit,
        )
    method_timeout_seconds = _validate_timeout_seconds("method_timeout_seconds", method_timeout_seconds)
    fold_timeout_seconds = _validate_timeout_seconds("fold_timeout_seconds", fold_timeout_seconds)
    summary_csv = output_dir / "summary.csv"
    predictions_csv = output_dir / "predictions.csv"
    method_metadata_csv = output_dir / "method_metadata.csv"
    provenance_json = output_dir / "provenance.json"
    run_manifest_json = output_dir / "run_manifest.json"
    run_manifest_json.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(UTC).isoformat(),
                "config_path": str(config_path),
                "base_config_path": str(base_config_path),
                "methods": [spec.method for spec in selected],
                "protocols": sorted({int(spec.protocol_category) for spec in selected}),
                "summary_csv": str(summary_csv),
                "predictions_csv": str(predictions_csv),
                "method_metadata_csv": str(method_metadata_csv),
                "method_timeout_seconds": method_timeout_seconds,
                "fold_timeout_seconds": fold_timeout_seconds,
                "available_only": bool(available_only),
                "non_oracle": bool(non_oracle),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    def rebuild_top_level_from_partials() -> None:
        rebuild_all_protocol_outputs_from_partials(output_dir, all_protocols=all_protocols, method_specs=selected)

    summary_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    for spec in selected:
        method_config = _method_config(
            base_config,
            all_protocols,
            spec,
            data_dir=data_dir,
            participants=participants,
            max_folds=effective_max_folds,
            include_heavy=include_heavy,
            participant_limit=participant_limit,
            smoke_participants=smoke_participants,
            window_limit=window_limit,
        )
        method_participants = _participant_ids_from_config(method_config)
        raw_summary, raw_predictions, metadata = _run_method(
            spec,
            config=method_config,
            all_protocols=all_protocols,
            method_dir=output_dir / "methods" / spec.method,
            data_dir=data_dir,
            participants=method_participants,
            max_folds=effective_max_folds,
            resume=resume,
            include_heavy=include_heavy,
            aggregate_callback=rebuild_top_level_from_partials,
            method_timeout_seconds=method_timeout_seconds,
            fold_timeout_seconds=fold_timeout_seconds,
        )
        metadata_rows.append(metadata)
        if raw_summary.empty:
            continue
        summary_frames.append(_normalize_summary(raw_summary, raw_predictions, spec=spec, config=method_config))
        prediction_frames.append(_normalize_predictions(raw_predictions, spec=spec))

    rebuilt = rebuild_all_protocol_outputs_from_partials(output_dir, all_protocols=all_protocols, method_specs=selected)
    summary = rebuilt.summary
    predictions = rebuilt.predictions
    method_metadata = rebuilt.method_metadata
    provenance = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config_path": str(config_path),
        "base_config_path": str(base_config_path),
        "out_dir": str(output_dir),
        "data_dir": "" if data_dir is None else str(data_dir),
        "participants": "" if participants is None else participants,
        "effective_participants": _participant_ids_from_config(
            _apply_common_config_updates(
                base_config,
                data_dir=data_dir,
                participants=participants,
                participant_limit=participant_limit,
                smoke_participants=smoke_participants,
                window_limit=window_limit,
            )
        ),
        "participant_limit": participant_limit,
        "smoke_participants": "" if smoke_participants is None else smoke_participants,
        "methods": [spec.method for spec in selected],
        "protocols": sorted({int(spec.protocol_category) for spec in selected}),
        "max_folds": effective_max_folds,
        "requested_max_folds": max_folds,
        "fold_limit": fold_limit,
        "window_limit": window_limit,
        "resume": bool(resume),
        "n_jobs": int(n_jobs),
        "n_jobs_note": "serial execution; n_jobs is accepted for interface compatibility",
        "include_oracle": bool(include_oracle),
        "include_heavy": bool(include_heavy),
        "available_only": bool(available_only),
        "non_oracle": bool(non_oracle),
        "method_timeout_seconds": method_timeout_seconds,
        "fold_timeout_seconds": fold_timeout_seconds,
        "summary_csv": str(summary_csv),
        "predictions_csv": str(predictions_csv),
        "method_metadata_csv": str(method_metadata_csv),
        "run_manifest_json": str(run_manifest_json),
        "top_level_csvs_rebuilt_from_partials": True,
    }
    provenance_json.write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return AllProtocolsResult(
        summary_csv=summary_csv,
        predictions_csv=predictions_csv,
        method_metadata_csv=method_metadata_csv,
        provenance_json=provenance_json,
        summary=summary,
        predictions=predictions,
        method_metadata=method_metadata,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run unified BUSH-MEG evaluations across protocol categories.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="All-protocols config, e.g. configs/bush_meg/all_protocols.yml.")
    parser.add_argument("--data-dir", type=Path, default=None, help="BUSH-MEG data directory. Defaults to BUSH_MEG_DATA_DIR or dataset.root.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for unified all-protocol artifacts.")
    parser.add_argument("--participants", default=None, help="Participant ids/ranges, e.g. 1-4,6,8.")
    parser.add_argument("--methods", default=None, help="Comma-separated method ids. Defaults to all runnable Protocol 1/2 methods.")
    parser.add_argument("--protocols", default=None, help="Comma-separated protocol categories to run. Defaults to 1,2,3.")
    parser.add_argument("--max-folds", type=int, default=None, help="Maximum outer folds for smoke tests where supported by the underlying runner.")
    parser.add_argument("--participant-limit", type=int, default=None, help="Limit the configured participant list before any data loading/caching.")
    parser.add_argument("--smoke-participants", default=None, help="Participant ids/ranges for tiny smoke runs, applied before loading/caching.")
    parser.add_argument("--fold-limit", type=int, default=None, help="Alias/override for --max-folds that limits outer folds for tiny runs.")
    parser.add_argument("--window-limit", type=int, default=None, help="Limit preprocessing/source-LOSO window centers before running methods.")
    parser.add_argument("--method-timeout-seconds", type=float, default=None, help="Fail a method and continue if it exceeds this wall-clock timeout.")
    parser.add_argument("--fold-timeout-seconds", type=float, default=None, help="Fail a method and continue if any outer fold exceeds this wall-clock timeout.")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--n-jobs", type=int, default=1, help="Accepted for interface compatibility; execution is serial.")
    parser.add_argument("--include-oracle", action="store_true", help="Allow Protocol 4 oracle/debug methods.")
    parser.add_argument("--non-oracle", action="store_true", help="Exclude Protocol 4 oracle/debug methods even when they are otherwise selected.")
    parser.add_argument("--include-heavy", dest="include_heavy", action="store_true", help="Allow full-size compute-heavy methods whose config has enabled=false.")
    parser.add_argument("--exclude-heavy", dest="include_heavy", action="store_false", help="Skip compute-heavy methods; this is the default.")
    parser.add_argument("--available-only", action="store_true", help="Run only methods available in this checkout/config and exclude inventory-only methods.")
    parser.add_argument(
        "--canary",
        action="store_true",
        help="Tiny source_loso_logistic canary: --available-only --non-oracle --participants 1,2,3 --fold-limit 1 --window-limit 1 --methods source_loso_logistic --protocols 1.",
    )
    parser.add_argument("--audit-registry", action="store_true", help="Write registry_audit.csv and exit without running any evaluation.")
    parser.add_argument("--strict-available", action="store_true", help="With --audit-registry, exit nonzero if any requested method has missing required modules.")
    parser.add_argument("--profile-load-only", action="store_true", help="Load requested BUSH-MEG participants, write load_profile.csv, and exit before fitting.")
    parser.add_argument("--profile-one-fold", action="store_true", help="Build one fold/window feature matrix, fit one logistic model, write one_fold_profile.json, and exit.")
    args = parser.parse_args(argv)

    if args.profile_load_only and args.profile_one_fold:
        parser.error("--profile-load-only and --profile-one-fold are mutually exclusive.")

    if args.canary:
        args.available_only = True
        args.non_oracle = True
        args.participants = "1,2,3"
        args.fold_limit = 1
        args.window_limit = 1
        args.methods = "source_loso_logistic"
        args.protocols = "1"

    if args.audit_registry:
        audit, audit_csv, strict_failures = build_registry_audit(
            config_path=args.config,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            participants=args.participants,
            methods=args.methods,
            protocols=args.protocols,
            max_folds=args.max_folds,
            participant_limit=args.participant_limit,
            smoke_participants=args.smoke_participants,
            fold_limit=args.fold_limit,
            window_limit=args.window_limit,
            include_oracle=args.include_oracle,
            include_heavy=args.include_heavy,
            non_oracle=args.non_oracle,
            strict_available=args.strict_available,
        )
        display_columns = [
            "method",
            "protocol_category",
            "implementation_status",
            "skip_reason",
        ]
        print(audit[display_columns].to_string(index=False))
        print(f"Wrote registry audit: {audit_csv}")
        if strict_failures:
            print("Strict availability failures:")
            for failure in strict_failures:
                print(f"- {failure}")
            return 2
        return 0

    if args.profile_load_only:
        profile_bushmeg_load_only(
            config_path=args.config,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            participants=args.participants,
            participant_limit=args.participant_limit,
            smoke_participants=args.smoke_participants,
            window_limit=args.window_limit,
        )
        return 0

    if args.profile_one_fold:
        profile_bushmeg_one_fold(
            config_path=args.config,
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            participants=args.participants,
            methods=args.methods,
            protocols=args.protocols,
            max_folds=args.max_folds,
            participant_limit=args.participant_limit,
            smoke_participants=args.smoke_participants,
            fold_limit=args.fold_limit,
            window_limit=args.window_limit,
            include_oracle=args.include_oracle,
            include_heavy=args.include_heavy,
            non_oracle=args.non_oracle,
        )
        return 0

    result = run_bushmeg_all_protocols(
        config_path=args.config,
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        participants=args.participants,
        methods=args.methods,
        protocols=args.protocols,
        max_folds=args.max_folds,
        participant_limit=args.participant_limit,
        smoke_participants=args.smoke_participants,
        fold_limit=args.fold_limit,
        window_limit=args.window_limit,
        resume=args.resume,
        n_jobs=args.n_jobs,
        include_oracle=args.include_oracle,
        include_heavy=args.include_heavy,
        available_only=args.available_only,
        non_oracle=args.non_oracle,
        method_timeout_seconds=args.method_timeout_seconds,
        fold_timeout_seconds=args.fold_timeout_seconds,
    )
    print(f"Wrote summary: {result.summary_csv}")
    print(f"Wrote predictions: {result.predictions_csv}")
    print(f"Wrote method metadata: {result.method_metadata_csv}")
    print(f"Wrote provenance: {result.provenance_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

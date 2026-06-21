"""Make oracle target-alignment runs explicit in metadata and diagnostics.

``oracle_target_calibrated_alignment`` intentionally fits the held-out target
projection from scored target labels or anchor values.  It is a diagnostic upper
bound, not a benchmark-valid protocol.  The core source-alignment implementation
already enforces the guardrails; this compatibility patch makes the fitted target
transform names and source-inner diagnostic labels unambiguous so downstream CSVs
cannot confuse oracle target fitting with source-only group projection or disjoint
target calibration.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from types import ModuleType
from typing import Any

_PATCH_MARKER = "_neureptrace_source_alignment_oracle_patch_installed"
_TARGET_MODULE = "neureptrace.decoding.source_alignment"
_DOWNSTREAM_MODULES = (
    "neureptrace.mne_time_decode",
    "neureptrace.mne_time_decode_foldlocal",
    "neureptrace.bushmeg_source_loso",
)


def _oracle_target_projection_fit_name(method: object, current: object = "") -> str:
    """Return an explicit oracle target-projection fit label."""

    text = str(current or "")
    if text.startswith("oracle_target_calibrated_"):
        return text
    if text.startswith("oracle_"):
        return "oracle_target_calibrated_" + text.removeprefix("oracle_")

    method_name = str(method)
    if method_name == "mcca":
        return "oracle_target_calibrated_template_ridge_least_squares"
    if method_name in {"procrustes", "hyperalignment"}:
        return "oracle_target_calibrated_template_procrustes"
    if method_name == "contrastive":
        return "oracle_target_calibrated_contrastive_ridge_projection"
    return f"oracle_target_calibrated_{method_name}_projection"


def _oracle_metadata(metadata: Mapping[str, Any], *, transform_name: str, source_alignment: ModuleType) -> dict[str, Any]:
    updated = dict(metadata)
    updated.update(
        {
            "alignment_target_projection": source_alignment.ORACLE_TARGET_CALIBRATED_ALIGNMENT,
            "alignment_target_projection_fit": transform_name,
            "alignment_target_calibrated": False,
            "alignment_oracle_target_calibrated": True,
            "alignment_debug_upper_bound": True,
            "alignment_valid_for_benchmark": False,
            "alignment_valid_for_strict_source_only": False,
            "alignment_protocol": source_alignment.ORACLE_TARGET_CALIBRATED_ALIGNMENT,
            "alignment_protocol_note": (
                "uses scored held-out target labels or anchor values to fit the target projection; "
                "debug upper bound only; not valid for benchmark claims"
            ),
        }
    )
    return updated


def _oracle_diagnostics(diagnostics: Mapping[str, Any], *, transform_name: str, source_alignment: ModuleType) -> dict[str, Any]:
    updated = dict(diagnostics)
    updated.update(
        {
            "target_transform_type": transform_name,
            "alignment_target_projection": source_alignment.ORACLE_TARGET_CALIBRATED_ALIGNMENT,
            "alignment_target_projection_fit": transform_name,
            "alignment_oracle_target_calibrated": True,
            "alignment_debug_upper_bound": True,
            "alignment_valid_for_benchmark": False,
            "alignment_valid_for_strict_source_only": False,
            "alignment_protocol": source_alignment.ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        }
    )
    return updated


def _patch_source_alignment(source_alignment: ModuleType) -> None:
    if getattr(source_alignment, _PATCH_MARKER, False):
        return

    original_static_metadata = source_alignment.SourceAlignmentConfig.static_metadata
    original_align_train_test_features = source_alignment.align_train_test_features
    original_source_inner_validation_type = source_alignment._source_inner_validation_type

    def static_metadata(config) -> dict[str, Any]:
        metadata = dict(original_static_metadata(config))
        if getattr(config, "oracle_target_calibrated", False):
            transform_name = _oracle_target_projection_fit_name(config.method, metadata.get("alignment_target_projection_fit", ""))
            metadata = _oracle_metadata(metadata, transform_name=transform_name, source_alignment=source_alignment)
        return metadata

    def _source_inner_validation_type(config, *, compute_source_inner_diagnostics: bool) -> str:
        if compute_source_inner_diagnostics and getattr(config, "oracle_target_calibrated", False):
            return "source_loso_nearest_centroid_oracle_target_projection"
        return original_source_inner_validation_type(
            config,
            compute_source_inner_diagnostics=compute_source_inner_diagnostics,
        )

    def align_train_test_features(*, config, **kwargs):
        result = original_align_train_test_features(config=config, **kwargs)
        if not getattr(config, "oracle_target_calibrated", False):
            return result

        transform_name = _oracle_target_projection_fit_name(
            config.method,
            result.metadata.get("alignment_target_projection_fit", result.diagnostics.get("target_transform_type", "")),
        )
        return source_alignment.SourceAlignmentResult(
            train_features=result.train_features,
            test_features=result.test_features,
            metadata=_oracle_metadata(result.metadata, transform_name=transform_name, source_alignment=source_alignment),
            diagnostics=_oracle_diagnostics(result.diagnostics, transform_name=transform_name, source_alignment=source_alignment),
        )

    source_alignment.SourceAlignmentConfig.static_metadata = static_metadata
    source_alignment._source_inner_validation_type = _source_inner_validation_type
    source_alignment.align_train_test_features = align_train_test_features
    setattr(source_alignment, _PATCH_MARKER, True)

    for module_name in _DOWNSTREAM_MODULES:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "align_train_test_features"):
            module.align_train_test_features = align_train_test_features


def install() -> None:
    """Install explicit oracle target-alignment metadata patch."""

    source_alignment = importlib.import_module(_TARGET_MODULE)
    _patch_source_alignment(source_alignment)

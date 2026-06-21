"""Patch oracle target-alignment provenance and diagnostics.

``oracle_target_calibrated_alignment`` intentionally fits the target projection
from the scored held-out target rows. That is useful as a debug upper bound, but
it must be unmistakable in metadata and diagnostics so downstream reports cannot
mistake the projection for a benchmark-valid calibrated or source-only path.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Hashable, Sequence
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.decoding.source_alignment"
_PATCH_MARKER = "_neureptrace_source_alignment_oracle_patch_installed"
_FINDER_MARKER = "_neureptrace_source_alignment_oracle_finder"
_ORACLE_PROTOCOL_NOTE = (
    "uses scored held-out target labels or anchors to fit the target projection; "
    "debug upper bound only; not valid for benchmark claims"
)
_ORACLE_PROJECTION_SOURCE = "scored_heldout_target_rows"


def _is_oracle_config(config: Any) -> bool:
    return bool(getattr(config, "oracle_target_calibrated", False))


def _row_count(features: Sequence[Sequence[float]] | np.ndarray | None) -> int | str:
    if features is None:
        return ""
    return int(np.asarray(features).shape[0])


def _uses_decoder_label_anchors(source_alignment: ModuleType, config: Any) -> bool:
    helper = getattr(source_alignment, "_uses_decoder_label_anchors", None)
    if callable(helper):
        return bool(helper(config.anchor_mode))
    return str(getattr(config, "anchor_mode", "")) in getattr(
        source_alignment,
        "SOURCE_ALIGNMENT_CLASS_ANCHOR_MODES",
        ("class_mean", "class_repetition"),
    )


def _oracle_projection_fit_name(config: Any, existing: object = "") -> str:
    text = "" if existing is None else str(existing)
    if text.startswith("oracle_target_"):
        return text
    method = str(getattr(config, "method", ""))
    if method == "contrastive" or "contrastive" in text:
        return "oracle_target_contrastive_ridge_projection"
    if method == "mcca" or "ridge_least_squares" in text:
        return "oracle_target_template_ridge_least_squares"
    if method in {"procrustes", "hyperalignment"} or "procrustes" in text:
        return "oracle_target_template_procrustes"
    return "oracle_target_projection"


def _mark_oracle_mapping(
    source_alignment: ModuleType,
    config: Any,
    values: dict[str, Any],
    *,
    test_features: Sequence[Sequence[float]] | np.ndarray | None,
    target_labels: Sequence[Any] | np.ndarray | None,
    target_anchor_values: Sequence[Any] | np.ndarray | None,
    projection_fit: str,
) -> dict[str, Any]:
    marked = dict(values)
    class_label_anchors = _uses_decoder_label_anchors(source_alignment, config)
    labels_used = target_labels is not None
    anchors_used = target_anchor_values is not None or (labels_used and class_label_anchors)
    marked.update(
        {
            "alignment_strict_source_only": False,
            "alignment_uses_unlabeled_target_data": False,
            "alignment_target_projection_fit": projection_fit,
            "alignment_target_labels_used": bool(labels_used),
            "alignment_target_anchor_values_used": bool(anchors_used),
            "alignment_oracle_target_calibrated": True,
            "alignment_debug_upper_bound": True,
            "alignment_valid_for_benchmark": False,
            "alignment_valid_for_strict_source_only": False,
            "alignment_protocol": source_alignment.ORACLE_TARGET_CALIBRATED_ALIGNMENT,
            "alignment_protocol_note": _ORACLE_PROTOCOL_NOTE,
            "alignment_oracle_target_projection_source": _ORACLE_PROJECTION_SOURCE,
            "alignment_oracle_target_raw_rows_used": _row_count(test_features),
            "alignment_oracle_target_labels_or_anchors_used": bool(labels_used or anchors_used),
        }
    )
    if "target_transform_type" in marked:
        marked["target_transform_type"] = projection_fit
    return marked


def _patch_source_alignment(source_alignment: ModuleType) -> None:
    if getattr(source_alignment, _PATCH_MARKER, False):
        return

    original_static_metadata = source_alignment.SourceAlignmentConfig.static_metadata
    original_source_inner_validation_type = source_alignment._source_inner_validation_type
    original_align_train_test_features = source_alignment.align_train_test_features

    def static_metadata(config) -> dict[str, Any]:
        metadata = dict(original_static_metadata(config))
        if not _is_oracle_config(config):
            return metadata
        projection_fit = _oracle_projection_fit_name(config, metadata.get("alignment_target_projection_fit", ""))
        metadata.update(
            {
                "alignment_strict_source_only": False,
                "alignment_uses_unlabeled_target_data": False,
                "alignment_target_projection_fit": projection_fit,
                "alignment_oracle_target_calibrated": True,
                "alignment_debug_upper_bound": True,
                "alignment_valid_for_benchmark": False,
                "alignment_valid_for_strict_source_only": False,
                "alignment_protocol": source_alignment.ORACLE_TARGET_CALIBRATED_ALIGNMENT,
                "alignment_protocol_note": _ORACLE_PROTOCOL_NOTE,
                "alignment_oracle_target_projection_source": _ORACLE_PROJECTION_SOURCE,
            }
        )
        return metadata

    def _source_inner_validation_type(config, *, compute_source_inner_diagnostics: bool) -> str:
        if not compute_source_inner_diagnostics:
            return ""
        if _is_oracle_config(config):
            return "source_loso_nearest_centroid_oracle_target_projection_same_rows"
        return original_source_inner_validation_type(
            config,
            compute_source_inner_diagnostics=compute_source_inner_diagnostics,
        )

    def align_train_test_features(
        *,
        train_features: Sequence[Sequence[float]] | np.ndarray,
        train_labels: Sequence[Any] | np.ndarray,
        train_subject_ids: Sequence[Hashable] | np.ndarray,
        test_features: Sequence[Sequence[float]] | np.ndarray,
        config,
        target_labels: Sequence[Any] | np.ndarray | None = None,
        train_anchor_values: Sequence[Any] | np.ndarray | None = None,
        target_anchor_values: Sequence[Any] | np.ndarray | None = None,
        target_calibration_features: Sequence[Sequence[float]] | np.ndarray | None = None,
        target_calibration_labels: Sequence[Any] | np.ndarray | None = None,
        target_calibration_anchor_values: Sequence[Any] | np.ndarray | None = None,
        compute_source_inner_diagnostics: bool = True,
    ):
        result = original_align_train_test_features(
            train_features=train_features,
            train_labels=train_labels,
            train_subject_ids=train_subject_ids,
            test_features=test_features,
            config=config,
            target_labels=target_labels,
            train_anchor_values=train_anchor_values,
            target_anchor_values=target_anchor_values,
            target_calibration_features=target_calibration_features,
            target_calibration_labels=target_calibration_labels,
            target_calibration_anchor_values=target_calibration_anchor_values,
            compute_source_inner_diagnostics=compute_source_inner_diagnostics,
        )
        if not _is_oracle_config(config):
            return result

        existing_projection_fit = result.metadata.get(
            "alignment_target_projection_fit",
            result.diagnostics.get("target_transform_type", ""),
        )
        projection_fit = _oracle_projection_fit_name(config, existing_projection_fit)
        metadata = _mark_oracle_mapping(
            source_alignment,
            config,
            result.metadata,
            test_features=test_features,
            target_labels=target_labels,
            target_anchor_values=target_anchor_values,
            projection_fit=projection_fit,
        )
        diagnostics = _mark_oracle_mapping(
            source_alignment,
            config,
            result.diagnostics,
            test_features=test_features,
            target_labels=target_labels,
            target_anchor_values=target_anchor_values,
            projection_fit=projection_fit,
        )
        return source_alignment.SourceAlignmentResult(
            train_features=result.train_features,
            test_features=result.test_features,
            metadata=metadata,
            diagnostics=diagnostics,
        )

    source_alignment.SourceAlignmentConfig.static_metadata = static_metadata
    source_alignment._source_inner_validation_type = _source_inner_validation_type
    source_alignment.align_train_test_features = align_train_test_features
    setattr(source_alignment, _PATCH_MARKER, True)


class _SourceAlignmentOraclePatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader: importlib.abc.Loader) -> None:
        self.wrapped_loader = wrapped_loader

    def create_module(self, spec):  # type: ignore[override]
        create_module = getattr(self.wrapped_loader, "create_module", None)
        if create_module is None:
            return None
        return create_module(spec)

    def exec_module(self, module: ModuleType) -> None:
        self.wrapped_loader.exec_module(module)
        _patch_source_alignment(module)


class _SourceAlignmentOraclePatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _SourceAlignmentOraclePatchLoader):
            return spec
        spec.loader = _SourceAlignmentOraclePatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install explicit oracle target-alignment provenance semantics."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_source_alignment(loaded)
        return

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _SourceAlignmentOraclePatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)

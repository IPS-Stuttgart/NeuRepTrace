"""Patch pseudo-label target-calibrated class-repetition alignment caps.

``pseudo_label_target_calibrated_alignment`` uses target calibration rows just like
``target_calibrated_alignment``; the only difference is that the row labels are
classifier-generated pseudo labels.  Class-repetition source fits must therefore
cap the source anchor repetitions to ``alignment_target_calibration_per_anchor``.
Otherwise source fits can request many repetitions per anchor while the target
pseudo-calibration matrix contains only a small pseudo-labeled calibration subset,
causing target projection failures or misleading availability diagnostics.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from collections.abc import Hashable, Mapping
from types import ModuleType

import numpy as np

_TARGET_MODULE = "neureptrace.decoding.source_alignment"
_PATCH_MARKER = "_neureptrace_source_alignment_pseudo_calibration_patch_installed"
_FINDER_MARKER = "_neureptrace_source_alignment_pseudo_calibration_finder"


def _patch_source_alignment(source_alignment: ModuleType) -> None:
    if getattr(source_alignment, _PATCH_MARKER, False):
        return

    def _effective_repetitions_per_class(
        labels_by_subject: Mapping[Hashable, np.ndarray],
        sample_mode: str,
        config,
    ) -> int | None:
        if sample_mode != "class_repetition":
            return None
        subject_ids = tuple(labels_by_subject)
        first_classes = source_alignment._ordered_unique_anchor_values(labels_by_subject[subject_ids[0]])
        counts = []
        for subject_id in subject_ids:
            classes = source_alignment._ordered_unique_anchor_values(labels_by_subject[subject_id])
            if not source_alignment._same_anchor_value_set(first_classes, classes):
                raise ValueError(f"Subject {subject_id!r} does not contain the common alignment classes.")
            counts.extend(
                source_alignment._count_anchor_value(labels_by_subject[subject_id], class_label)
                for class_label in first_classes
            )
        available = min(counts)
        if available < 1:
            raise ValueError("Every source subject must have at least one sample per alignment class.")
        repetition_cap = available if config.repetition_cap is None else int(config.repetition_cap)
        calibration_cap = (
            int(config.target_calibration_per_anchor)
            if config.target_calibrated or getattr(config, "pseudo_label_target_calibrated", False)
            else available
        )
        return int(min(available, repetition_cap, calibration_cap))

    def _source_alignment_repetition_selection(config, sample_mode: str) -> str:
        """Return source-anchor repetition selection for class-repetition alignment."""

        if sample_mode == "class_repetition" and (
            config.target_calibrated or getattr(config, "pseudo_label_target_calibrated", False)
        ):
            return "first"
        return source_alignment.DEFAULT_CLASS_LIMIT_SELECTION

    source_alignment._effective_repetitions_per_class = _effective_repetitions_per_class
    source_alignment._source_alignment_repetition_selection = _source_alignment_repetition_selection
    setattr(source_alignment, _PATCH_MARKER, True)


class _SourceAlignmentPseudoCalibrationPatchLoader(importlib.abc.Loader):
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


class _SourceAlignmentPseudoCalibrationPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _SourceAlignmentPseudoCalibrationPatchLoader):
            return spec
        spec.loader = _SourceAlignmentPseudoCalibrationPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install pseudo-label calibration caps for source-alignment helpers."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_source_alignment(loaded)
        return

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _SourceAlignmentPseudoCalibrationPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)

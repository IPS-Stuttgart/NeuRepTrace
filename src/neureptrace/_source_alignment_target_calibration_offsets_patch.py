"""Finalize source-alignment patch composition and calibrated repetition offsets.

Several source-alignment extensions are installed lazily through import hooks.  The
last installed hook is the one that sees a future ``decoding.source_alignment``
import first, so this final patch explicitly applies the earlier source-alignment
patches in the intended order before adding its own guardrail.

The guardrail fixes target-calibrated and oracle class-repetition projections.
Source fits may select arbitrary source repetition offsets, but separately
prepared target calibration matrices and held-out oracle target matrices are
local target rows.  Reusing source repetition offsets on those rows can index
outside the available target repetitions or silently pick the wrong rows.  For
those target projection modes we therefore use offsets local to the target
matrix.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any

import numpy as np

_TARGET_MODULE = "neureptrace.decoding.source_alignment"
_PATCH_MARKER = "_neureptrace_source_alignment_target_calibration_offsets_patch_installed"
_FINDER_MARKER = "_neureptrace_source_alignment_target_calibration_offsets_finder"
_SOURCE_ALIGNMENT_PATCH_MODULES = (
    "neureptrace._source_alignment_anchor_patch",
    "neureptrace._source_alignment_pseudo_calibration_patch",
    "neureptrace._source_alignment_pseudo_repetition_patch",
    "neureptrace._source_alignment_optimal_transport_patch",
    "neureptrace._mne_alignment_calibration_anchor_patch",
    "neureptrace._source_alignment_contrastive_patch",
    "neureptrace._source_alignment_oracle_patch",
)


def _apply_source_alignment_patch(module_name: str, source_alignment: ModuleType) -> None:
    """Apply a source-alignment runtime patch module if it exposes one."""

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return
    patch = getattr(module, "_patch_source_alignment", None)
    if callable(patch):
        patch(source_alignment)


def _calibrated_projection_repetition_offsets(
    *,
    classes: Sequence[Any] | np.ndarray,
    config: Any,
    n_repetitions_per_class: int | None,
    selected_offsets_by_class: Mapping[int, Sequence[int] | np.ndarray] | None,
) -> Mapping[int, np.ndarray] | None:
    """Return target-local offsets for calibrated target projection matrices."""

    if n_repetitions_per_class is None:
        return selected_offsets_by_class
    if not bool(
        getattr(config, "target_calibrated", False)
        or getattr(config, "pseudo_label_target_calibrated", False)
        or getattr(config, "oracle_target_calibrated", False)
    ):
        return selected_offsets_by_class

    repetitions = int(n_repetitions_per_class)
    if repetitions < 1:
        return selected_offsets_by_class
    return {class_position: np.arange(repetitions, dtype=int) for class_position, _class_label in enumerate(classes)}


def _patch_source_alignment(source_alignment: ModuleType) -> None:
    if getattr(source_alignment, _PATCH_MARKER, False):
        return

    for module_name in _SOURCE_ALIGNMENT_PATCH_MODULES:
        _apply_source_alignment_patch(module_name, source_alignment)

    original_target_alignment_matrix = source_alignment._target_alignment_matrix

    def _target_alignment_matrix(
        features: np.ndarray,
        labels: np.ndarray | None,
        *,
        classes: np.ndarray,
        config,
        n_repetitions_per_class: int | None,
        selected_offsets_by_class: Mapping[int, Sequence[int] | np.ndarray] | None = None,
    ) -> np.ndarray:
        local_offsets = _calibrated_projection_repetition_offsets(
            classes=classes,
            config=config,
            n_repetitions_per_class=n_repetitions_per_class,
            selected_offsets_by_class=selected_offsets_by_class,
        )
        return original_target_alignment_matrix(
            features,
            labels,
            classes=classes,
            config=config,
            n_repetitions_per_class=n_repetitions_per_class,
            selected_offsets_by_class=local_offsets,
        )

    source_alignment._target_alignment_matrix = _target_alignment_matrix
    setattr(source_alignment, _PATCH_MARKER, True)


class _SourceAlignmentTargetCalibrationOffsetsPatchLoader(importlib.abc.Loader):
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


class _SourceAlignmentTargetCalibrationOffsetsPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _SourceAlignmentTargetCalibrationOffsetsPatchLoader):
            return spec
        spec.loader = _SourceAlignmentTargetCalibrationOffsetsPatchLoader(spec.loader)
        return spec


def install() -> None:
    """Install the final source-alignment target-calibration offset patch."""

    loaded = sys.modules.get(_TARGET_MODULE)
    if loaded is not None:
        _patch_source_alignment(loaded)
        return

    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        return
    finder = _SourceAlignmentTargetCalibrationOffsetsPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)

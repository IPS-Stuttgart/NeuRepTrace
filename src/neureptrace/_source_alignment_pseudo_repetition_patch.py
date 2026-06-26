"""Keep pseudo-label target calibration row order aligned for class repetition.

Pseudo-label target calibration passes a compact target-calibration matrix to the
common source-alignment target-projection machinery.  For ``class_repetition``
anchors the target projection therefore interprets calibration rows as the first
within-anchor repetitions.  Source fits must use the same deterministic first-row
selection; otherwise the source template rows and target-calibration rows can
silently refer to different repetition offsets.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
from types import ModuleType

_TARGET_MODULE = "neureptrace.decoding.source_alignment"
_PATCH_MARKER = "_neureptrace_source_alignment_pseudo_repetition_patch_installed"
_FINDER_MARKER = "_neureptrace_source_alignment_pseudo_repetition_finder"


def _patch_source_alignment(source_alignment: ModuleType) -> None:
    if getattr(source_alignment, _PATCH_MARKER, False):
        return

    original_selection = source_alignment._source_alignment_repetition_selection

    def _source_alignment_repetition_selection(config, sample_mode: str) -> str:
        if sample_mode == "class_repetition" and (
            config.target_calibrated or getattr(config, "pseudo_label_target_calibrated", False)
        ):
            return "first"
        return original_selection(config, sample_mode)

    source_alignment._source_alignment_repetition_selection = _source_alignment_repetition_selection
    setattr(source_alignment, _PATCH_MARKER, True)


def _install_config_validation() -> None:
    importlib.import_module("neureptrace._source_alignment_target_seed_patch").install()


class _SourceAlignmentPseudoRepetitionPatchLoader(importlib.abc.Loader):
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


class _SourceAlignmentPseudoRepetitionPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):  # type: ignore[override]
        if fullname != _TARGET_MODULE:
            return None

        try:
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            if self not in sys.meta_path:
                sys.meta_path.insert(0, self)

        if spec is None or spec.loader is None or isinstance(spec.loader, _SourceAlignmentPseudoRepetitionPatchLoader):
            return spec
        spec.loader = _SourceAlignmentPseudoRepetitionPatchLoader(spec.loader)
        return spec


def install() -> None:
    module = sys.modules.get(_TARGET_MODULE)
    if module is not None:
        _patch_source_alignment(module)
        _install_config_validation()
        return
    if any(getattr(finder, _FINDER_MARKER, False) for finder in sys.meta_path):
        _install_config_validation()
        return
    finder = _SourceAlignmentPseudoRepetitionPatchFinder()
    setattr(finder, _FINDER_MARKER, True)
    sys.meta_path.insert(0, finder)
    _install_config_validation()


__all__ = ["install"]

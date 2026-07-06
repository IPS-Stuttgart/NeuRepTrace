"""Compatibility wrapper that installs source-helper runtime patches."""

from __future__ import annotations

import importlib
from pathlib import Path

_ORIGINAL_PATH = Path(__file__).resolve().parent.parent / "_source_gaussian_mahalanobis_numpy_scalar_patch.py"
exec(compile(_ORIGINAL_PATH.read_text(encoding="utf-8"), str(_ORIGINAL_PATH), "exec"), globals())
_ORIGINAL_INSTALL = install


def install() -> None:
    """Install source-helper patches, including source interpolation input handling."""

    importlib.import_module("neureptrace._source_interpolation_one_pass_patch").install()
    _ORIGINAL_INSTALL()


__all__ = ["install"]

from __future__ import annotations

import sys

from neureptrace import _bushmeg_source_loso_ensemble_numeric_patch as patch


def test_source_loso_ensemble_numeric_patch_is_installed_on_package_import() -> None:
    target = sys.modules.get("neureptrace.bushmeg_source_loso_ensemble")
    target_is_patched = bool(target is not None and getattr(target, patch._PATCH_MARKER, False))
    import_hook_is_installed = any(getattr(finder, patch._FINDER_MARKER, False) for finder in sys.meta_path)

    assert target_is_patched or import_hook_is_installed

"""Compatibility shim for the current source-free soft-prototype patch.

The singular module name is kept for historical imports.  The maintained
implementation lives in :mod:`neureptrace._source_free_soft_prototypes_patch`.
Delegating here prevents an explicit legacy ``install()`` call from wrapping and
shadowing the newer runtime patch after package initialization.
"""

from __future__ import annotations

from ._source_free_soft_prototypes_patch import _prototype_estimator_mode, install

__all__ = ["install", "_prototype_estimator_mode"]

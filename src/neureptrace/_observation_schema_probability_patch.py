"""Compatibility shim for probability-observation validation.

Strict probability-domain checks now live directly in
``neureptrace.observation_schema``.  The install hook remains so older package
initialization paths can keep calling it without applying a second validator
wrapper and duplicating diagnostics.
"""

from __future__ import annotations


_PATCH_MARKER = "_neureptrace_observation_probability_patch_installed"


def install() -> None:
    """Mark the legacy probability-domain patch as installed without wrapping validation."""

    from neureptrace import observation_schema

    setattr(observation_schema, _PATCH_MARKER, True)

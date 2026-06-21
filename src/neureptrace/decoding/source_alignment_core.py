"""Compatibility shim for canonical source-alignment helpers.

This module previously duplicated :mod:`neureptrace.decoding.source_alignment`.
That duplicate drifted from the canonical implementation and could report stale
benchmark-validity metadata for newer target-projection protocols.  Keep this
legacy import path as a pure re-export so all callers observe exactly the same
alignment behavior and provenance flags.
"""

from neureptrace.decoding.source_alignment import *  # noqa: F403

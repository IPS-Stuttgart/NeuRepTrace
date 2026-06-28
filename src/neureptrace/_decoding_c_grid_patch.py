"""Runtime compatibility patch for decoder option/numeric validation.

This module keeps the public ``neureptrace.decoding`` API stable while adding
strict validation for typed decoder options, non-finite regularization grids, and
classifier-parameter values. It can be removed once the overridden functions are
folded directly into ``neureptrace.decoding``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

_PATCH_MARKER = "_neureptrace_c_grid_patch_installed"


def install() -> None:
    """Install stricter validation for decoder configuration parameters."""

    from neureptrace import _decoding_classifier_param_patch, _decoding_option_type_validation_patch, decoding

    _decoding_option_type_validation_patch.install()
    _decoding_classifier_param_patch.install()

    if getattr(decoding, _PATCH_MARKER, False):
        return

    original_parse_c_grid = decoding.parse_c_grid

    def parse_c_grid(values: Sequence[float] | str | None) -> tuple[float, ...]:
        grid = tuple(float(value) for value in original_parse_c_grid(values))
        if any(not np.isfinite(value) or value <= 0.0 for value in grid):
            raise ValueError("All C values must be positive finite numbers.")
        return grid

    parse_c_grid.__doc__ = original_parse_c_grid.__doc__
    decoding.parse_c_grid = parse_c_grid
    setattr(decoding, _PATCH_MARKER, True)

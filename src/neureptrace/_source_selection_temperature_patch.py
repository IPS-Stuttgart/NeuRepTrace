"""Validation guard for source-domain selection temperature controls."""

from __future__ import annotations

import numpy as np

from . import _source_numpy_string_alias_config_patch

_PATCH_ATTR = "_neureptrace_rejects_boolean_source_selection_temperature"


def install() -> None:
    """Reject boolean softmax temperatures and install source config alias guards."""

    _source_numpy_string_alias_config_patch.install()

    from neureptrace.decoding import source_selection

    original = source_selection._resolve_temperature
    if getattr(original, _PATCH_ATTR, False):
        return

    def _resolve_temperature_checked(distance_gaps: np.ndarray, temperature: float | str) -> float:
        if isinstance(temperature, (bool, np.bool_)):
            raise ValueError("softmax_temperature must be a positive finite value or 'auto', not a boolean.")
        return original(distance_gaps, temperature)

    setattr(_resolve_temperature_checked, _PATCH_ATTR, True)
    _resolve_temperature_checked.__wrapped__ = original
    source_selection._resolve_temperature = _resolve_temperature_checked

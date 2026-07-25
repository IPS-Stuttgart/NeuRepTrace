"""Reject textual values passed as complete frequency-band specifications."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

_PATCH_MARKER = "_neureptrace_signal_band_text_validation_patch_installed"


def install() -> None:
    """Patch band validation so text is not unpacked character by character."""

    band = importlib.import_module("neureptrace.signal.band")
    if getattr(band.validate_band_hz, _PATCH_MARKER, False):
        return

    original_validate_band_hz = band.validate_band_hz

    @wraps(original_validate_band_hz)
    def validate_band_hz(band_hz: Any, sampling_rate: Any) -> tuple[float, float]:
        if isinstance(band_hz, (str, bytes, bytearray)):
            raise ValueError(
                "band_hz must contain exactly two cutoff frequencies, not a textual value."
            )
        return original_validate_band_hz(band_hz, sampling_rate)

    setattr(validate_band_hz, _PATCH_MARKER, True)
    band.validate_band_hz = validate_band_hz


__all__ = ["install"]

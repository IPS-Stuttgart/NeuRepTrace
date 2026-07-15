"""Bound matched-filter template offsets to the requested time window."""

from __future__ import annotations

import numpy as np

_PATCH_MARKER = "_neureptrace_matched_filter_template_offsets_patch_installed"


def _bounded_template_offsets(template_window: tuple[float, float], template_step: float) -> np.ndarray:
    """Return an inclusive offset grid without stepping beyond the window stop."""

    start, stop = map(float, template_window)
    step = float(template_step)
    if not np.isfinite(start) or not np.isfinite(stop):
        raise ValueError("template_window bounds must be finite.")
    if stop < start:
        raise ValueError("template_window stop must be greater than or equal to start.")
    if not np.isfinite(step) or step <= 0:
        raise ValueError("template_step must be positive and finite.")

    ratio = (stop - start) / step
    n_steps = int(np.floor(np.nextafter(ratio, np.inf)))
    offsets = start + np.arange(n_steps + 1, dtype=float) * step
    return offsets[offsets <= np.nextafter(stop, np.inf)]


def install() -> None:
    """Install the bounded offset-grid implementation on the public module."""

    from neureptrace import matched_filter_detection

    if getattr(matched_filter_detection, _PATCH_MARKER, False):
        return
    matched_filter_detection._template_offsets = _bounded_template_offsets  # noqa: SLF001
    setattr(matched_filter_detection, _PATCH_MARKER, True)


__all__ = ["install"]

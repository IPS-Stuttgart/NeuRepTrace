"""Validate the logarithmic power floor used by oscillatory summaries."""

from __future__ import annotations

import importlib
from functools import wraps
from typing import Any

import numpy as np

_PATCH_MARKER = "_neureptrace_oscillatory_log_power_eps_patch_installed"
_EPS_ERROR = "eps must be a positive finite real scalar"


def _normalize_eps(eps: Any) -> float:
    """Return a validated positive finite real scalar."""

    if isinstance(eps, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(_EPS_ERROR)
    if isinstance(eps, np.ndarray):
        if eps.ndim != 0:
            raise ValueError(_EPS_ERROR)
        eps = eps.item()
        if isinstance(eps, (bool, np.bool_, complex, np.complexfloating)):
            raise ValueError(_EPS_ERROR)
    try:
        numeric = float(eps)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_EPS_ERROR) from exc
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(_EPS_ERROR)
    return numeric


def install() -> None:
    """Patch ``summarize_analytic_window`` to reject invalid log floors."""

    oscillatory = importlib.import_module("neureptrace.features.oscillatory")
    if getattr(oscillatory, _PATCH_MARKER, False):
        return

    original_summarize = oscillatory.summarize_analytic_window

    @wraps(original_summarize)
    def summarize_analytic_window(
        analytic_window: Any,
        *,
        outputs: Any = oscillatory.DEFAULT_BAND_FEATURE_OUTPUTS,
        eps: Any = 1e-12,
    ) -> dict[str, float]:
        output_set = tuple(outputs)
        if "log_power" in output_set:
            eps = _normalize_eps(eps)
        return original_summarize(
            analytic_window,
            outputs=output_set,
            eps=eps,
        )

    oscillatory.summarize_analytic_window = summarize_analytic_window
    setattr(oscillatory, _PATCH_MARKER, True)


__all__ = ["install"]

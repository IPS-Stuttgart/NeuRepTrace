"""Match stimulus annotations independently and validate match tolerances."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from functools import wraps

import numpy as np
import pandas as pd

_PUBLIC_MODULE = f"{__package__}._stimulus_detection_public"
_MATCH_NAME = "match_stimulus_annotations"
_PATCH_MARKER = "_nrt_duplicate_event_index_matching_installed"


def _normalize_match_tolerance(value: object) -> float:
    message = "match_tolerance must be a non-negative finite number."
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    try:
        tolerance = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(message)
    return tolerance


def install() -> None:
    public_module = importlib.import_module(_PUBLIC_MODULE)
    if public_module.__dict__.get(_PATCH_MARKER, False):
        return

    original_match = public_module.__dict__[_MATCH_NAME]

    @wraps(original_match)
    def match_stimulus_annotations(
        events: pd.DataFrame,
        annotations: pd.DataFrame,
        *,
        stream_columns: Sequence[str] | None = None,
        match_tolerance: float = 0.1,
        require_class_match: bool = True,
    ) -> pd.DataFrame:
        tolerance = _normalize_match_tolerance(match_tolerance)
        original_index = events.index.copy()
        matched = original_match(
            events.reset_index(drop=True),
            annotations,
            stream_columns=stream_columns,
            match_tolerance=tolerance,
            require_class_match=require_class_match,
        )
        matched.index = original_index
        return matched

    public_module.__dict__[_MATCH_NAME] = match_stimulus_annotations
    public_module.__dict__[_PATCH_MARKER] = True


__all__ = ["install"]

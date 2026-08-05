"""Reject ambiguous ds004276 event-to-behavior alignment."""

from __future__ import annotations

from functools import wraps

import pandas as pd

_PATCH_MARKER = "_neureptrace_openneuro_word_event_validation_patch_installed"
_WORD_EVENT_TYPES = frozenset({"item", "item_post_probe"})


def install() -> None:
    """Require explicit ds004276 auditory-word event rows."""

    import neureptrace.openneuro_meg as openneuro_meg

    original_sound_events = openneuro_meg._ds004276_sound_events
    if getattr(original_sound_events, _PATCH_MARKER, False):
        return

    @wraps(original_sound_events)
    def _ds004276_sound_events(events: pd.DataFrame) -> pd.DataFrame:
        if "trial_type" not in events.columns:
            raise ValueError(
                "ds004276 events must contain a 'trial_type' column so auditory word rows can be aligned "
                "with the behavior table."
            )

        trial_type = events["trial_type"].astype("string").str.strip()
        word_rows = trial_type.isin(_WORD_EVENT_TYPES)
        if not bool(word_rows.any()):
            raise ValueError(
                "ds004276 events contain no recognized auditory word rows; expected trial_type 'item' "
                "or 'item_post_probe'."
            )

        selected = events.loc[word_rows].copy()
        selected["trial_type"] = trial_type.loc[word_rows].astype(str)
        return selected.reset_index(drop=True)

    setattr(_ds004276_sound_events, _PATCH_MARKER, True)
    openneuro_meg._ds004276_sound_events = _ds004276_sound_events


__all__ = ["install"]

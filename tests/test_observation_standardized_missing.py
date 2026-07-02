from __future__ import annotations

import pandas as pd

from neureptrace.observations import ProbabilityObservationTable


def test_standardized_treats_whitespace_only_values_as_missing() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.1, 0.2],
            "subject": ["source", "   "],
            "backend": ["sklearn", "\t"],
            "decoder": ["base", "base"],
            "custom_group": ["kept", " \n "],
        }
    )

    standardized = ProbabilityObservationTable(frame).standardized(defaults={"subject": "unknown", "backend": "fallback", "custom_group": "default"}).frame

    assert standardized.loc[0, "subject"] == "source"
    assert standardized.loc[1, "subject"] == "unknown"
    assert standardized.loc[1, "backend"] == "fallback"
    assert standardized.loc[0, "custom_group"] == "kept"
    assert standardized.loc[1, "custom_group"] == "default"


def test_standardized_preserves_nonblank_string_values_with_surrounding_whitespace() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.1],
            "subject": [" source "],
            "backend": [" sklearn "],
        }
    )

    standardized = ProbabilityObservationTable(frame).standardized(defaults={"subject": "unknown", "backend": "fallback"}).frame

    assert standardized.loc[0, "subject"] == " source "
    assert standardized.loc[0, "backend"] == " sklearn "

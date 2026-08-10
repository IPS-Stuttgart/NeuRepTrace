from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.katja_cohort_diagnostic import (
    participant_number,
    summarize_cohort_shift,
)


def test_participant_number() -> None:
    assert participant_number("s05") == 5
    assert participant_number("subject_28") == 28


def test_summarize_cohort_shift_preserves_paired_targets() -> None:
    frame = pd.DataFrame(
        {
            "configuration": [
                "single",
                "hybrid",
                "single",
                "hybrid",
            ],
            "target": ["s16", "s16", "s28", "s28"],
            "k": [20, 20, 20, 20],
            "independent_accuracy": [0.40, 0.50, 0.55, 0.61],
        }
    )
    targets, summary = summarize_cohort_shift(
        frame,
        reference_configuration="single",
        candidate_configuration="hybrid",
    )
    assert targets.set_index("target").loc[
        "s16", "candidate_minus_reference"
    ] == pytest.approx(0.10)
    assert targets.set_index("target").loc[
        "s28", "candidate_minus_reference"
    ] == pytest.approx(0.06)
    assert set(summary["cohort"]) == {
        "early_s05_to_s18",
        "late_s20_to_s28",
    }

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.katja_execution_time import add_execution_time_reference


def test_add_execution_time_reference_retains_both_clocks():
    timing = pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "trial_id": [1, 1],
            "behavior_time_seconds": [1.5, 2.5],
            "trigger_time_seconds": [1.53, np.nan],
            "expected_trigger_time_seconds": [1.53, 2.53],
            "recommended_time_seconds": [1.53, 2.53],
        }
    )
    cue = pd.DataFrame(
        {"subject": ["s1"], "trial_id": [1], "cue_duration_seconds": [0.5]}
    )
    enriched, metadata = add_execution_time_reference(timing, cue)
    np.testing.assert_allclose(
        enriched["recommended_time_execution_seconds"], [1.03, 2.03]
    )
    np.testing.assert_allclose(
        enriched["behavior_time_execution_seconds"], [1.0, 2.0]
    )
    assert metadata["recommended_execution_column"] == (
        "recommended_time_execution_seconds"
    )


def test_add_execution_time_reference_rejects_missing_trial_duration():
    timing = pd.DataFrame(
        {
            "subject": ["s1"],
            "trial_id": [2],
            "behavior_time_seconds": [1.0],
            "trigger_time_seconds": [1.03],
            "recommended_time_seconds": [1.03],
        }
    )
    cue = pd.DataFrame(
        {"subject": ["s1"], "trial_id": [1], "cue_duration_seconds": [0.5]}
    )
    with pytest.raises(ValueError, match="unavailable"):
        add_execution_time_reference(timing, cue)

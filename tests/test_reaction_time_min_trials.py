from __future__ import annotations

import math

import numpy as np
import pytest

from neureptrace.behavior.reaction_time import analyze_metric_reaction_times


@pytest.mark.parametrize("min_trials", [False, True, 0, -1, 1.5, math.nan, "3", None])
def test_reaction_time_analysis_rejects_invalid_min_trials(min_trials: object):
    with pytest.raises(ValueError, match="min_trials must be a positive integer"):
        analyze_metric_reaction_times([], metrics=("metric",), min_trials=min_trials)


def test_reaction_time_analysis_accepts_numpy_integer_min_trials():
    summary = analyze_metric_reaction_times([], metrics=("metric",), min_trials=np.int64(1))

    assert len(summary) == 1
    assert summary[0]["scope"] == "pooled_within_participant"
    assert summary[0]["metric"] == "metric"
    assert summary[0]["n_trials"] == 0

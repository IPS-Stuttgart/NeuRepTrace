from __future__ import annotations

import pandas as pd

from neureptrace.openneuro_meg import _filter_metadata, _limit_metadata_per_label


def test_zero_event_cap_disables_openneuro_per_label_limit():
    metadata = pd.DataFrame(
        {
            "onset": [0.1, 0.2, 0.3, 0.4],
            "condition": ["a", "a", "b", "b"],
        }
    )

    filtered = _filter_metadata(
        metadata,
        label_column="condition",
        include_labels=None,
        max_events_per_label=0,
        selection="random",
        seed=13,
    )

    assert filtered["condition"].tolist() == ["a", "a", "b", "b"]


def test_positive_event_cap_still_limits_each_openneuro_label():
    metadata = pd.DataFrame(
        {
            "onset": [0.1, 0.2, 0.3, 0.4],
            "condition": ["a", "a", "b", "b"],
        }
    )

    filtered = _limit_metadata_per_label(
        metadata,
        label_column="condition",
        max_events_per_label=1,
        selection="first",
        seed=13,
    )

    assert filtered["condition"].tolist() == ["a", "b"]

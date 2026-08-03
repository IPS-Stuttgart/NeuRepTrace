from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.continuous_stimulus_scan import label_event_table


def test_label_event_table_strips_surrounding_label_whitespace() -> None:
    events = pd.DataFrame(
        {
            "onset": [1.0, 2.0],
            "stimulus_class": ["  target", "baseline\t"],
        }
    )

    labeled = label_event_table(events)

    assert labeled["stimulus_class"].tolist() == ["target", "baseline"]


@pytest.mark.parametrize("blank_label", ["", " ", "\t", "\n"])
def test_label_event_table_rejects_blank_labels(blank_label: str) -> None:
    events = pd.DataFrame(
        {
            "onset": [1.0],
            "stimulus_class": [blank_label],
        }
    )

    with pytest.raises(ValueError, match="Event labels .* must not be blank"):
        label_event_table(events)


@pytest.mark.parametrize("onset", [np.nan, np.inf, -np.inf])
def test_label_event_table_rejects_nonfinite_onsets(onset: float) -> None:
    events = pd.DataFrame(
        {
            "onset": [onset],
            "stimulus_class": ["target"],
        }
    )

    with pytest.raises(ValueError, match="Event onset values .* must be finite"):
        label_event_table(events)

from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.matched_filter_detection import fit_stimulus_event_templates


def _observations() -> pd.DataFrame:
    rows = []
    event_profile = {0.0: 0.20, 0.2: 0.40, 0.4: 0.10}
    for time in np.round(np.arange(-0.2, 0.61, 0.1), 1):
        class_a = 0.25 + event_profile.get(float(time), 0.0)
        rows.append(
            {
                "subject": "sub-01",
                "stream_id": "train",
                "decoder": "logistic",
                "emission_mode": "calibrated",
                "time": float(time),
                "class_0": "A",
                "class_1": "B",
                "prob_class_0": class_a,
                "prob_class_1": 1.0 - class_a,
            }
        )
    return pd.DataFrame(rows)


def test_template_offsets_do_not_exceed_nondivisible_window_stop() -> None:
    templates = fit_stimulus_event_templates(
        _observations(),
        pd.DataFrame(
            [
                {
                    "stream_id": "train",
                    "stimulus_class": "A",
                    "onset_time": 0.0,
                }
            ]
        ),
        template_window=(0.0, 0.35),
        template_step=0.2,
        target_classes=("A",),
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    assert templates["template_time"].tolist() == [0.0, 0.2]
    assert templates["template_time"].max() <= 0.35

from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.matched_filter_detection import fit_stimulus_event_templates


def test_fit_templates_matches_integral_stream_ids_after_float_upcast():
    rows = []
    bump = {0.0: 0.20, 0.1: 0.42, 0.2: 0.48, 0.3: 0.28}
    for time in np.round(np.arange(-0.5, 0.51, 0.1), 1):
        class_a = 0.25 + bump.get(round(float(time), 1), 0.0)
        rows.append(
            {
                "subject": "sub-01",
                "stream_id": 1,
                "decoder": "logistic",
                "emission_mode": "calibrated",
                "time": float(time),
                "class_0": "A",
                "class_1": "B",
                "prob_class_0": class_a,
                "prob_class_1": 1.0 - class_a,
            }
        )
    observations = pd.DataFrame(rows)
    annotations = pd.DataFrame([{"stream_id": 1.0, "stimulus_class": "A", "onset_time": 0.0}])

    templates = fit_stimulus_event_templates(
        observations,
        annotations,
        template_window=(0.0, 0.3),
        template_step=0.1,
        target_classes=["A"],
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    assert not templates.empty
    assert templates["n_template_events"].eq(1).all()

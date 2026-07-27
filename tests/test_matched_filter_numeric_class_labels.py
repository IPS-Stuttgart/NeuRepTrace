from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.matched_filter_detection import fit_stimulus_event_templates


def _numeric_label_observations() -> pd.DataFrame:
    rows = []
    bump = {0.0: 0.10, 0.1: 0.25, 0.2: 0.20, 0.3: 0.10}
    for time in np.round(np.arange(-0.5, 0.51, 0.1), 1):
        class_a = 0.60 + bump.get(round(float(time), 1), 0.0)
        rows.append(
            {
                "subject": "sub-01",
                "stream_id": "train",
                "decoder": "logistic",
                "emission_mode": "calibrated",
                "time": float(time),
                "predicted_label": 0.0,
                "confidence": class_a,
                "class_0": "A",
                "class_1": "B",
                "prob_class_0": class_a,
                "prob_class_1": 1.0 - class_a,
            }
        )
    return pd.DataFrame(rows)


def test_predicted_confidence_matches_integral_labels_after_float_upcast():
    observations = _numeric_label_observations()
    annotations = pd.DataFrame([{"stream_id": "train", "stimulus_class": "A", "onset_time": 0.0}])

    templates = fit_stimulus_event_templates(
        observations,
        annotations,
        template_window=(0.0, 0.3),
        template_step=0.1,
        score_mode="predicted_class_confidence",
        target_classes=[0],
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    assert not templates.empty
    assert templates["stimulus_label"].eq(0).all()
    assert templates["n_template_events"].eq(1).all()


def test_template_annotations_match_integral_labels_after_float_upcast():
    observations = _numeric_label_observations()
    annotations = pd.DataFrame([{"stream_id": "train", "stimulus_label": 0.0, "onset_time": 0.0}])

    templates = fit_stimulus_event_templates(
        observations,
        annotations,
        template_window=(0.0, 0.3),
        template_step=0.1,
        target_classes=[0],
        stream_columns=("stream_id",),
        min_template_coverage=1.0,
    )

    assert not templates.empty
    assert templates["stimulus_label"].eq(0).all()
    assert templates["n_template_events"].eq(1).all()

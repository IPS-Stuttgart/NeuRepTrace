from __future__ import annotations

import pandas as pd

from neureptrace.temporal_state_workflow import build_temporal_state_summary


def test_temporal_state_summary_selects_peak_subjects_with_duplicate_indices() -> None:
    emission_compare = pd.DataFrame(
        [
            {
                "task": "nod_animate",
                "task_label": "NOD animate/inanimate",
                "decoder": "logistic",
                "preferred_emission_mode": "calibrated",
                "delta_control_margin": 0.1,
                "calibrated_control_margin": 0.2,
                "uncalibrated_control_margin": 0.1,
                "delta_effect_minus_baseline_gain": 0.05,
                "calibrated_best_stay_probability": 0.9,
                "uncalibrated_best_stay_probability": 0.8,
            }
        ]
    )
    stage_time = pd.DataFrame(
        {
            "task": ["nod_animate", "nod_animate"],
            "decoder": ["logistic", "logistic"],
            "emission_mode": ["calibrated", "calibrated"],
            "posterior_true_class_mean": [0.9, 0.8],
            "n_subjects": [11, 9],
        },
        index=[7, 7],
    )

    summary = build_temporal_state_summary(
        emission_compare,
        stages=pd.DataFrame(),
        stage_time=stage_time,
    )

    assert summary.loc[0, "calibrated_peak_posterior_true_class"] == 0.9
    assert summary.loc[0, "calibrated_peak_n_subjects"] == 11

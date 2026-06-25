from __future__ import annotations

import pandas as pd


def category2_autoencoder_result_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.DataFrame(
        [
            {
                "outer_test_subject": "1",
                "balanced_accuracy": 1.0,
                "accuracy": 1.0,
                "top2_accuracy": 1.0,
                "top3_accuracy": 1.0,
                "log_loss": 0.1,
                "brier": 0.05,
                "ece": 0.0,
                "n_train_subjects": 2,
                "n_source_trials": 4,
                "n_target_trials": 2,
                "n_classes": 2,
                "class_names": "0|1",
                "feature_kind": "evoked_dct",
                "temporal_bins": 4,
                "window_centers": "0.184",
                "window_widths": "0.1",
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "outer_test_subject": "1",
                "trial_index": 0,
                "true_label": 0,
                "predicted_label": 0,
                "prob_class_0": 0.9,
                "prob_class_1": 0.1,
            },
            {
                "outer_test_subject": "1",
                "trial_index": 1,
                "true_label": 1,
                "predicted_label": 1,
                "prob_class_0": 0.2,
                "prob_class_1": 0.8,
            },
        ]
    )
    return summary, predictions

from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.calibration import summarize_calibration_metrics


def test_summarize_calibration_metrics_selects_best_ece_with_duplicate_indices() -> None:
    summary = pd.DataFrame(
        {
            "decoder": ["logistic", "logistic", "logistic"],
            "time": [-0.05, 0.10, 0.15],
            "accuracy_mean": [0.50, 0.57, 0.60],
            "log_loss_mean": [0.70, 0.68, 0.66],
            "brier_mean": [0.50, 0.49, 0.47],
            "ece_mean": [0.09, 0.08, 0.06],
            "n_subjects": [5, 5, 5],
        },
        index=[7, 7, 7],
    )

    result = summarize_calibration_metrics(
        summary,
        baseline_window=(-0.1, 0.0),
        effect_window=(0.1, 0.2),
    )

    assert len(result) == 1
    assert result.loc[0, "best_ece_time"] == pytest.approx(0.15)
    assert result.loc[0, "best_ece"] == pytest.approx(0.06)
    assert result.loc[0, "accuracy_at_best_ece"] == pytest.approx(0.60)

from __future__ import annotations

import pandas as pd
import pytest

from neureptrace.results import build_provenance_table


@pytest.mark.parametrize("selection_metric", ["accuracy", "log_loss"])
def test_build_provenance_table_selects_by_position_with_duplicate_index(selection_metric: str) -> None:
    summary = pd.DataFrame(
        {
            "time": [0.1, 0.2],
            "accuracy_mean": [0.6, 0.8],
            "log_loss_mean": [0.5, 0.4],
            "brier_mean": [0.3, 0.2],
            "ece_mean": [0.2, 0.1],
            "n_subjects": [2, 2],
        },
        index=[42, 42],
    )

    provenance = build_provenance_table(
        summary,
        baseline_window=(0.1, 0.1),
        effect_window=(0.2, 0.2),
        selection_metric=selection_metric,
    )

    assert len(provenance) == 1
    assert provenance.loc[0, "selected_time"] == pytest.approx(0.2)
    assert provenance.loc[0, "selected_accuracy"] == pytest.approx(0.8)
    assert provenance.loc[0, "selected_log_loss"] == pytest.approx(0.4)

from pathlib import Path

import pandas as pd
import pytest

from neureptrace.plot_time_decode import plot_time_decode_results


def test_plot_time_decode_results_rejects_duplicate_metric_selection(tmp_path: Path):
    results_csv = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "time": [0.1, 0.2],
            "accuracy": [0.7, 0.8],
        }
    ).to_csv(results_csv, index=False)

    with pytest.raises(ValueError, match=r"Metrics must be unique.*accuracy"):
        plot_time_decode_results(
            results_csv,
            out_path=tmp_path / "plot.png",
            metrics=("accuracy", "accuracy"),
        )

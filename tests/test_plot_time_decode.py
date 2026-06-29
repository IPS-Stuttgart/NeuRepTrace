from pathlib import Path

import pandas as pd

from neureptrace.plot_time_decode import _summary_from_csv, plot_time_decode_results


def test_plot_time_decode_results_writes_png(tmp_path: Path):
    results_csv = tmp_path / "results.csv"
    out_path = tmp_path / "plot.png"
    pd.DataFrame(
        {
            "time": [0.1, 0.1, 0.2, 0.2],
            "accuracy": [0.6, 0.8, 0.7, 0.9],
            "log_loss": [0.5, 0.4, 0.45, 0.35],
            "brier": [0.3, 0.2, 0.25, 0.15],
            "ece": [0.1, 0.2, 0.15, 0.25],
        }
    ).to_csv(results_csv, index=False)

    plot_time_decode_results(results_csv, out_path=out_path, metrics=("accuracy",), chance=0.5)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_summary_from_csv_keeps_grouped_sem_aligned(tmp_path: Path):
    results_csv = tmp_path / "grouped_results.csv"
    pd.DataFrame(
        {
            "decoder": ["linear", "linear", "linear", "linear", "ridge", "ridge", "ridge", "ridge"],
            "time": [0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1],
            "accuracy": [0.6, 0.8, 0.7, 0.9, 0.2, 0.4, 0.3, 0.5],
            "log_loss": [0.4, 0.2, 0.3, 0.1, 0.8, 0.6, 0.7, 0.5],
            "brier": [0.3, 0.1, 0.2, 0.0, 0.7, 0.5, 0.6, 0.4],
            "ece": [0.2, 0.0, 0.1, 0.0, 0.6, 0.4, 0.5, 0.3],
        }
    ).to_csv(results_csv, index=False)

    summary = _summary_from_csv(results_csv)

    assert len(summary) == 4
    assert not summary.duplicated(["decoder", "time"]).any()

    indexed = summary.set_index(["decoder", "time"])
    assert indexed.loc[("linear", 0.0), "accuracy_mean"] == 0.7
    assert abs(indexed.loc[("linear", 0.0), "accuracy_sem"] - 0.1) < 1e-12
    assert indexed.loc[("ridge", 0.0), "accuracy_mean"] == 0.3
    assert abs(indexed.loc[("ridge", 0.0), "accuracy_sem"] - 0.1) < 1e-12

from pathlib import Path

import pandas as pd
import pytest

from neureptrace.plot_calibration import plot_reliability_diagram


def test_plot_reliability_diagram_reports_empty_input(tmp_path: Path) -> None:
    bins_csv = tmp_path / "empty_reliability_bins.csv"
    out_path = tmp_path / "reliability.png"
    pd.DataFrame(
        columns=["time", "bin", "bin_left", "bin_right", "n_samples", "accuracy", "confidence"]
    ).to_csv(bins_csv, index=False)

    with pytest.raises(ValueError, match="No reliability-bin rows available to plot"):
        plot_reliability_diagram(bins_csv, out_path)

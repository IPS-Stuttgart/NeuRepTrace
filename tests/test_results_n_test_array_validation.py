from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.results import aggregate_time_decode_results


def _fold_result_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["s1", "s1"],
            "fold": [0, 1],
            "time": [0.1, 0.1],
            "accuracy": [0.8, 0.6],
            "log_loss": [0.4, 0.6],
            "brier": [0.2, 0.3],
            "ece": [0.1, 0.2],
            "n_test": pd.Series([2, 2], dtype=object),
        }
    )


@pytest.mark.parametrize("bad_n_test", [np.asarray(True), np.asarray(2), np.asarray("2")])
def test_aggregate_time_decode_results_rejects_array_n_test_cells(bad_n_test: object) -> None:
    results = _fold_result_frame()
    results.loc[0, "n_test"] = bad_n_test

    with pytest.raises(ValueError, match="positive integer fold sizes"):
        aggregate_time_decode_results(results)

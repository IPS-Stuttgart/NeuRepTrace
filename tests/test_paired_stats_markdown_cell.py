from __future__ import annotations

import numpy as np
import pandas as pd

import neureptrace.paired_stats as paired_stats


def test_markdown_cell_handles_vector_like_values_without_ambiguous_na_check() -> None:
    cell = paired_stats._markdown_cell(np.asarray(["alpha|beta", "line\nbreak"], dtype=object))

    assert "alpha\\|beta" in cell
    assert "line break" in cell
    assert "\n" not in cell


def test_markdown_cell_preserves_missing_scalar_as_empty_cell() -> None:
    assert paired_stats._markdown_cell(pd.NA) == ""

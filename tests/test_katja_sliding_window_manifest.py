from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.katja_sliding_window_manifest import (
    build_sliding_window_manifest,
    interval_intersections,
    make_window_grid,
)


def _timing_rows():
    return pd.DataFrame(
        {
            "subject": ["s1"] * 5,
            "trial_id": [1] * 5,
            "sequence_id": [2] * 5,
            "press_position": [1, 2, 3, 4, 5],
            "finger_code": [4, 1, 2, 3, 5],
            "recommended_time_seconds": [0.5, 1.5, 2.5, 3.5, 4.5],
            "correct_order": [True] * 5,
        }
    )


def test_start_grid_has_138_full_windows_for_julia_defaults():
    grid = make_window_grid()
    assert len(grid.starts_seconds) == 138
    assert grid.starts_seconds[0] == 0.0
    np.testing.assert_allclose(grid.starts_seconds[-1], 5.48)
    np.testing.assert_allclose(grid.stops_seconds[-1], 5.98)


def test_interval_intersections_are_exact():
    result = interval_intersections([0.0, 0.4], [0.5, 0.9], [0.1], [0.6])
    np.testing.assert_allclose(result[:, 0], [0.4, 0.2])


def test_manifest_keeps_raw_overlaps_without_labels():
    manifest, metadata = build_sliding_window_manifest(_timing_rows())
    assert manifest.shape[0] == 138
    assert "true_label" not in manifest
    assert metadata["defines_null_labels"] is False
    first = manifest.iloc[0]
    # First press interval is [0.1, 0.6], first decoding window [0, 0.5].
    assert np.isclose(first["press_1_intersection_seconds"], 0.4)
    assert first["max_overlap_press_position"] == 1

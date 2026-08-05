from __future__ import annotations

import numpy as np

from neureptrace._katja_finger_sequence_support import (
    DEFAULT_PARTICIPANTS,
    _stable_source_selection,
    katja_nested_trial_calibration_indices,
)


def test_recovered_s05_source_subset_matches_original_comparison() -> None:
    selected = _stable_source_selection(
        DEFAULT_PARTICIPANTS,
        target="s05",
        n_sources=9,
        seed=2026,
    )

    assert selected == (
        "s06",
        "s10",
        "s11",
        "s13",
        "s17",
        "s18",
        "s20",
        "s21",
        "s25",
    )


def test_recovered_nested_split_uses_one_sequential_rng() -> None:
    strata = np.repeat(np.asarray([0, 1, 2, 3]), 8)

    calibration, evaluation, pool = katja_nested_trial_calibration_indices(
        strata,
        (1, 3, 5),
        seed=13,
    )

    np.testing.assert_array_equal(
        calibration[1],
        np.asarray([0, 14, 19, 29]),
    )
    np.testing.assert_array_equal(
        calibration[3],
        np.asarray([0, 3, 7, 9, 13, 14, 19, 20, 23, 29, 30, 31]),
    )
    np.testing.assert_array_equal(
        calibration[5],
        np.asarray(
            [
                0,
                2,
                3,
                5,
                7,
                8,
                9,
                12,
                13,
                14,
                16,
                19,
                20,
                22,
                23,
                27,
                28,
                29,
                30,
                31,
            ]
        ),
    )
    np.testing.assert_array_equal(
        evaluation,
        np.asarray([1, 4, 6, 10, 11, 15, 17, 18, 21, 24, 25, 26]),
    )
    np.testing.assert_array_equal(pool, calibration[5])
    assert set(calibration[1]).issubset(set(calibration[3]))
    assert set(calibration[3]).issubset(set(calibration[5]))
    assert not np.intersect1d(pool, evaluation).size

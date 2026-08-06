from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.decoding.prior_shift import adapt_probability_blocks_for_prior_shift, prior_from_labels


def test_prior_from_labels_counts_explicit_nan_class() -> None:
    labels = ["seen", np.nan, "seen", float("nan")]

    prior, classes = prior_from_labels(labels, classes=["seen", np.nan])

    assert classes[0] == "seen"
    assert np.isnan(classes[1])
    np.testing.assert_allclose(prior, [0.5, 0.5])


def test_prior_from_labels_collapses_repeated_missing_labels() -> None:
    prior, classes = prior_from_labels([pd.NA, pd.NA, "seen"])

    assert len(classes) == 2
    assert pd.isna(classes[0])
    assert classes[1] == "seen"
    np.testing.assert_allclose(prior, [2.0 / 3.0, 1.0 / 3.0])


def test_prior_from_labels_matches_missing_values_inside_composite_labels() -> None:
    labels = [(pd.NA, "cue"), ("seen", "cue"), (pd.NA, "cue")]
    classes = [(pd.NA, "cue"), ("seen", "cue")]

    prior, class_order = prior_from_labels(labels, classes=classes)

    assert pd.isna(class_order[0][0])
    assert class_order[0][1] == "cue"
    assert class_order[1] == ("seen", "cue")
    np.testing.assert_allclose(prior, [2.0 / 3.0, 1.0 / 3.0])


def test_prior_from_labels_keeps_none_distinct_from_missing() -> None:
    prior, classes = prior_from_labels([None, pd.NA, None], classes=[None, pd.NA])

    assert classes[0] is None
    assert pd.isna(classes[1])
    np.testing.assert_allclose(prior, [2.0 / 3.0, 1.0 / 3.0])


def test_prior_from_labels_keeps_missing_sentinel_kinds_distinct() -> None:
    prior, classes = prior_from_labels(
        [pd.NA, pd.NA, np.nan, "seen"],
        classes=[pd.NA, np.nan, "seen"],
    )

    assert classes[0] is pd.NA
    assert isinstance(classes[1], float) and np.isnan(classes[1])
    assert classes[2] == "seen"
    np.testing.assert_allclose(prior, [0.5, 0.25, 0.25])


def test_blockwise_prior_shift_groups_nan_block_ids() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        dtype=float,
    )

    result = adapt_probability_blocks_for_prior_shift(
        probabilities,
        [np.nan, float("nan"), "other", "other"],
        source_prior=[0.5, 0.5],
        min_block_rows=2,
        smoothing=0.0,
    )

    assert len(result.block_results) == 2
    missing_key = next(key for key in result.block_results if isinstance(key, float) and np.isnan(key))
    assert result.block_results[missing_key].target_prior[0] > result.block_results[missing_key].target_prior[1]
    assert result.block_results["other"].target_prior[1] > result.block_results["other"].target_prior[0]


def test_blockwise_prior_shift_keeps_missing_sentinel_kinds_distinct() -> None:
    probabilities = np.asarray(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.1, 0.9],
            [0.2, 0.8],
        ],
        dtype=float,
    )

    result = adapt_probability_blocks_for_prior_shift(
        probabilities,
        [pd.NA, pd.NA, np.nan, float("nan")],
        source_prior=[0.5, 0.5],
        min_block_rows=2,
        smoothing=0.0,
    )

    assert len(result.block_results) == 2
    pandas_missing_key = next(key for key in result.block_results if key is pd.NA)
    numpy_missing_key = next(key for key in result.block_results if isinstance(key, float) and np.isnan(key))
    assert result.block_results[pandas_missing_key].target_prior[0] > result.block_results[pandas_missing_key].target_prior[1]
    assert result.block_results[numpy_missing_key].target_prior[1] > result.block_results[numpy_missing_key].target_prior[0]

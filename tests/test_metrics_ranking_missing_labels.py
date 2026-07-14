from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.metrics.ranking import rank_class_scores


def test_rank_class_scores_matches_pandas_missing_labels() -> None:
    result = rank_class_scores(
        [[0.9, 0.1], [0.2, 0.8]],
        [pd.NA, "known"],
        [pd.NA, "known"],
        top_k=(1,),
        row_top_k=1,
    )

    np.testing.assert_array_equal(result["true_label_ranks"], np.asarray([1.0, 1.0]))
    assert result["top_k_accuracy"] == {1: pytest.approx(1.0)}
    assert result["rows"][0]["true_label_score"] == pytest.approx(0.9)


def test_rank_class_scores_keeps_none_and_numpy_nat_classes_distinct() -> None:
    nat_label = np.datetime64("NaT")

    result = rank_class_scores(
        [[0.1, 0.9], [0.8, 0.2]],
        [None, nat_label],
        [nat_label, None],
        top_k=(1,),
        row_top_k=1,
    )

    np.testing.assert_array_equal(result["true_label_ranks"], np.asarray([1.0, 1.0]))
    assert result["top_k_accuracy"] == {1: pytest.approx(1.0)}
    assert isinstance(result["rows"][0]["rank1_class"], np.datetime64)
    assert np.isnat(result["rows"][0]["rank1_class"])
    assert result["rows"][1]["rank1_class"] is None


def test_rank_class_scores_preserves_numpy_nat_array_labels() -> None:
    classes = np.asarray(["NaT", "2020-01-01"], dtype="datetime64[D]")

    result = rank_class_scores(
        [[0.9, 0.1], [0.2, 0.8]],
        classes,
        classes.copy(),
        top_k=(1,),
        row_top_k=1,
    )

    np.testing.assert_array_equal(result["true_label_ranks"], np.asarray([1.0, 1.0]))
    assert isinstance(result["rows"][0]["rank1_class"], np.datetime64)
    assert np.isnat(result["rows"][0]["rank1_class"])

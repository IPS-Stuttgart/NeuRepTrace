from __future__ import annotations

import pytest

from neureptrace.metrics import rank_class_scores

_HUGE_INTEGER = 10**400


def test_rank_class_scores_accepts_large_top_k() -> None:
    result = rank_class_scores(
        [[0.2, 0.8]],
        ["left", "right"],
        ["right"],
        top_k=(_HUGE_INTEGER,),
    )

    assert result["top_k_accuracy"][_HUGE_INTEGER] == pytest.approx(1.0)


def test_rank_class_scores_caps_large_row_top_k_at_class_count() -> None:
    result = rank_class_scores(
        [[0.2, 0.8]],
        ["left", "right"],
        ["right"],
        top_k=(1,),
        row_top_k=_HUGE_INTEGER,
    )

    assert result["rows"][0] == {
        "true_label_rank": 1.0,
        "true_label_score": 0.8,
        "rank1_class": "right",
        "rank1_score": 0.8,
        "rank2_class": "left",
        "rank2_score": 0.2,
    }

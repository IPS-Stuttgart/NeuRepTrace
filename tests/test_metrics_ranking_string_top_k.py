from __future__ import annotations

import pytest

from neureptrace.metrics import rank_class_scores


@pytest.mark.parametrize("top_k", ["23", b"23", bytearray(b"23"), memoryview(b"23")])
def test_rank_class_scores_rejects_string_like_top_k_sequences(top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k must be a sequence of integers"):
        rank_class_scores(
            [[0.8, 0.2], [0.3, 0.7]],
            ["left", "right"],
            ["left", "right"],
            top_k=top_k,
        )


def test_rank_class_scores_keeps_numeric_string_entries_supported() -> None:
    result = rank_class_scores(
        [[0.8, 0.2], [0.3, 0.7]],
        ["left", "right"],
        ["left", "right"],
        top_k=("1", "2"),
    )

    assert result["top_k_accuracy"] == {1: 1.0, 2: 1.0}

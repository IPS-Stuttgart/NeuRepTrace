from __future__ import annotations

import numpy as np
import pytest

from neureptrace._katja_finger_sequence_support import load_katja_feature_cache


def _write_cache(tmp_path, correct_order: np.ndarray):
    path = tmp_path / "katja-cache.npz"
    np.savez(
        path,
        features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        subjects=np.asarray(["s05", "s05"], dtype=object),
        trial_ids=np.asarray([1, 1]),
        press_positions=np.asarray([2, 3]),
        sequence_ids=np.asarray([10, 10]),
        labels=np.asarray([0, 1]),
        correct_order=correct_order,
    )
    return path


@pytest.mark.parametrize(
    "correct_order",
    [
        np.asarray(["False", "True"], dtype=object),
        np.asarray([0, 2]),
        np.asarray([0.0, np.nan]),
        np.asarray([None, True], dtype=object),
    ],
)
def test_load_katja_feature_cache_rejects_non_boolean_correct_order_values(
    tmp_path,
    correct_order: np.ndarray,
) -> None:
    path = _write_cache(tmp_path, correct_order)

    with pytest.raises(ValueError, match="correct_order.*boolean or numeric 0/1"):
        load_katja_feature_cache(path)


@pytest.mark.parametrize(
    ("correct_order", "expected"),
    [
        (np.asarray([False, True]), [False, True]),
        (np.asarray([0, 1], dtype=np.int8), [False, True]),
        (np.asarray([0.0, 1.0]), [False, True]),
        (np.asarray([np.bool_(True), np.int64(0)], dtype=object), [True, False]),
    ],
)
def test_load_katja_feature_cache_accepts_explicit_binary_correct_order_values(
    tmp_path,
    correct_order: np.ndarray,
    expected: list[bool],
) -> None:
    path = _write_cache(tmp_path, correct_order)

    cache = load_katja_feature_cache(path)

    assert cache["correct_order"].dtype == np.dtype(bool)
    assert cache["correct_order"].tolist() == expected

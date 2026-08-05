from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neureptrace._katja_finger_sequence_support import load_katja_feature_cache


def _write_cache(path: Path, press_positions: np.ndarray) -> None:
    n_rows = int(press_positions.shape[0])
    np.savez(
        path,
        features=np.ones((n_rows, 2), dtype=np.float32),
        subjects=np.asarray(["s05"] * n_rows),
        trial_ids=np.arange(n_rows),
        press_positions=press_positions,
        sequence_ids=np.zeros(n_rows, dtype=int),
        finger_codes=np.arange(n_rows),
    )


def test_katja_cache_preserves_integral_press_positions(tmp_path: Path) -> None:
    cache_path = tmp_path / "features.npz"
    positions = np.asarray([np.iinfo(np.int64).min, -1, 4, np.iinfo(np.int64).max], dtype=np.int64)
    _write_cache(cache_path, positions)

    cache = load_katja_feature_cache(cache_path)

    assert cache["press_positions"].dtype == np.int64
    np.testing.assert_array_equal(cache["press_positions"], positions)


@pytest.mark.parametrize(
    "press_positions",
    [
        pytest.param(np.asarray([1.0, 2.5, 3.0, 4.0]), id="fractional"),
        pytest.param(np.asarray([1, 2 + 1j, 3, 4]), id="complex"),
        pytest.param(np.asarray([True, False, True, False]), id="boolean"),
        pytest.param(np.asarray([1.0, np.nan, 3.0, 4.0]), id="non-finite"),
        pytest.param(np.asarray([1, complex(2, 1), 3, 4], dtype=object), id="object-complex"),
        pytest.param(np.asarray([1, 2, 3, 2**63], dtype=np.uint64), id="out-of-range"),
    ],
)
def test_katja_cache_rejects_non_integer_press_positions(
    tmp_path: Path,
    press_positions: np.ndarray,
) -> None:
    cache_path = tmp_path / "features.npz"
    _write_cache(cache_path, press_positions)

    with pytest.raises(ValueError, match="press_positions.*finite integer"):
        load_katja_feature_cache(cache_path)

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from neureptrace.katja_finger_sequence_benchmark import load_katja_feature_cache

if TYPE_CHECKING:
    from pathlib import Path


def _write_cache(path: Path, features: np.ndarray) -> None:
    np.savez(
        path,
        features=features,
        subjects=np.array(["s1"] * 4),
        trial_ids=np.array([1, 1, 1, 1]),
        press_positions=np.arange(2, 6),
        sequence_ids=np.zeros(4, dtype=int),
        labels=np.arange(4),
    )


@pytest.mark.parametrize(
    "features",
    [
        np.array([[1.0 + 2.0j, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.complex64),
        np.array([[1.0 + 2.0j, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=object),
    ],
    ids=("complex-dtype", "object-complex"),
)
def test_load_katja_feature_cache_rejects_complex_features(tmp_path: Path, features: np.ndarray):
    path = tmp_path / "cache.npz"
    _write_cache(path, features)

    with pytest.raises(ValueError, match="real-valued features"):
        load_katja_feature_cache(path)


def test_load_katja_feature_cache_accepts_real_object_features(tmp_path: Path):
    path = tmp_path / "cache.npz"
    features = np.array([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]], dtype=object)
    _write_cache(path, features)

    cache = load_katja_feature_cache(path)

    assert cache["features"].dtype == np.float32
    np.testing.assert_array_equal(cache["features"], np.asarray(features, dtype=np.float32))

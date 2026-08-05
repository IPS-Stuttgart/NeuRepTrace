from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

import neureptrace.katja_finger_sequence_benchmark as benchmark
from neureptrace.katja_finger_sequence_benchmark import (
    derive_participant_local_finger_labels,
    load_katja_feature_cache,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_load_cache_flattens_feature_tail_and_accepts_physical_codes(tmp_path: Path):
    path = tmp_path / "cache.npz"
    np.savez(
        path,
        features=np.arange(48, dtype=float).reshape(8, 2, 3),
        subjects=np.array(["s1"] * 4 + ["s2"] * 4),
        trial_ids=np.array([1, 1, 1, 1, 2, 2, 2, 2]),
        press_positions=np.tile(np.arange(2, 6), 2),
        sequence_ids=np.array([0] * 4 + [1] * 4),
        finger_codes=np.tile(np.array([5, 2, 9, 7]), 2),
    )

    cache = load_katja_feature_cache(path)

    assert cache["features"].shape == (8, 6)
    assert cache["correct_order"].all()
    assert "finger_codes" in cache


def test_derive_participant_local_labels_uses_sorted_codes_per_subject():
    subjects = np.array(["s1"] * 4 + ["s2"] * 4)
    codes = np.array([20, 10, 40, 30, 4, 1, 3, 2])
    labels = derive_participant_local_finger_labels(
        subjects,
        codes,
        included_mask=np.ones(8, dtype=bool),
        expected_classes=4,
    )

    np.testing.assert_array_equal(labels[:4], np.array([1, 0, 3, 2]))
    np.testing.assert_array_equal(labels[4:], np.array([3, 0, 2, 1]))


def test_benchmark_rejects_selected_sources_without_included_rows(monkeypatch: pytest.MonkeyPatch):
    cache = {
        "features": np.arange(16, dtype=float).reshape(8, 2),
        "subjects": np.array(["s1"] * 4 + ["s2"] * 4),
        "trial_ids": np.array([1] * 4 + [2] * 4),
        "press_positions": np.tile(np.arange(2, 6), 2),
        "sequence_ids": np.array([0] * 4 + [1] * 4),
        "labels": np.tile(np.arange(4), 2),
        "correct_order": np.ones(8, dtype=bool),
    }

    monkeypatch.setattr(
        benchmark,
        "_fit_source_preprocessor",
        lambda *args, **kwargs: pytest.fail("preprocessing must not run for an incomplete source set"),
    )

    with pytest.raises(ValueError, match=r"Selected source participants.*lack included rows.*s3"):
        benchmark.run_katja_finger_sequence_benchmark(
            cache,
            participants=("s1", "s2", "s3"),
            target_participants=("s1",),
            calibration_counts=(1,),
            calibration_seeds=(13,),
            source_map={"s1": ("s2", "s3")},
        )

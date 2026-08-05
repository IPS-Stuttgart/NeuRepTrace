from __future__ import annotations

import numpy as np
import pytest

from neureptrace.katja_physical_finger import (
    infer_global_physical_codes,
    participant_physical_finger_maps,
    physical_probabilities_to_local,
)


def test_participant_physical_maps_detect_local_semantic_mismatch() -> None:
    subjects = np.asarray(
        ["a"] * 4 + ["b"] * 4 + ["c"] * 4,
        dtype=object,
    )
    codes = np.asarray(
        [1, 2, 4, 5, 1, 3, 4, 5, 1, 2, 3, 5],
        dtype=int,
    )
    universe = infer_global_physical_codes(codes)
    mappings = participant_physical_finger_maps(
        subjects,
        codes,
        global_codes=universe,
    )

    assert universe == (1, 2, 3, 4, 5)
    assert mappings["a"].variable_codes == (1, 2, 4, 5)
    assert mappings["a"].fixed_code == 3
    assert mappings["b"].variable_codes == (1, 3, 4, 5)
    assert mappings["b"].fixed_code == 2
    assert mappings["c"].variable_codes == (1, 2, 3, 5)
    assert mappings["c"].fixed_code == 4
    assert mappings["a"].variable_codes[1] != mappings["b"].variable_codes[1]


def test_physical_probabilities_are_masked_and_renormalized() -> None:
    probabilities = np.asarray(
        [
            [
                [0.10, 0.15, 0.50, 0.20, 0.05],
                [0.05, 0.60, 0.10, 0.10, 0.15],
            ]
        ]
    )
    mapped = physical_probabilities_to_local(
        probabilities,
        model_classes=np.asarray([1, 2, 3, 4, 5]),
        variable_codes=(1, 2, 4, 5),
    )

    expected_first = np.asarray([0.10, 0.15, 0.20, 0.05]) / 0.50
    expected_second = np.asarray([0.05, 0.60, 0.10, 0.15]) / 0.90
    np.testing.assert_allclose(mapped[0, 0], expected_first)
    np.testing.assert_allclose(mapped[0, 1], expected_second)
    np.testing.assert_allclose(mapped.sum(axis=2), 1.0)


def test_physical_probability_mapping_rejects_missing_code() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        physical_probabilities_to_local(
            np.ones((1, 2, 4), dtype=float),
            model_classes=np.asarray([1, 2, 4, 5]),
            variable_codes=(1, 2, 3, 5),
        )

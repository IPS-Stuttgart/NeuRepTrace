from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_distance_weighting import compute_source_distance_weights


def test_class_domain_distance_weighting_preserves_composite_group_values() -> None:
    features = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.1],
            [5.0, 5.0, 2.0],
            [5.1, 5.0, 2.1],
        ],
        dtype=float,
    )
    labels = np.asarray(
        [
            ["class_a", "side_1"],
            ["class_a", "side_1"],
            ["class_b", "side_2"],
            ["class_b", "side_2"],
        ],
        dtype=object,
    )
    domains = [("subject_1", "run_1"), ("subject_1", "run_1"), ("subject_2", "run_1"), ("subject_2", "run_1")]

    result = compute_source_distance_weights(
        features,
        labels,
        source_domains=domains,
        config={"group_mode": "class_domain", "robust": False},
    )

    expected_keys = (
        (("class_a", "side_1"), ("subject_1", "run_1")),
        (("class_a", "side_1"), ("subject_1", "run_1")),
        (("class_b", "side_2"), ("subject_2", "run_1")),
        (("class_b", "side_2"), ("subject_2", "run_1")),
    )
    assert result.group_keys == expected_keys
    assert set(result.group_centers) == {
        (("class_a", "side_1"), ("subject_1", "run_1")),
        (("class_b", "side_2"), ("subject_2", "run_1")),
    }
    assert result.sample_weights.shape == (4,)

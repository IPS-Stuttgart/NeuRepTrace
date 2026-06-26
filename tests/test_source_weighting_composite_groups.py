import numpy as np
import pytest

from neureptrace.decoding.source_weighting import dynamic_source_group_weights, weights_from_scores


def test_explicit_group_matrix_is_preserved_for_score_weights():
    groups = np.asarray([["study_a", "subject_1"], ["study_b", "subject_2"], ["study_a", "subject_1"]], dtype=object)

    weights = weights_from_scores(
        {("study_a", "subject_1"): 0.90, ("study_b", "subject_2"): 0.20},
        groups=groups,
        temperature=0.10,
    )

    assert tuple(weights) == (("study_a", "subject_1"), ("study_b", "subject_2"))
    assert weights[("study_a", "subject_1")] > weights[("study_b", "subject_2")]
    assert np.mean(list(weights.values())) == pytest.approx(1.0)


def test_target_similarity_accepts_explicit_group_matrix():
    groups = np.asarray([["study_a", "subject_1"], ["study_b", "subject_2"]], dtype=object)
    source_features = {
        ("study_a", "subject_1"): np.asarray([[1.0, 0.0], [0.9, 0.1]]),
        ("study_b", "subject_2"): np.asarray([[-1.0, 0.0], [-0.9, -0.1]]),
    }
    target_features = np.asarray([[1.0, 0.0], [1.1, -0.1]])

    weights = dynamic_source_group_weights(
        config={"mode": "target_similarity", "temperature": 0.10},
        groups=groups,
        source_features=source_features,
        target_features=target_features,
    )

    assert tuple(weights) == (("study_a", "subject_1"), ("study_b", "subject_2"))
    assert weights[("study_a", "subject_1")] > weights[("study_b", "subject_2")]
    assert np.mean(list(weights.values())) == pytest.approx(1.0)

import numpy as np
import pytest

from neureptrace.decoding.source_weighting import (
    combine_source_reliability_and_similarity,
    dynamic_source_group_weights,
    normalize_source_group_weighting_mode,
    sample_weights_from_group_weights,
    selected_source_groups,
    source_group_weighting_config,
    target_similarity_scores,
    weights_from_scores,
)


def test_source_reliability_weights_rank_higher_scores_and_top_k():
    weights = weights_from_scores(
        {"s1": 0.80, "s2": 0.60, "s3": 0.40},
        metric="balanced_accuracy",
        temperature=0.10,
        top_k=2,
    )

    assert weights["s1"] > weights["s2"] > weights["s3"]
    assert weights["s3"] == pytest.approx(0.0)
    assert np.mean(list(weights.values())) == pytest.approx(1.0)
    assert selected_source_groups(weights) == ("s1", "s2")


def test_minimized_metric_treats_lower_scores_as_better():
    weights = weights_from_scores({"good": 0.20, "bad": 1.00}, metric="log_loss", temperature=0.25)

    assert weights["good"] > weights["bad"]
    assert np.mean(list(weights.values())) == pytest.approx(1.0)


def test_target_similarity_uses_unlabeled_feature_centroids():
    source_features = {
        "near": np.asarray([[1.0, 0.0], [0.9, 0.1]]),
        "far": np.asarray([[-1.0, 0.0], [-0.9, -0.1]]),
    }
    target_features = np.asarray([[1.0, 0.0], [1.1, -0.1]])

    scores = target_similarity_scores(source_features, target_features)
    weights = dynamic_source_group_weights(
        config={"mode": "target_similarity", "temperature": 0.10},
        source_features=source_features,
        target_features=target_features,
    )

    assert scores["near"] > scores["far"]
    assert weights["near"] > weights["far"]
    assert np.mean(list(weights.values())) == pytest.approx(1.0)


def test_hybrid_mode_combines_source_reliability_and_target_similarity():
    source_scores = {"reliable": 0.90, "similar": 0.50, "weak": 0.40}
    source_features = {
        "reliable": np.asarray([[-1.0, 0.0], [-0.9, 0.1]]),
        "similar": np.asarray([[1.0, 0.0], [0.9, 0.1]]),
        "weak": np.asarray([[0.0, 1.0], [0.1, 0.9]]),
    }
    target_features = np.asarray([[1.0, 0.0], [1.1, -0.1]])

    weights = dynamic_source_group_weights(
        config={"mode": "hybrid", "temperature": 0.50, "target_similarity_weight": 0.75},
        source_scores=source_scores,
        source_features=source_features,
        target_features=target_features,
    )

    assert set(weights) == {"reliable", "similar", "weak"}
    assert weights["similar"] > weights["weak"]
    assert np.mean(list(weights.values())) == pytest.approx(1.0)


def test_sample_weights_from_group_weights_are_row_level_and_mean_one():
    sample_weights = sample_weights_from_group_weights(
        np.asarray(["s1", "s1", "s2", "s3"]),
        {"s1": 2.0, "s2": 0.5, "s3": 0.0},
    )

    assert sample_weights.shape == (4,)
    assert sample_weights.mean() == pytest.approx(1.0)
    assert sample_weights[0] == sample_weights[1]
    assert sample_weights[0] > sample_weights[2] > sample_weights[3]


def test_config_metadata_marks_category_two_modes():
    cfg = source_group_weighting_config({"mode": "target_similarity", "top_k": 2, "blend": 0.75})
    metadata = cfg.metadata()

    assert cfg.protocol == "unlabeled_target_adaptive"
    assert metadata["source_group_weighting_uses_unlabeled_target_data"] is True
    assert metadata["source_group_weighting_uses_target_labels"] is False


def test_enabled_aliases_select_source_reliability_mode():
    assert normalize_source_group_weighting_mode(True) == "source_reliability"
    assert normalize_source_group_weighting_mode("true") == "source_reliability"
    assert normalize_source_group_weighting_mode("on") == "source_reliability"
    assert normalize_source_group_weighting_mode("yes") == "source_reliability"

    cfg = source_group_weighting_config({"enabled": True})

    assert cfg.mode == "source_reliability"
    assert cfg.protocol == "strict_source_only"
    assert cfg.uses_unlabeled_target_data is False


def test_numeric_enabled_aliases_toggle_source_reliability_mode():
    assert normalize_source_group_weighting_mode(1) == "source_reliability"
    assert normalize_source_group_weighting_mode("1") == "source_reliability"
    assert normalize_source_group_weighting_mode(0) == "none"
    assert normalize_source_group_weighting_mode("0") == "none"

    assert source_group_weighting_config({"enabled": 1}).mode == "source_reliability"
    assert source_group_weighting_config({"enabled": "1"}).mode == "source_reliability"
    assert source_group_weighting_config({"enabled": 0}).mode == "none"
    assert source_group_weighting_config({"enabled": "0"}).mode == "none"


@pytest.mark.parametrize("score", [True, np.bool_(True), np.asarray(True)])
def test_source_group_weighting_rejects_boolean_scores(score):
    with pytest.raises(ValueError, match="source-group scores"):
        weights_from_scores({"s1": score, "s2": 0.50})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("temperature", np.asarray(True)),
        ("temperature", np.asarray([0.25])),
        ("top_k", np.asarray(True)),
        ("top_k", np.asarray([2])),
        ("blend", np.asarray(False)),
        ("blend", np.asarray([0.5])),
        ("hybrid_target_similarity_weight", np.asarray(True)),
        ("hybrid_target_similarity_weight", np.asarray([0.5])),
    ],
)
def test_source_group_weighting_config_rejects_array_like_numeric_scalars(field, value):
    with pytest.raises(ValueError, match="source_group_weighting"):
        source_group_weighting_config({"mode": "hybrid", field: value})


def test_source_group_weighting_config_accepts_zero_dimensional_numeric_arrays():
    cfg = source_group_weighting_config(
        {
            "mode": "source_reliability",
            "temperature": np.asarray(0.25),
            "top_k": np.asarray(1),
            "blend": np.asarray(0.5),
        }
    )

    assert cfg.temperature == pytest.approx(0.25)
    assert cfg.top_k == 1
    assert cfg.blend == pytest.approx(0.5)


def test_weights_from_scores_rejects_array_score_scalars():
    with pytest.raises(ValueError, match="source-group scores"):
        weights_from_scores({"s1": np.asarray([0.80]), "s2": 0.50})


def test_combine_source_reliability_rejects_array_target_similarity_scores():
    with pytest.raises(ValueError, match="target-similarity scores"):
        combine_source_reliability_and_similarity({"s1": 0.80, "s2": 0.70}, {"s1": np.asarray([0.20]), "s2": 0.10})


def test_sample_weight_helpers_reject_array_like_weights():
    with pytest.raises(ValueError, match="source_group_weight"):
        sample_weights_from_group_weights(np.asarray(["s1"]), {"s1": np.asarray([1.0])})

    with pytest.raises(ValueError, match="source_group_weight"):
        selected_source_groups({"s1": np.asarray([1.0])})


@pytest.mark.parametrize(
    "bad_target_features",
    [
        np.asarray([[1.0, np.nan], [1.0, 0.0]]),
        np.asarray([1.0, np.inf]),
    ],
)
def test_target_similarity_rejects_nonfinite_target_features(bad_target_features):
    source_features = {"s1": np.asarray([[1.0, 0.0], [0.9, 0.1]])}

    with pytest.raises(ValueError, match="finite"):
        target_similarity_scores(source_features, bad_target_features)


@pytest.mark.parametrize(
    "bad_source_features",
    [
        np.asarray([[1.0, np.nan], [1.0, 0.0]]),
        np.asarray([np.inf, 1.0]),
    ],
)
def test_target_similarity_rejects_nonfinite_source_features(bad_source_features):
    source_features = {"bad": bad_source_features}
    target_features = np.asarray([[1.0, 0.0], [1.1, -0.1]])

    with pytest.raises(ValueError, match="finite"):
        target_similarity_scores(source_features, target_features)

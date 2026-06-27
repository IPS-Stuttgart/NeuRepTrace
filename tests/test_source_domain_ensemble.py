from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_ensemble import (
    SOURCE_DOMAIN_ENSEMBLE_CATEGORY_1,
    SOURCE_DOMAIN_ENSEMBLE_CATEGORY_2,
    fit_source_domain_probability_ensemble,
    normalize_ensemble_weighting,
)


def _toy_domains():
    source_features = np.asarray(
        [
            [-2.0, 0.0],
            [-1.6, 0.2],
            [1.7, -0.1],
            [2.1, 0.1],
            [-1.8, 3.0],
            [-1.4, 3.2],
            [1.8, 2.8],
            [2.2, 3.1],
        ],
        dtype=float,
    )
    source_labels = np.asarray(["left", "left", "right", "right", "left", "left", "right", "right"], dtype=object)
    source_domains = np.asarray(["a", "a", "a", "a", "b", "b", "b", "b"], dtype=object)
    target_features = np.asarray([[-1.7, 0.1], [1.9, 0.0]], dtype=float)
    return source_features, source_labels, source_domains, target_features


def test_uniform_source_domain_ensemble_is_protocol1() -> None:
    source_features, source_labels, source_domains, target_features = _toy_domains()

    result = fit_source_domain_probability_ensemble(
        source_features=source_features,
        source_labels=source_labels,
        source_domains=source_domains,
        target_features=target_features,
        weighting="uniform",
    )

    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)
    assert set(result.predictions.tolist()) <= {"left", "right"}
    assert result.predictions.shape == (2,)
    assert set(result.domain_weights) == {"a", "b"}
    assert np.isclose(sum(result.domain_weights.values()), 1.0)
    assert result.metadata["source_domain_ensemble_protocol_category"] == SOURCE_DOMAIN_ENSEMBLE_CATEGORY_1
    assert result.metadata["source_domain_ensemble_valid_for_strict_source_only"] is True
    assert result.metadata["source_domain_ensemble_uses_target_features_for_weighting"] is False
    assert result.metadata["source_domain_ensemble_uses_target_labels"] is False


def test_source_domain_ensemble_preserves_composite_tuple_labels() -> None:
    source_features = np.asarray(
        [
            [-2.0, 0.0],
            [-1.6, 0.2],
            [1.7, -0.1],
            [2.1, 0.1],
            [-1.8, 3.0],
            [-1.4, 3.2],
            [1.8, 2.8],
            [2.2, 3.1],
        ],
        dtype=float,
    )
    source_labels = [
        ("face", "left"),
        ("face", "left"),
        ("object", "right"),
        ("object", "right"),
        ("face", "left"),
        ("face", "left"),
        ("object", "right"),
        ("object", "right"),
    ]
    source_domains = ["a", "a", "a", "a", "b", "b", "b", "b"]
    target_features = np.asarray([[-1.7, 0.1], [1.9, 0.0]], dtype=float)

    result = fit_source_domain_probability_ensemble(
        source_features=source_features,
        source_labels=source_labels,
        source_domains=source_domains,
        target_features=target_features,
        weighting="uniform",
    )

    assert result.classes.tolist() == [("face", "left"), ("object", "right")]
    assert result.predictions.shape == (2,)
    assert set(result.predictions.tolist()) <= {("face", "left"), ("object", "right")}
    assert result.models["a"].classes.tolist() == [("face", "left"), ("object", "right")]
    assert result.metadata["source_domain_ensemble_n_classes"] == 2


def test_target_confidence_weighting_is_protocol2() -> None:
    source_features, source_labels, source_domains, target_features = _toy_domains()

    result = fit_source_domain_probability_ensemble(
        source_features=source_features,
        source_labels=source_labels,
        source_domains=source_domains,
        target_features=target_features,
        weighting="target_confidence",
        temperature=0.5,
    )

    assert result.metadata["source_domain_ensemble_protocol_category"] == SOURCE_DOMAIN_ENSEMBLE_CATEGORY_2
    assert result.metadata["source_domain_ensemble_valid_for_strict_source_only"] is False
    assert result.metadata["source_domain_ensemble_uses_target_features_for_weighting"] is True
    assert np.isclose(sum(result.domain_weights.values()), 1.0)
    assert all(weight > 0.0 for weight in result.domain_weights.values())


def test_target_feature_similarity_weighting_prefers_closer_domain() -> None:
    source_features, source_labels, source_domains, target_features = _toy_domains()

    result = fit_source_domain_probability_ensemble(
        source_features=source_features,
        source_labels=source_labels,
        source_domains=source_domains,
        target_features=target_features,
        weighting="target_feature_similarity",
        temperature=0.2,
    )

    assert result.domain_weights["a"] > result.domain_weights["b"]
    assert np.isclose(sum(result.domain_weights.values()), 1.0)
    assert result.metadata["source_domain_ensemble_protocol_category"] == SOURCE_DOMAIN_ENSEMBLE_CATEGORY_2


def test_domains_missing_classes_are_skipped() -> None:
    source_features = np.asarray([[-2.0], [-1.5], [1.5], [2.0], [8.0], [8.2]], dtype=float)
    source_labels = np.asarray([0, 0, 1, 1, 0, 0], dtype=object)
    source_domains = np.asarray(["good", "good", "good", "good", "single", "single"], dtype=object)
    target_features = np.asarray([[-1.8], [1.8]], dtype=float)

    result = fit_source_domain_probability_ensemble(
        source_features=source_features,
        source_labels=source_labels,
        source_domains=source_domains,
        target_features=target_features,
        min_classes_per_domain=2,
    )

    assert set(result.models) == {"good"}
    assert result.domain_weights == {"good": 1.0}
    assert result.metadata["source_domain_ensemble_n_trained_domains"] == 1


def test_all_domains_missing_classes_raise_error() -> None:
    with pytest.raises(ValueError, match="No source domain"):
        fit_source_domain_probability_ensemble(
            source_features=[[0.0], [1.0], [2.0]],
            source_labels=[0, 0, 1],
            source_domains=["a", "a", "b"],
            target_features=[[0.5]],
            min_classes_per_domain=2,
        )


def test_weighting_aliases_and_validation() -> None:
    assert normalize_ensemble_weighting("confidence") == "target_confidence"
    assert normalize_ensemble_weighting("low-entropy") == "target_entropy"
    assert normalize_ensemble_weighting("mean-covariance") == "target_feature_similarity"

    with pytest.raises(ValueError, match="weighting"):
        normalize_ensemble_weighting("unknown")


def test_target_labels_are_not_part_of_public_api() -> None:
    source_features, source_labels, source_domains, target_features = _toy_domains()

    with pytest.raises(TypeError):
        fit_source_domain_probability_ensemble(
            source_features=source_features,
            source_labels=source_labels,
            source_domains=source_domains,
            target_features=target_features,
            target_labels=["left", "right"],  # type: ignore[call-arg]
        )

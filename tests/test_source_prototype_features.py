from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_prototype_features import (
    SOURCE_PROTOTYPE_FEATURES_CATEGORY,
    SourcePrototypeFeatureConfig,
    class_prototypes,
    fit_source_prototype_features,
    normalize_prototype_metric,
    normalize_prototype_output,
    prototype_distance_features,
)


def test_prototype_distance_features_use_source_classes_only() -> None:
    source = np.asarray([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=float)
    labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test = np.asarray([[-1.5, 0.0], [1.5, 0.0]], dtype=float)

    result = fit_source_prototype_features(
        source_features=source,
        source_labels=labels,
        test_features=test,
        config={"metric": "squared_euclidean", "use_diagonal_scale": False},
    )

    assert result.classes.tolist() == ["left", "right"]
    assert np.allclose(result.prototypes, np.asarray([[-1.5, 0.0], [1.5, 0.0]]))
    assert result.train_features.shape == (4, 2)
    assert result.test_features.shape == (2, 2)
    assert result.test_features[0, 0] < result.test_features[0, 1]
    assert result.test_features[1, 1] < result.test_features[1, 0]
    assert result.metadata["source_prototype_features_protocol_category"] == SOURCE_PROTOTYPE_FEATURES_CATEGORY
    assert result.metadata["source_prototype_features_uses_source_labels"] is True
    assert result.metadata["source_prototype_features_uses_test_features_for_fitting"] is False
    assert result.metadata["source_prototype_features_uses_test_labels"] is False
    assert result.metadata["source_prototype_features_valid_for_strict_source_only"] is True


def test_direct_prototype_config_normalizes_like_mapping_config() -> None:
    source = np.asarray([[0.0, 0.0], [0.0, 2.0], [3.0, 0.0], [3.0, 2.0]], dtype=float)
    labels = np.asarray(["left", "left", "right", "right"], dtype=object)
    test = np.asarray([[0.0, 1.0], [3.0, 1.0]], dtype=float)
    direct_config = SourcePrototypeFeatureConfig(
        metric="l2",
        output="rbf",
        use_diagonal_scale="false",
        temperature=np.asarray(2.0),
        epsilon=np.float64(1e-6),
    )

    direct = fit_source_prototype_features(
        source_features=source,
        source_labels=labels,
        test_features=test,
        config=direct_config,
    )
    mapping = fit_source_prototype_features(
        source_features=source,
        source_labels=labels,
        test_features=test,
        config={"metric": "l2", "output": "rbf", "use_diagonal_scale": "false", "temperature": np.asarray(2.0), "epsilon": np.float64(1e-6)},
    )

    assert direct_config.metric == "euclidean"
    assert direct_config.output == "rbf_similarity"
    assert direct_config.use_diagonal_scale is False
    assert direct_config.temperature == 2.0
    assert direct_config.epsilon == pytest.approx(1e-6)
    assert np.allclose(direct.feature_scale, np.ones(source.shape[1]))
    assert np.allclose(direct.train_features, mapping.train_features)
    assert np.allclose(direct.test_features, mapping.test_features)
    assert direct.metadata["source_prototype_features_metric"] == "euclidean"
    assert direct.metadata["source_prototype_features_output"] == "rbf_similarity"
    assert direct.metadata["source_prototype_features_use_diagonal_scale"] is False


@pytest.mark.parametrize("field", ["temperature", "epsilon"])
def test_direct_prototype_config_rejects_boolean_numeric_controls(field: str) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        SourcePrototypeFeatureConfig(**{field: True})


@pytest.mark.parametrize("field", ["temperature", "epsilon"])
def test_prototype_config_rejects_vector_numeric_controls(field: str) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        SourcePrototypeFeatureConfig(**{field: np.asarray([1.0, 2.0])})


def test_prototype_similarity_output_is_positive() -> None:
    features = np.asarray([[0.0], [2.0]], dtype=float)
    prototypes = np.asarray([[0.0], [2.0]], dtype=float)

    sims = prototype_distance_features(features, prototypes, output="rbf", feature_scale=[1.0], temperature=1.0)

    assert sims.shape == (2, 2)
    assert np.all(sims > 0.0)
    assert sims[0, 0] > sims[0, 1]
    assert sims[1, 1] > sims[1, 0]


def test_cosine_distance_features() -> None:
    features = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)
    prototypes = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=float)

    distances = prototype_distance_features(features, prototypes, metric="cosine", feature_scale=[1.0, 1.0])

    assert np.allclose(np.diag(distances), 0.0)
    assert np.allclose(distances[0, 1], 1.0)
    assert np.allclose(distances[1, 0], 1.0)


def test_class_prototypes_preserve_composite_label_groups() -> None:
    source = np.asarray([[0.0], [2.0], [10.0], [12.0]], dtype=float)
    labels = [("a", 1), ("a", 1), ("b", 2), ("b", 2)]

    prototypes = class_prototypes(source, labels)

    assert np.allclose(prototypes.ravel(), np.asarray([1.0, 11.0]))


def test_prototype_features_reject_empty_source_labels_with_stable_message() -> None:
    with pytest.raises(ValueError, match="source_labels must contain one value per source row"):
        fit_source_prototype_features(
            source_features=[[0.0], [1.0]],
            source_labels=[],
            test_features=[[0.5]],
        )


def test_aliases_and_validation() -> None:
    assert normalize_prototype_metric("l2") == "euclidean"
    assert normalize_prototype_output("similarity") == "rbf_similarity"

    with pytest.raises(ValueError, match="prototype metric"):
        normalize_prototype_metric("bad")

    with pytest.raises(ValueError, match="prototype output"):
        normalize_prototype_output("bad")

    with pytest.raises(ValueError, match="feature_scale"):
        prototype_distance_features([[0.0, 1.0]], [[0.0, 1.0]], feature_scale=[1.0])


def test_prototype_features_reject_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_prototype_features(
            source_features=[[0.0, 1.0], [1.0, 0.0]],
            source_labels=[0, 1],
            test_features=[[0.0]],
        )


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_prototype_features(
            source_features=[[0.0], [1.0]],
            source_labels=[0, 1],
            test_features=[[0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )

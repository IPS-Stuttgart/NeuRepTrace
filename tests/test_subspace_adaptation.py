from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.subspace_adaptation import (
    SUBSPACE_ADAPTATION_CATEGORY,
    fit_subspace_adaptation,
    normalize_subspace_method,
    subspace_adaptation_config,
    transform_subspace_features,
)


def _shifted_domains():
    source = np.asarray(
        [
            [-1.0, -0.1, 0.0],
            [-0.8, 0.1, 0.2],
            [-1.2, 0.0, -0.1],
            [1.0, 0.1, 0.0],
            [1.2, -0.1, -0.2],
            [0.8, 0.0, 0.1],
        ],
        dtype=float,
    )
    target = source + np.asarray([4.0, 0.25, -0.15])
    labels = np.asarray(["left", "left", "left", "right", "right", "right"], dtype=object)
    return source, target, labels


def test_tca_subspace_uses_unlabeled_target_features_and_reduces_domain_gap() -> None:
    source, target, labels = _shifted_domains()

    result = fit_subspace_adaptation(source, target, source_labels=labels, n_components=1, regularization=1e-4)

    assert result.source_features.shape == (6, 1)
    assert result.target_features.shape == (6, 1)
    assert result.projection.shape == (3, 1)
    assert result.metadata["subspace_adaptation_protocol_category"] == SUBSPACE_ADAPTATION_CATEGORY
    assert result.metadata["subspace_adaptation_uses_target_features"] is True
    assert result.metadata["subspace_adaptation_uses_target_labels"] is False
    assert result.metadata["subspace_adaptation_valid_for_strict_source_only"] is False
    assert result.metadata["subspace_adaptation_valid_for_unlabeled_target_adaptation"] is True
    assert result.metadata["subspace_adaptation_latent_mean_gap"] <= result.metadata["subspace_adaptation_raw_mean_gap"]
    assert np.all(np.isfinite(result.source_features))
    assert np.all(np.isfinite(result.target_features))


def test_transform_subspace_features_reuses_fitted_projection() -> None:
    source, target, _labels = _shifted_domains()
    result = fit_subspace_adaptation(source, target, n_components=2)

    transformed = transform_subspace_features(target[:2], result)

    assert transformed.shape == (2, 2)
    assert np.allclose(transformed, result.target_features[:2])


def test_balanced_tca_uses_source_labels_only() -> None:
    source = np.asarray([[0.0, 0.0], [0.1, 0.0], [0.2, 0.1], [3.0, 3.0], [3.2, 3.1]])
    target = source + np.asarray([1.0, -0.5])
    labels = [("major", 1), ("major", 1), ("major", 1), ("minor", 2), ("minor", 2)]

    result = fit_subspace_adaptation(source, target, source_labels=labels, method="balanced-transfer-component-analysis", n_components="all")

    assert result.metadata["subspace_adaptation_method"] == "balanced_tca"
    assert result.metadata["subspace_adaptation_uses_source_labels"] is True
    assert np.isclose(result.source_weights[:3].sum(), 0.5)
    assert np.isclose(result.source_weights[3:].sum(), 0.5)
    assert result.source_features.shape[1] == 2


def test_balanced_tca_requires_source_labels() -> None:
    source, target, _labels = _shifted_domains()

    with pytest.raises(ValueError, match="requires source_labels"):
        fit_subspace_adaptation(source, target, method="balanced_tca")


def test_subspace_config_and_aliases() -> None:
    assert normalize_subspace_method("transfer-component-analysis") == "tca"
    assert normalize_subspace_method("class-balanced-tca") == "balanced_tca"

    config = subspace_adaptation_config(method="class-balanced-tca", n_components="all", regularization="0.01")
    assert config.method == "balanced_tca"
    assert config.class_balance_source is True
    assert config.n_components == "all"
    assert np.isclose(config.regularization, 0.01)


def test_subspace_adaptation_is_deterministic() -> None:
    source, target, labels = _shifted_domains()

    first = fit_subspace_adaptation(source, target, source_labels=labels, n_components=2)
    second = fit_subspace_adaptation(source, target, source_labels=labels, n_components=2)

    assert np.allclose(first.source_features, second.source_features)
    assert np.allclose(first.target_features, second.target_features)
    assert np.allclose(first.projection, second.projection)


def test_subspace_adaptation_rejects_target_labels_argument() -> None:
    source, target, _labels = _shifted_domains()

    with pytest.raises(TypeError):
        fit_subspace_adaptation(
            source,
            target,
            target_labels=[0, 1],  # type: ignore[call-arg]
        )


def test_subspace_adaptation_guardrails() -> None:
    source, target, _labels = _shifted_domains()

    with pytest.raises(ValueError, match="same feature width"):
        fit_subspace_adaptation(source, target[:, :2])

    with pytest.raises(ValueError, match="n_components"):
        fit_subspace_adaptation(source, target, n_components=0)

    result = fit_subspace_adaptation(source, target, n_components=99)
    assert result.source_features.shape[1] == min(source.shape[1], source.shape[0] + target.shape[0] - 1)

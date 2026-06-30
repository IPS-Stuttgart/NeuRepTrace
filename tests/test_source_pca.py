from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_pca import (
    SOURCE_PCA_CATEGORY,
    SourcePCAConfig,
    apply_source_pca_transform,
    fit_source_pca_reference,
    fit_source_pca_transform,
    source_pca_config,
)


def test_source_pca_transform_shapes_and_metadata() -> None:
    source = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    test = np.asarray([[0.5, 0.5, 1.0], [2.0, 0.0, 1.0]], dtype=float)

    result = fit_source_pca_transform(
        source_features=source,
        test_features=test,
        config={"n_components": 2},
    )

    assert result.train_features.shape == (4, 2)
    assert result.test_features.shape == (2, 2)
    assert result.reference.components.shape == (2, 3)
    assert result.metadata["source_pca_protocol_category"] == SOURCE_PCA_CATEGORY
    assert result.metadata["source_pca_uses_source_features"] is True
    assert result.metadata["source_pca_uses_test_features_for_fitting"] is False
    assert result.metadata["source_pca_uses_test_labels"] is False
    assert result.metadata["source_pca_valid_for_strict_source_only"] is True


def test_source_pca_all_components_are_capped_by_rank() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    test = np.asarray([[0.5, 0.0, 0.0]], dtype=float)

    result = fit_source_pca_transform(source_features=source, test_features=test, config={"n_components": "all"})

    assert result.train_features.shape == (2, 1)
    assert result.test_features.shape == (1, 1)
    assert result.metadata["source_pca_n_components"] == 1


def test_source_pca_reference_can_be_reused() -> None:
    source = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=float)
    test = np.asarray([[1.0, 2.0]], dtype=float)
    reference = fit_source_pca_reference(source, config={"n_components": 1})

    direct = apply_source_pca_transform(test, reference)
    via_fit = fit_source_pca_transform(source_features=source, test_features=test, config={"n_components": 1})

    assert np.allclose(direct, via_fit.test_features)


def test_source_pca_whitening_changes_scale() -> None:
    source = np.asarray([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]], dtype=float)
    test = np.asarray([[3.0, 0.0]], dtype=float)

    unwhitened = fit_source_pca_transform(source_features=source, test_features=test, config={"n_components": 1, "whiten": False})
    whitened = fit_source_pca_transform(source_features=source, test_features=test, config={"n_components": 1, "whiten": True})

    assert not np.allclose(unwhitened.train_features, whitened.train_features)
    assert np.isclose(np.var(whitened.train_features[:, 0], ddof=1), 1.0)
    assert whitened.metadata["source_pca_whiten"] is True


def test_source_pca_config_validation() -> None:
    cfg = source_pca_config(n_components="all", scale=True, whiten=True, epsilon="1e-6")
    assert cfg.n_components == "all"
    assert cfg.scale is True
    assert cfg.whiten is True
    assert np.isclose(cfg.epsilon, 1e-6)

    with pytest.raises(ValueError, match="n_components"):
        fit_source_pca_transform(source_features=[[0.0], [1.0]], test_features=[[0.5]], config={"n_components": 0})


def test_source_pca_config_parses_string_booleans() -> None:
    cfg = source_pca_config(center="false", scale="yes", whiten="off")

    assert cfg.center is False
    assert cfg.scale is True
    assert cfg.whiten is False

    with pytest.raises(ValueError, match="center"):
        source_pca_config(center="maybe")


def test_source_pca_boolean_config_parses_cli_style_values() -> None:
    cfg = source_pca_config(center="false", scale="TRUE", whiten="0")

    assert cfg.center is False
    assert cfg.scale is True
    assert cfg.whiten is False

    result = fit_source_pca_transform(
        source_features=[[0.0], [1.0], [2.0]],
        test_features=[[0.5]],
        config={"n_components": 1, "center": "false", "scale": 0, "whiten": 1},
    )

    assert result.metadata["source_pca_center"] is False
    assert result.metadata["source_pca_scale"] is False
    assert result.metadata["source_pca_whiten"] is True


@pytest.mark.parametrize("field", ["center", "scale", "whiten"])
@pytest.mark.parametrize("bad_value", ["maybe", 2, 0.5, np.asarray([True, False])])
def test_source_pca_rejects_ambiguous_boolean_config(field: str, bad_value: object) -> None:
    with pytest.raises(ValueError, match=field):
        source_pca_config(**{field: bad_value})


def test_source_pca_revalidates_direct_config_instances() -> None:
    with pytest.raises(ValueError, match="epsilon"):
        fit_source_pca_transform(source_features=[[0.0], [1.0]], test_features=[[0.5]], config=SourcePCAConfig(epsilon=0.0))


def test_source_pca_direct_config_uses_boolean_normalizer() -> None:
    result = fit_source_pca_transform(
        source_features=[[1.0], [2.0]],
        test_features=[[3.0]],
        config=SourcePCAConfig(n_components=1, center="false"),
    )

    assert np.allclose(result.reference.mean, [0.0])
    assert result.metadata["source_pca_center"] is False


def test_source_pca_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_pca_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_pca_transform(
            source_features=[[0.0], [1.0]],
            test_features=[[0.5]],
            heldout_labels=[0],
        )

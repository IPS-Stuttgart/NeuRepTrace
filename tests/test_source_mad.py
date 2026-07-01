from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_mad import (
    SOURCE_MAD_CATEGORY,
    SourceMADConfig,
    apply_source_mad_transform,
    fit_source_mad_reference,
    fit_source_mad_transform,
    source_mad_config,
)


def test_source_mad_transform_shapes_and_metadata() -> None:
    source = np.asarray([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [100.0, 13.0]], dtype=float)
    test = np.asarray([[1.0, 11.5], [200.0, 14.0]], dtype=float)

    result = fit_source_mad_transform(source_features=source, test_features=test)

    assert result.train_features.shape == source.shape
    assert result.test_features.shape == test.shape
    assert result.reference.center.shape == (2,)
    assert result.reference.scale.shape == (2,)
    assert result.metadata["source_mad_protocol_category"] == SOURCE_MAD_CATEGORY
    assert result.metadata["source_mad_uses_source_features"] is True
    assert result.metadata["source_mad_uses_test_features_for_fitting"] is False
    assert result.metadata["source_mad_uses_test_labels"] is False
    assert result.metadata["source_mad_valid_for_strict_source_only"] is True


def test_source_mad_reference_reuse_matches_fit_result() -> None:
    source = np.asarray([[0.0], [2.0], [4.0]], dtype=float)
    test = np.asarray([[1.0], [3.0]], dtype=float)
    reference = fit_source_mad_reference(source, config={"normal_consistency": False})

    direct = apply_source_mad_transform(test, reference)
    via_fit = fit_source_mad_transform(source_features=source, test_features=test, config={"normal_consistency": False})

    assert np.allclose(direct, via_fit.test_features)
    assert np.allclose(reference.center, np.asarray([2.0]))
    assert np.allclose(reference.scale, np.asarray([2.0]))


def test_source_mad_can_disable_center_or_scale() -> None:
    source = np.asarray([[1.0], [3.0], [5.0]], dtype=float)
    test = np.asarray([[3.0]], dtype=float)

    no_center = fit_source_mad_transform(source_features=source, test_features=test, config={"center": False, "normal_consistency": False})
    no_scale = fit_source_mad_transform(source_features=source, test_features=test, config={"scale": False})

    assert np.allclose(no_center.reference.center, np.asarray([0.0]))
    assert np.allclose(no_scale.reference.scale, np.asarray([1.0]))


def test_source_mad_scale_stays_centered_when_output_centering_is_disabled() -> None:
    source = np.asarray([[101.0], [103.0], [105.0]], dtype=float)

    reference = fit_source_mad_reference(source, config={"center": False, "normal_consistency": False})

    assert np.allclose(reference.center, np.asarray([0.0]))
    assert np.allclose(reference.scale, np.asarray([2.0]))


def test_source_mad_config_validation() -> None:
    cfg = source_mad_config(center="true", scale="false", epsilon="1e-5")
    assert cfg.center is True
    assert cfg.scale is False
    assert np.isclose(cfg.epsilon, 1e-5)

    with pytest.raises(ValueError, match="boolean"):
        source_mad_config(center="maybe")

    with pytest.raises(ValueError, match="epsilon"):
        source_mad_config(epsilon=0.0)


def test_source_mad_config_accepts_numpy_scalar_controls() -> None:
    cfg = source_mad_config(
        center=np.asarray(True),
        scale=np.asarray(0),
        normal_consistency=np.asarray(1.0),
        epsilon=np.asarray(1e-5),
    )

    assert cfg.center is True
    assert cfg.scale is False
    assert cfg.normal_consistency is True
    assert np.isclose(cfg.epsilon, 1e-5)

    direct = SourceMADConfig(
        center=np.asarray(False),
        scale=np.asarray(True),
        normal_consistency=np.asarray(False),
        epsilon=np.asarray(1e-4),
    )

    assert direct.center is False
    assert direct.scale is True
    assert direct.normal_consistency is False
    assert np.isclose(direct.epsilon, 1e-4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"center": np.asarray([True])}, "center"),
        ({"scale": np.asarray([False])}, "scale"),
        ({"normal_consistency": np.asarray([True])}, "normal_consistency"),
        ({"epsilon": np.asarray([1e-5])}, "epsilon"),
        ({"epsilon": np.asarray(True)}, "epsilon"),
    ],
)
def test_source_mad_config_rejects_vector_or_boolean_numeric_arrays(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        source_mad_config(**kwargs)


def test_source_mad_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_mad_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_mad_transform(source_features=[[0.0], [1.0]], test_features=[[0.5]], heldout_labels=[0])  # type: ignore[call-arg]

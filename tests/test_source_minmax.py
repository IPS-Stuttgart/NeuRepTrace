from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_minmax import (
    SOURCE_MINMAX_CATEGORY,
    SourceMinMaxReference,
    apply_source_minmax_transform,
    fit_source_minmax_reference,
    fit_source_minmax_transform,
)


def test_source_minmax_transform_uses_source_bounds_only() -> None:
    source = np.asarray([[0.0, 10.0], [2.0, 20.0]], dtype=float)
    test = np.array([[1.0, 15.0], [4.0, 30.0]], dtype=float)

    result = fit_source_minmax_transform(source_features=source, test_features=test)

    assert np.allclose(result.train_features, np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert np.allclose(result.test_features, np.array([[0.5, 0.5], [2.0, 2.0]]))
    assert result.metadata["source_minmax_protocol_category"] == SOURCE_MINMAX_CATEGORY
    assert result.metadata["source_minmax_uses_source_features"] is True
    assert result.metadata["source_minmax_uses_test_features_for_fitting"] is False
    assert result.metadata["source_minmax_uses_test_labels"] is False
    assert result.metadata["source_minmax_valid_for_strict_source_only"] is True
    assert result.metadata["source_minmax_clip"] is False


def test_source_minmax_custom_range() -> None:
    result = fit_source_minmax_transform(
        source_features=[[0.0], [2.0]],
        test_features=[[1.0]],
        feature_range=(-1.0, 1.0),
    )

    assert np.allclose(result.train_features.ravel(), np.array([-1.0, 1.0]))
    assert np.allclose(result.test_features.ravel(), np.array([0.0]))


def test_source_minmax_zero_source_range_uses_unit_denominator() -> None:
    result = fit_source_minmax_transform(
        source_features=[[5.0, 0.0], [5.0, 2.0]],
        test_features=[[6.0, 1.0]],
    )

    assert np.allclose(result.train_features[:, 0], np.array([0.0, 0.0]))
    assert np.allclose(result.test_features, np.array([[1.0, 0.5]], dtype=np.float32))
    assert np.all(np.isfinite(result.test_features))


def test_source_minmax_accepts_numpy_vector_range() -> None:
    reference = fit_source_minmax_reference([[0.0], [2.0]], feature_range=np.array([-1.0, 1.0]))

    assert reference.feature_range == (-1.0, 1.0)


def test_source_minmax_reference_can_be_reused() -> None:
    reference = fit_source_minmax_reference([[0.0], [4.0]])
    transformed = apply_source_minmax_transform([[2.0]], reference)

    assert np.allclose(transformed.ravel(), np.array([0.5]))


def test_source_minmax_clip_limits_transformed_values() -> None:
    result = fit_source_minmax_transform(
        source_features=[[0.0], [2.0]],
        test_features=[[-2.0], [1.0], [4.0]],
        clip=True,
    )

    assert np.allclose(result.test_features.ravel(), np.array([0.0, 0.5, 1.0]))
    assert result.metadata["source_minmax_clip"] is True


def test_source_minmax_clip_aliases_work_on_reused_reference() -> None:
    reference = fit_source_minmax_reference([[0.0], [2.0]])

    unclipped = apply_source_minmax_transform([[4.0]], reference)
    clipped = apply_source_minmax_transform([[4.0]], reference, clip="true")

    assert np.allclose(unclipped.ravel(), np.array([2.0]))
    assert np.allclose(clipped.ravel(), np.array([1.0]))


def test_source_minmax_rejects_bad_clip_value() -> None:
    reference = fit_source_minmax_reference([[0.0], [2.0]])

    with pytest.raises(ValueError, match="clip"):
        apply_source_minmax_transform([[1.0]], reference, clip="maybe")


def test_source_minmax_rejects_width_mismatch() -> None:
    with pytest.raises(ValueError, match="same feature width"):
        fit_source_minmax_transform(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_source_minmax_rejects_bad_range() -> None:
    with pytest.raises(ValueError, match="feature_range"):
        fit_source_minmax_reference([[0.0], [1.0]], feature_range=(1.0, 0.0))


@pytest.mark.parametrize(
    "feature_range",
    [
        False,
        (False, True),
        (0.0, True),
        (0.0,),
        (0.0, 1.0, 2.0),
        "01",
        np.array([[0.0], [1.0]]),
    ],
)
def test_source_minmax_rejects_malformed_or_boolean_ranges(feature_range: object) -> None:
    with pytest.raises(ValueError, match="feature_range"):
        fit_source_minmax_reference([[0.0], [1.0]], feature_range=feature_range)


@pytest.mark.parametrize(
    "reference",
    [
        SourceMinMaxReference(minimum=[0.0], maximum=[np.inf], feature_range=(0.0, 1.0), n_fit_rows=2),
        SourceMinMaxReference(minimum=[0.0, 1.0], maximum=[1.0], feature_range=(0.0, 1.0), n_fit_rows=2),
    ],
)
def test_source_minmax_rejects_invalid_reused_reference(reference: SourceMinMaxReference) -> None:
    with pytest.raises(ValueError, match="source minmax reference bounds"):
        apply_source_minmax_transform([[0.5]], reference)


def test_source_minmax_rejects_reused_reference_bad_range() -> None:
    reference = SourceMinMaxReference(minimum=[0.0], maximum=[1.0], feature_range=(1.0, 0.0), n_fit_rows=2)

    with pytest.raises(ValueError, match="feature_range"):
        apply_source_minmax_transform([[0.5]], reference)


def test_heldout_labels_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        fit_source_minmax_transform(
            source_features=[[0.0], [1.0]],
            test_features=[[0.5]],
            heldout_labels=[0],  # type: ignore[call-arg]
        )

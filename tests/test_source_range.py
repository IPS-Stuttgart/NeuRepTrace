from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_range import (
    SOURCE_RANGE_CATEGORY,
    apply_source_range_clip,
    source_feature_range,
    source_range_clip,
)


def test_source_feature_range_uses_source_rows_only() -> None:
    source = np.asarray([[0.0, 10.0], [2.0, 12.0], [1.0, 11.0]], dtype=float)

    lower, upper = source_feature_range(source)

    assert SOURCE_RANGE_CATEGORY == "1_strict_source_only"
    assert np.allclose(lower, np.asarray([0.0, 10.0]))
    assert np.allclose(upper, np.asarray([2.0, 12.0]))


def test_source_range_clip_clips_test_rows_with_source_bounds() -> None:
    source = np.asarray([[0.0, 10.0], [2.0, 12.0]], dtype=float)
    test = np.asarray([[-5.0, 11.0], [5.0, 20.0]], dtype=float)

    train, test_out, lower, upper, train_mask, test_mask, metadata = source_range_clip(source_features=source, test_features=test)

    assert np.allclose(train, source)
    assert np.allclose(test_out, np.asarray([[0.0, 11.0], [2.0, 12.0]]))
    assert np.allclose(lower, np.asarray([0.0, 10.0]))
    assert np.allclose(upper, np.asarray([2.0, 12.0]))
    assert not np.any(train_mask)
    assert test_mask.tolist() == [[True, False], [True, True]]
    assert metadata["source_range_protocol_category"] == SOURCE_RANGE_CATEGORY
    assert metadata["source_range_uses_test_features_for_fitting"] is False
    assert metadata["source_range_uses_test_labels"] is False
    assert metadata["source_range_valid_for_strict_source_only"] is True
    assert metadata["source_range_valid_for_benchmark"] is True


def test_source_range_helpers_accept_one_pass_feature_iterables() -> None:
    source_rows = (iter(row) for row in ([0.0, 10.0], [2.0, 12.0], [1.0, 11.0]))

    lower, upper = source_feature_range(source_rows)

    np.testing.assert_allclose(lower, [0.0, 10.0])
    np.testing.assert_allclose(upper, [2.0, 12.0])

    train, test_out, _lower, _upper, train_mask, test_mask, metadata = source_range_clip(
        source_features=(iter(row) for row in ([0.0, 10.0], [2.0, 12.0])),
        test_features=(iter(row) for row in ([-5.0, 11.0], [5.0, 20.0])),
    )

    np.testing.assert_allclose(train, [[0.0, 10.0], [2.0, 12.0]])
    np.testing.assert_allclose(test_out, [[0.0, 11.0], [2.0, 12.0]])
    assert not np.any(train_mask)
    assert test_mask.tolist() == [[True, False], [True, True]]
    assert metadata["source_range_n_source_rows"] == 2
    assert metadata["source_range_n_test_rows"] == 2


def test_source_range_accepts_object_arrays_containing_generator_rows() -> None:
    source = np.asarray([iter([0.0, 10.0]), iter([2.0, 12.0])], dtype=object)
    test = np.asarray([iter([-5.0, 11.0]), iter([5.0, 20.0])], dtype=object)

    train, test_out, lower, upper, train_mask, test_mask, metadata = source_range_clip(source_features=source, test_features=test)

    np.testing.assert_allclose(train, [[0.0, 10.0], [2.0, 12.0]])
    np.testing.assert_allclose(test_out, [[0.0, 11.0], [2.0, 12.0]])
    np.testing.assert_allclose(lower, [0.0, 10.0])
    np.testing.assert_allclose(upper, [2.0, 12.0])
    assert not np.any(train_mask)
    assert test_mask.tolist() == [[True, False], [True, True]]
    assert metadata["source_range_n_source_rows"] == 2
    assert metadata["source_range_n_test_rows"] == 2


def test_apply_source_range_clip_accepts_one_pass_bounds() -> None:
    clipped, mask = apply_source_range_clip(
        (iter(row) for row in ([-5.0, 11.0], [5.0, 20.0])),
        lower=(value for value in [0.0, 10.0]),
        upper=(value for value in [2.0, 12.0]),
    )

    np.testing.assert_allclose(clipped, [[0.0, 11.0], [2.0, 12.0]])
    assert mask.tolist() == [[True, False], [True, True]]


def test_apply_source_range_clip_validates_width() -> None:
    with pytest.raises(ValueError, match="width"):
        apply_source_range_clip([[0.0, 1.0]], lower=[0.0], upper=[1.0])


def test_apply_source_range_clip_rejects_nonfinite_bounds() -> None:
    with pytest.raises(ValueError, match="finite"):
        apply_source_range_clip([[0.0, 1.0]], lower=[0.0, np.nan], upper=[1.0, 2.0])
    with pytest.raises(ValueError, match="finite"):
        apply_source_range_clip([[0.0, 1.0]], lower=[0.0, 0.0], upper=[1.0, np.inf])


def test_apply_source_range_clip_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="lower bounds.*upper"):
        apply_source_range_clip([[0.0, 1.0]], lower=[1.0, 0.0], upper=[0.0, 2.0])


def test_source_feature_range_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        source_feature_range([[0.0], [float("nan")]])


@pytest.mark.parametrize(
    "bad_source_features",
    [
        [[True, False], [False, True]],
        np.asarray([[True, False], [False, True]], dtype=bool),
        np.asarray([[True, 0.0], [False, 1.0]], dtype=object),
    ],
)
def test_source_feature_range_rejects_boolean_values(bad_source_features) -> None:
    with pytest.raises(ValueError, match="source_features.*non-boolean"):
        source_feature_range(bad_source_features)


def test_source_range_clip_rejects_boolean_test_features() -> None:
    with pytest.raises(ValueError, match="test_features.*non-boolean"):
        source_range_clip(source_features=[[0.0, 1.0]], test_features=[[True, False]])


def test_apply_source_range_clip_rejects_boolean_bounds() -> None:
    with pytest.raises(ValueError, match="lower.*non-boolean"):
        apply_source_range_clip([[0.0, 1.0]], lower=[False, 0.0], upper=[1.0, 2.0])
    with pytest.raises(ValueError, match="upper.*non-boolean"):
        apply_source_range_clip([[0.0, 1.0]], lower=[0.0, 0.0], upper=[True, 2.0])


def test_heldout_arguments_are_not_part_of_public_api() -> None:
    with pytest.raises(TypeError):
        source_feature_range([[0.0], [1.0]], heldout_features=[[0.5]])  # type: ignore[call-arg]

from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_range_selector import fit_source_range_selector, select_source_range_features


def test_source_range_selector_drops_constant_columns() -> None:
    source = np.asarray([[1.0, 0.0, 0.0, 5.0], [1.0, 1.0, 0.0, 7.0], [1.0, 2.0, 0.0, 9.0]])
    test = np.asarray([[99.0, 3.0, 4.0, 11.0]])

    result = fit_source_range_selector(source_features=source, test_features=test, min_range=0.0)

    assert result.selected_indices.tolist() == [1, 3]
    assert result.test_features.tolist() == [[3.0, 11.0]]
    assert result.metadata["source_range_selector_uses_test_features_for_fitting"] is False


def test_source_range_selector_top_k_and_fallback() -> None:
    source = np.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 10.0], [2.0, 4.0, 20.0]])
    test = np.asarray([[3.0, 6.0, 30.0]])

    result = fit_source_range_selector(source_features=source, test_features=test, top_k=1)

    assert result.selected_indices.tolist() == [2]
    assert select_source_range_features([0.0, 0.0, 2.0], min_range=10.0).tolist() == [2]


def test_source_range_selector_accepts_string_scalar_controls() -> None:
    selected = select_source_range_features([0.0, 2.0, 4.0], min_range="1", top_k="2")

    assert selected.tolist() == [1, 2]


def test_source_range_selector_accepts_one_pass_feature_iterables() -> None:
    source_rows = (iter(row) for row in ([1.0, 0.0, 5.0], [1.0, 2.0, 9.0]))
    test_rows = (iter(row) for row in ([99.0, 3.0, 11.0],))

    result = fit_source_range_selector(source_features=source_rows, test_features=test_rows)

    assert result.selected_indices.tolist() == [1, 2]
    np.testing.assert_allclose(result.test_features, [[3.0, 11.0]])


def test_source_range_selector_accepts_one_pass_manual_ranges() -> None:
    ranges = (value for value in [0.0, 2.0, 4.0])

    selected = select_source_range_features(ranges, min_range=1.0)

    assert selected.tolist() == [1, 2]


def test_source_range_selector_validation() -> None:
    with pytest.raises(ValueError):
        select_source_range_features([1.0], min_range=-0.1)

    with pytest.raises(ValueError):
        select_source_range_features([1.0], top_k=0)

    with pytest.raises(ValueError, match="feature widths"):
        fit_source_range_selector(source_features=[[0.0, 1.0]], test_features=[[0.0]])


def test_source_range_selector_rejects_ambiguous_scalar_controls() -> None:
    with pytest.raises(ValueError, match="top_k"):
        select_source_range_features([1.0, 2.0], top_k=1.5)

    with pytest.raises(ValueError, match="top_k"):
        select_source_range_features([1.0, 2.0], top_k=True)

    with pytest.raises(ValueError, match="min_range"):
        select_source_range_features([1.0, 2.0], min_range=[0.0, 1.0])


@pytest.mark.parametrize(
    "bad_source_features",
    [
        [[True, False], [False, True]],
        np.asarray([[True, False], [False, True]], dtype=bool),
        np.asarray([[True, 0.0], [False, 1.0]], dtype=object),
        (iter(row) for row in ([True, 0.0], [False, 1.0])),
    ],
)
def test_source_range_selector_rejects_boolean_source_features(bad_source_features) -> None:
    with pytest.raises(ValueError, match="source_features.*boolean"):
        fit_source_range_selector(source_features=bad_source_features, test_features=[[0.0, 1.0]])


def test_source_range_selector_rejects_boolean_test_features() -> None:
    with pytest.raises(ValueError, match="test_features.*boolean"):
        fit_source_range_selector(source_features=[[0.0, 1.0]], test_features=[[True, False]])


def test_select_source_range_features_rejects_boolean_ranges() -> None:
    with pytest.raises(ValueError, match="ranges.*non-boolean"):
        select_source_range_features([False, True])


@pytest.mark.parametrize(
    "bad_source_features",
    [
        [[1.0 + 2.0j, 0.0], [2.0, 1.0]],
        np.asarray([[1.0 + 2.0j, 0.0], [2.0, 1.0]], dtype=np.complex128),
        np.asarray([[1.0 + 2.0j, 0.0], [2.0, 1.0]], dtype=object),
        (iter(row) for row in ([1.0 + 2.0j, 0.0], [2.0, 1.0])),
    ],
)
def test_source_range_selector_rejects_complex_source_features(bad_source_features) -> None:
    with pytest.raises(ValueError, match="source_features.*complex"):
        fit_source_range_selector(source_features=bad_source_features, test_features=[[0.0, 1.0]])


def test_source_range_selector_rejects_complex_test_features() -> None:
    complex_test = np.asarray([[1.0 + 2.0j, 0.0]], dtype=np.complex128)

    with pytest.raises(ValueError, match="test_features.*complex"):
        fit_source_range_selector(source_features=[[0.0, 1.0], [1.0, 2.0]], test_features=complex_test)


def test_select_source_range_features_rejects_complex_ranges() -> None:
    complex_ranges = np.asarray([1.0 + 2.0j, 3.0], dtype=np.complex128)

    with pytest.raises(ValueError, match="ranges.*complex"):
        select_source_range_features(complex_ranges)

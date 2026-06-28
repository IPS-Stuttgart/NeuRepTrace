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


def test_source_range_selector_validation() -> None:
    with pytest.raises(ValueError):
        select_source_range_features([1.0], min_range=-0.1)

    with pytest.raises(ValueError):
        select_source_range_features([1.0], top_k=0)

    with pytest.raises(ValueError, match="feature widths"):
        fit_source_range_selector(source_features=[[0.0, 1.0]], test_features=[[0.0]])

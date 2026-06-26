from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_selection import select_source_domains_by_target_similarity


_SOURCE_FEATURES = np.asarray([[0.0], [0.1], [2.0], [2.2]], dtype=float)
_SOURCE_DOMAINS = np.asarray(["near", "near", "far", "far"], dtype=object)
_TARGET_FEATURES = np.asarray([[0.05], [0.0]], dtype=float)


@pytest.mark.parametrize("bad_top_k", [[1], (1,), {"k": 1}, {1}, np.asarray([1])])
def test_source_selection_rejects_nonscalar_top_k(bad_top_k: object) -> None:
    with pytest.raises(ValueError, match="top_k"):
        select_source_domains_by_target_similarity(
            _SOURCE_FEATURES,
            _SOURCE_DOMAINS,
            _TARGET_FEATURES,
            top_k=bad_top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_max_distance", [[0.5], (0.5,), {"d": 0.5}, {0.5}, np.asarray([0.5])])
def test_source_selection_rejects_nonscalar_max_distance(bad_max_distance: object) -> None:
    with pytest.raises(ValueError, match="max_distance"):
        select_source_domains_by_target_similarity(
            _SOURCE_FEATURES,
            _SOURCE_DOMAINS,
            _TARGET_FEATURES,
            max_distance=bad_max_distance,  # type: ignore[arg-type]
        )


def test_source_selection_accepts_scalar_numpy_optional_bounds() -> None:
    result = select_source_domains_by_target_similarity(
        _SOURCE_FEATURES,
        _SOURCE_DOMAINS,
        _TARGET_FEATURES,
        metric="mean",
        top_k=np.asarray(1),  # type: ignore[arg-type]
        max_distance=np.asarray(10.0),  # type: ignore[arg-type]
    )

    assert result.selected_domains == ("near",)
    assert result.selected_mask.tolist() == [True, True, False, False]

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


_SOURCE_FEATURES = np.asarray([[0.0], [0.1], [2.0], [2.2]], dtype=float)
_SOURCE_DOMAINS = np.asarray(["near", "near", "far", "far"], dtype=object)
_TARGET_FEATURES = np.asarray([[0.05], [0.0]], dtype=float)


def _load_source_selection_core():
    path = Path(__file__).resolve().parents[1] / "src" / "neureptrace" / "decoding" / "source_selection.py"
    spec = importlib.util.spec_from_file_location("neureptrace_source_selection_core_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("bad_top_k", [[1], (1,), {"k": 1}, {1}, np.asarray([1])])
def test_source_selection_core_rejects_nonscalar_top_k(bad_top_k: object) -> None:
    module = _load_source_selection_core()

    with pytest.raises(ValueError, match="top_k"):
        module.select_source_domains_by_target_similarity(
            _SOURCE_FEATURES,
            _SOURCE_DOMAINS,
            _TARGET_FEATURES,
            top_k=bad_top_k,
        )


@pytest.mark.parametrize("bad_max_distance", [[0.5], (0.5,), {"d": 0.5}, {0.5}, np.asarray([0.5])])
def test_source_selection_core_rejects_nonscalar_max_distance(bad_max_distance: object) -> None:
    module = _load_source_selection_core()

    with pytest.raises(ValueError, match="max_distance"):
        module.select_source_domains_by_target_similarity(
            _SOURCE_FEATURES,
            _SOURCE_DOMAINS,
            _TARGET_FEATURES,
            max_distance=bad_max_distance,
        )


def test_source_selection_core_accepts_scalar_numpy_optional_bounds() -> None:
    module = _load_source_selection_core()

    result = module.select_source_domains_by_target_similarity(
        _SOURCE_FEATURES,
        _SOURCE_DOMAINS,
        _TARGET_FEATURES,
        metric="mean",
        top_k=np.asarray(1),
        max_distance=np.asarray(10.0),
    )

    assert result.selected_domains == ("near",)
    assert result.selected_mask.tolist() == [True, True, False, False]

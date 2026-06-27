from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.mcca import fit_class_mcca, fit_mcca


def _aligned_by_subject() -> dict[str, np.ndarray]:
    return {
        "a": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "b": np.array([[0.0, 0.0], [0.9, 0.1], [0.1, 1.0]]),
    }


def _features_by_subject() -> dict[str, np.ndarray]:
    return {
        "a": np.array([[1.0], [2.0], [10.0], [20.0]]),
        "b": np.array([[101.0], [102.0], [110.0], [120.0]]),
    }


def _labels_by_subject() -> dict[str, np.ndarray]:
    return {
        "a": np.array([1, 1, 2, 2]),
        "b": np.array([1, 1, 2, 2]),
    }


def test_fit_mcca_rejects_flag_component_count() -> None:
    with pytest.raises(ValueError, match="n_components"):
        fit_mcca(_aligned_by_subject(), n_components=True)


def test_fit_class_mcca_rejects_flag_component_count() -> None:
    with pytest.raises(ValueError, match="n_components"):
        fit_class_mcca(
            _features_by_subject(),
            _labels_by_subject(),
            n_components=True,
        )


def test_fit_mcca_still_accepts_integral_float_component_count() -> None:
    model = fit_mcca(_aligned_by_subject(), n_components=np.float64(1.0))

    assert model.n_components == 1

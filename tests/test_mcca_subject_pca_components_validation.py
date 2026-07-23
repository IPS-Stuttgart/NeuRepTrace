from __future__ import annotations

import numpy as np
import pytest

from neureptrace._mcca_subject_pca_components_patch import _normalize_subject_pca_components
from neureptrace.decoding.mcca import fit_mcca


def _features_by_subject() -> dict[str, np.ndarray]:
    return {
        "a": np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "b": np.array([[0.0, 0.0], [0.9, 0.1], [0.1, 1.0]]),
    }


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), 1.5, np.float64(2.5), np.nan, "9007199254740993.5"],
)
def test_fit_mcca_rejects_invalid_subject_pca_components(value: object) -> None:
    with pytest.raises(ValueError, match="subject_pca_components"):
        fit_mcca(_features_by_subject(), n_components=1, subject_pca_components=value)  # type: ignore[arg-type]


def test_fit_mcca_accepts_integral_subject_pca_component_cap() -> None:
    model = fit_mcca(_features_by_subject(), n_components=1, subject_pca_components=1.0)

    assert model.n_components == 1
    assert all(projection.rank == 1 for projection in model.projections.values())


def test_subject_pca_component_parser_preserves_large_exact_integers() -> None:
    components = 2**53 + 1

    assert _normalize_subject_pca_components(components) == components
    assert _normalize_subject_pca_components(str(components)) == components


def test_fit_mcca_accepts_integer_component_cap_beyond_float_range() -> None:
    model = fit_mcca(_features_by_subject(), n_components=1, subject_pca_components=10**400)

    assert model.n_components == 1
    assert all(projection.rank == 2 for projection in model.projections.values())

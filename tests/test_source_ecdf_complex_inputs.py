from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding import source_ecdf, source_ecdf_uniform


ECDF_MODULES = (source_ecdf, source_ecdf_uniform)
REFERENCE_APIS = (
    (source_ecdf, "fit_source_ecdf_reference", "apply_source_ecdf_transform"),
    (source_ecdf_uniform, "fit_source_ecdf_map", "apply_source_ecdf_transform"),
)


@pytest.mark.parametrize("module", ECDF_MODULES, ids=["rank-ecdf", "quantile-ecdf"])
@pytest.mark.parametrize("complex_argument", ["source_features", "test_features"])
def test_source_ecdf_fit_rejects_complex_feature_matrices(module, complex_argument: str) -> None:
    real_features = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float)
    complex_features = real_features.astype(np.complex128)
    complex_features[0, 0] += 1.0j
    arguments = {
        "source_features": real_features,
        "test_features": real_features.copy(),
    }
    arguments[complex_argument] = complex_features

    with pytest.raises(ValueError, match=rf"{complex_argument}.*real-valued"):
        module.fit_source_ecdf_transform(**arguments)


@pytest.mark.parametrize(
    "module,fit_name",
    [(module, fit_name) for module, fit_name, _ in REFERENCE_APIS],
    ids=["rank-reference", "quantile-map"],
)
def test_source_ecdf_reference_fit_rejects_complex_features(module, fit_name: str) -> None:
    complex_features = np.array([[0.0 + 1.0j], [1.0 + 0.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match=r"source_features.*real-valued"):
        getattr(module, fit_name)(complex_features)


@pytest.mark.parametrize(
    "module,fit_name,apply_name",
    REFERENCE_APIS,
    ids=["rank-reference", "quantile-map"],
)
def test_source_ecdf_apply_rejects_complex_features(module, fit_name: str, apply_name: str) -> None:
    real_features = np.array([[0.0], [1.0]], dtype=float)
    reference = getattr(module, fit_name)(real_features)
    complex_features = np.array([[0.5 + 2.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match=r"features.*real-valued"):
        getattr(module, apply_name)(complex_features, reference)

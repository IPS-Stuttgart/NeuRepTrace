from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding import source_ecdf, source_ecdf_uniform


ECDF_APIS = (
    (source_ecdf, source_ecdf.SourceECDFConfig, "fit_source_ecdf_reference", "apply_source_ecdf_transform"),
    (source_ecdf_uniform, source_ecdf_uniform.SourceEcdfConfig, "fit_source_ecdf_map", "apply_source_ecdf_transform"),
)


@pytest.mark.parametrize("module,_,__,___", ECDF_APIS, ids=["rank-ecdf", "quantile-ecdf"])
@pytest.mark.parametrize("boolean_argument", ["source_features", "test_features"])
def test_source_ecdf_fit_rejects_boolean_feature_matrices(module, _, __, ___, boolean_argument: str) -> None:
    real_features = np.asarray([[0.0, 1.0], [1.0, 2.0]], dtype=float)
    boolean_features = np.asarray([[True, False], [False, True]], dtype=bool)
    arguments = {
        "source_features": real_features,
        "test_features": real_features.copy(),
    }
    arguments[boolean_argument] = boolean_features

    with pytest.raises(ValueError, match=rf"{boolean_argument}.*non-boolean"):
        module.fit_source_ecdf_transform(**arguments)


@pytest.mark.parametrize("module,_,fit_name,apply_name", ECDF_APIS, ids=["rank-ecdf", "quantile-ecdf"])
def test_source_ecdf_reference_and_apply_reject_boolean_features(module, _, fit_name: str, apply_name: str) -> None:
    boolean_features = [[0.0, True], [1.0, False]]

    with pytest.raises(ValueError, match=r"source_features.*non-boolean"):
        getattr(module, fit_name)(boolean_features)

    reference = getattr(module, fit_name)([[0.0, 1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match=r"features.*non-boolean"):
        getattr(module, apply_name)(boolean_features, reference)


@pytest.mark.parametrize("module,config_type,_,__", ECDF_APIS, ids=["rank-ecdf", "quantile-ecdf"])
@pytest.mark.parametrize("epsilon", [np.complex64(1e-4 + 2j), np.complex128(1e-4 + 2j)])
def test_source_ecdf_rejects_complex_epsilon(module, config_type, _, __, epsilon) -> None:
    with pytest.raises(ValueError, match="epsilon"):
        module.source_ecdf_config(epsilon=epsilon)

    with pytest.raises(ValueError, match="epsilon"):
        config_type(epsilon=epsilon)

from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding import source_ecdf, source_ecdf_uniform


@pytest.mark.parametrize(
    ("module", "config_type", "fit_name"),
    [
        (source_ecdf, source_ecdf.SourceECDFConfig, "fit_source_ecdf_reference"),
        (source_ecdf_uniform, source_ecdf_uniform.SourceEcdfConfig, "fit_source_ecdf_map"),
    ],
    ids=["rank-ecdf", "quantile-ecdf"],
)
@pytest.mark.parametrize(
    "epsilon",
    [
        np.complex64(1e-6 + 0.25j),
        np.complex128(1e-6 + 0.25j),
    ],
    ids=["complex64", "complex128"],
)
def test_source_ecdf_rejects_complex_epsilon_controls(
    module,
    config_type,
    fit_name: str,
    epsilon,
) -> None:
    message = r"epsilon must be in \(0, 0\.5\)"

    with pytest.raises(ValueError, match=message):
        module.source_ecdf_config(epsilon=epsilon)

    with pytest.raises(ValueError, match=message):
        config_type(epsilon=epsilon)

    with pytest.raises(ValueError, match=message):
        getattr(module, fit_name)(
            [[0.0], [1.0]],
            config={"epsilon": epsilon},
        )

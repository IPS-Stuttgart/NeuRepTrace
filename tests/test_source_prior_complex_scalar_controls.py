from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_prior import SourcePriorConfig, estimate_source_class_prior, source_prior_config


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("smoothing", np.complex64(0.5 + 2.0j), "smoothing must be non-negative and finite"),
        ("smoothing", np.complex128(0.5 + 2.0j), "smoothing must be non-negative and finite"),
        ("epsilon", np.complex64(1e-6 + 2.0j), "epsilon must be positive and finite"),
        ("epsilon", np.complex128(1e-6 + 2.0j), "epsilon must be positive and finite"),
    ],
)
def test_source_prior_config_rejects_complex_numeric_controls(field: str, value, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        source_prior_config(**{field: value})

    with pytest.raises(ValueError, match=message):
        SourcePriorConfig(**{field: value})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"smoothing": np.complex128(0.5 + 2.0j)}, "smoothing must be non-negative and finite"),
        ({"epsilon": np.complex128(1e-6 + 2.0j)}, "epsilon must be positive and finite"),
    ],
)
def test_source_prior_estimation_rejects_complex_numeric_controls(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_source_class_prior([0, 1], **kwargs)

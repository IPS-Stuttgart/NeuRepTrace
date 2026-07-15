from __future__ import annotations

import numpy as np

from neureptrace.decoding.signed_sqrt import signed_sqrt_transform


def test_signed_sqrt_avoids_intermediate_scale_overflow() -> None:
    transformed = signed_sqrt_transform([[1e308, -1e308]], scale=1e-308)

    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed, np.asarray([[1e308, -1e308]]), rtol=1e-15)

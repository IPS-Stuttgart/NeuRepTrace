from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.domain_importance import domain_importance_config


@pytest.mark.parametrize(
    "clip",
    [
        [True, 2.0],
        [0.0, False],
        (np.bool_(True), 2.0),
        np.asarray([True, False], dtype=object),
    ],
)
def test_domain_importance_config_rejects_boolean_clip_bounds(clip: object) -> None:
    with pytest.raises(ValueError, match="clip bounds"):
        domain_importance_config(clip=clip)

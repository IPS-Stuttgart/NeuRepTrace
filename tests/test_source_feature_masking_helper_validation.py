from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_masking import feature_mask_indices


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_features": True}, "n_features"),
        ({"n_features": np.asarray(10)}, "n_features"),
        ({"mask_fraction": np.asarray(0.5)}, "mask_fraction"),
        ({"block_size": [3]}, "block_size"),
        ({"block_size": np.asarray([3])}, "block_size"),
    ],
)
def test_feature_mask_indices_rejects_non_scalar_numeric_controls(kwargs: dict[str, object], message: str) -> None:
    params = {
        "n_features": 10,
        "mask_fraction": 0.5,
        "mask_mode": "block",
        "block_size": 3,
        "rng": np.random.default_rng(13),
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        feature_mask_indices(**params)

from __future__ import annotations

import numpy as np
import pytest

from neureptrace.bushmeg_supervised_lowrank_loso import _as_bool, _candidate_grid


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        (np.bool_(True), True),
        (np.bool_(False), False),
        (1, True),
        (0, False),
        (np.int64(1), True),
        (np.int64(0), False),
        ("yes", True),
        ("off", False),
    ],
)
def test_supervised_lowrank_bool_config_accepts_unambiguous_values(value, expected):
    assert _as_bool(value) is expected


@pytest.mark.parametrize("value", [2, -1, np.int64(2), 0.0, 1.0, 0.5, "maybe", [1], np.asarray([1])])
def test_supervised_lowrank_bool_config_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match="boolean"):
        _as_bool(value)


def test_supervised_lowrank_candidate_grid_rejects_ambiguous_include_deltas():
    config = {
        "supervised_lowrank_loso": {
            "candidate_grid": {
                "epoch_windows": [{"name": "post", "start": 0.0, "stop": 0.2}],
                "temporal_bins": [2],
                "pls_components": [1],
                "decoders": ["lda"],
                "c_grid": [1.0],
                "include_deltas": [2],
            }
        }
    }

    with pytest.raises(ValueError, match="boolean"):
        _candidate_grid(config)

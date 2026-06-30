from __future__ import annotations

import numpy as np
import pytest

from neureptrace.bushmeg_supervised_lowrank_loso import _as_bool, _candidate_grid


def test_supervised_lowrank_bool_config_rejects_non_binary_numbers():
    with pytest.raises(ValueError, match="boolean"):
        _as_bool(2)
    with pytest.raises(ValueError, match="boolean"):
        _as_bool(np.int64(-1))
    with pytest.raises(ValueError, match="boolean"):
        _as_bool(0.5)


def test_supervised_lowrank_bool_config_accepts_explicit_scalar_values():
    assert _as_bool("false") is False
    assert _as_bool("ON") is True
    assert _as_bool(np.array(0)) is False
    assert _as_bool(np.float64(1.0)) is True


def test_supervised_lowrank_candidate_grid_rejects_invalid_include_deltas_value():
    config = {
        "supervised_lowrank_loso": {
            "candidate_grid": {
                "epoch_windows": [{"name": "toy", "start": 0.0, "stop": 0.1}],
                "temporal_bins": [1],
                "pls_components": [1],
                "c_grid": [1.0],
                "include_deltas": [2],
            }
        }
    }

    with pytest.raises(ValueError, match="boolean"):
        _candidate_grid(config)

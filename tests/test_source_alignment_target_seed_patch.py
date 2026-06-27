from __future__ import annotations

import pytest

from neureptrace.decoding.source_alignment import source_alignment_config


def test_source_alignment_config_rejects_negative_target_calibration_seed() -> None:
    with pytest.raises(ValueError, match="alignment_target_calibration_seed"):
        source_alignment_config(target_calibration_seed=-1)


def test_source_alignment_config_accepts_zero_target_calibration_seed() -> None:
    config = source_alignment_config(target_calibration_seed=0)

    assert config.target_calibration_seed == 0

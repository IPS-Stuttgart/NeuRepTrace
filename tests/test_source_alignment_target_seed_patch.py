import pytest

from neureptrace.decoding.source_alignment import TARGET_CALIBRATED_ALIGNMENT, source_alignment_config


def test_source_alignment_config_rejects_negative_target_calibration_seed():
    with pytest.raises(ValueError, match="alignment_target_calibration_seed must be at least 0"):
        source_alignment_config(
            method="procrustes",
            target_projection=TARGET_CALIBRATED_ALIGNMENT,
            target_calibration_seed=-1,
        )


def test_source_alignment_config_accepts_string_target_calibration_seed():
    config = source_alignment_config(
        method="procrustes",
        target_projection=TARGET_CALIBRATED_ALIGNMENT,
        target_calibration_seed="0",
    )

    assert config.target_calibration_seed == 0

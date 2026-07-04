from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.calibrated_prototypes import calibrated_prototype_config


def test_calibrated_prototype_fixed_weight_accepts_numpy_scalar() -> None:
    cfg = calibrated_prototype_config(fixed_calibration_weight=np.asarray(0.25))

    assert cfg.fixed_calibration_weight == pytest.approx(0.25)


def test_calibrated_prototype_fixed_weight_accepts_numpy_none_string() -> None:
    cfg = calibrated_prototype_config(fixed_calibration_weight=np.asarray("none"))

    assert cfg.fixed_calibration_weight is None


def test_calibrated_prototype_fixed_weight_rejects_numpy_vector_cleanly() -> None:
    with pytest.raises(ValueError, match="fixed_calibration_weight"):
        calibrated_prototype_config(fixed_calibration_weight=np.asarray([0.25, 0.5]))

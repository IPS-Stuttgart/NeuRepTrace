from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.calibrated_prototypes import (
    calibrated_prototype_config,
    fit_calibrated_prototype_decoder,
)


def test_calibrated_prototype_config_parses_string_boolean_controls() -> None:
    assert calibrated_prototype_config(diagonal_scale="false").diagonal_scale is False
    assert calibrated_prototype_config(diagonal_scale="0").diagonal_scale is False
    assert calibrated_prototype_config(diagonal_scale="on").diagonal_scale is True

    with pytest.raises(ValueError, match="diagonal_scale"):
        calibrated_prototype_config(diagonal_scale="maybe")


def test_calibrated_prototype_string_false_disables_feature_scaling() -> None:
    source = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 10.0], [2.0, 10.0]], dtype=float)
    source_labels = np.asarray(["a", "a", "b", "b"], dtype=object)
    calibration = np.asarray([[1.0, 0.0], [1.0, 10.0]], dtype=float)
    calibration_labels = np.asarray(["a", "b"], dtype=object)
    evaluation = np.asarray([[1.0, 5.0]], dtype=float)

    result = fit_calibrated_prototype_decoder(
        source_features=source,
        source_labels=source_labels,
        calibration_features=calibration,
        calibration_labels=calibration_labels,
        eval_features=evaluation,
        config={"diagonal_scale": "false"},
    )

    assert np.allclose(result.feature_scale, np.ones(source.shape[1], dtype=np.float32))
    assert result.metadata["calibrated_prototype_diagonal_scale"] is False

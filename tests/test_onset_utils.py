from __future__ import annotations

import numpy as np
import pytest

from neureptrace._onset_utils import validate_detection_options


def test_validate_detection_options_accepts_real_numeric_options() -> None:
    validate_detection_options(
        threshold_quantile=np.float64(0.5),
        threshold_method="adaptive",
        threshold_methods=("adaptive", "fixed"),
        min_consecutive=np.int64(2),
        min_duration=0.0,
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"threshold_quantile": True}, "threshold_quantile must be a real-valued number"),
        ({"threshold_quantile": "0.5"}, "threshold_quantile must be a real-valued number"),
        ({"threshold_quantile": 1.5}, "threshold_quantile must be between 0 and 1"),
        ({"min_consecutive": True}, "min_consecutive must be a real-valued number"),
        ({"min_consecutive": 1.5}, "min_consecutive must be an integer"),
        ({"min_consecutive": 0}, "min_consecutive must be at least 1"),
        ({"min_duration": False}, "min_duration must be a real-valued number"),
        ({"min_duration": float("nan")}, "min_duration must be finite"),
    ],
)
def test_validate_detection_options_rejects_malformed_numeric_options(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_detection_options(**kwargs)

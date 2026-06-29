from __future__ import annotations

import numpy as np
import pytest

from neureptrace.bushmeg_all_protocols import category3_calibration_evaluation_split, select_bushmeg_target_calibration_split


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"per_class": True}, "per_class.*boolean"),
        ({"seed": np.bool_(True)}, "seed.*boolean"),
        ({"min_evaluation_per_class": False}, "min_evaluation_per_class.*boolean"),
    ],
)
def test_protocol3_target_calibration_split_rejects_boolean_integer_options(override: dict[str, object], message: str) -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    kwargs: dict[str, object] = {"per_class": 1, "seed": 13, "min_evaluation_per_class": 1}
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        select_bushmeg_target_calibration_split(labels, **kwargs)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"calibration_per_class": True}, "calibration_per_class.*boolean"),
        ({"seed": np.asarray(True)}, "seed.*boolean"),
    ],
)
def test_category3_calibration_evaluation_split_rejects_boolean_integer_options(override: dict[str, object], message: str) -> None:
    labels = np.asarray([0, 0, 1, 1, 2, 2])
    kwargs: dict[str, object] = {"calibration_per_class": 1, "seed": 13}
    kwargs.update(override)

    with pytest.raises(ValueError, match=message):
        category3_calibration_evaluation_split(labels, **kwargs)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neureptrace.io import fieldtrip_mat


@pytest.mark.parametrize("invalid_time", [np.nan, np.inf, -np.inf])
def test_load_fieldtrip_mat_rejects_nonfinite_time_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_time: float,
) -> None:
    data = {
        "trial": np.asarray([[1.0, 2.0]]),
        "time": np.asarray([0.0, invalid_time]),
        "label": np.asarray(["MEG001"], dtype=object),
    }
    monkeypatch.setattr(fieldtrip_mat, "_loadmat", lambda _path: {"data": data})

    with pytest.raises(ValueError, match=r"Time vector 0 must contain only finite values"):
        fieldtrip_mat.load_fieldtrip_mat(
            tmp_path / "unused.mat",
            fieldtrip_mat.FieldTripMatSpec(variable="data", trialinfo_field=None),
        )

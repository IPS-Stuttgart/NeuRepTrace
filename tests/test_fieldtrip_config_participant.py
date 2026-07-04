from __future__ import annotations

import numpy as np
from scipy.io import savemat

from neureptrace.io import load_fieldtrip_mat_epochs


def test_fieldtrip_config_participant_is_added_to_metadata(tmp_path):
    mat_path = tmp_path / "participant.mat"
    data = np.arange(12, dtype=float).reshape(2, 2, 3)
    savemat(
        mat_path,
        {
            "data": {
                "trial": data,
                "time": np.array([0.0, 0.001, 0.002], dtype=float),
                "label": np.array(["MEG001", "MEG002"], dtype=object),
                "trialinfo": np.array([11, 12], dtype=int),
            }
        },
    )

    dataset = load_fieldtrip_mat_epochs(
        mat_path,
        {
            "variable": "data",
            "participant": 7,
            "metadata": {"columns": [{"name": "stimulus", "index": 0}]},
        },
    )

    assert dataset.metadata["participant"].tolist() == ["7", "7"]
    assert dataset.metadata["stimulus"].tolist() == [11, 12]

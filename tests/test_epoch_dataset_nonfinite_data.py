from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.io.dataset import EpochDataset


@pytest.mark.parametrize("nonfinite_value", [np.nan, np.inf, -np.inf])
def test_epoch_dataset_rejects_nonfinite_signal_data(nonfinite_value: float) -> None:
    data = np.ones((1, 1, 3), dtype=float)
    data[0, 0, 1] = nonfinite_value

    with pytest.raises(ValueError, match=r"EpochDataset\.data.*finite"):
        EpochDataset(
            data=data,
            times=np.asarray([0.0, 0.1, 0.2]),
            channel_names=["MEG001"],
            metadata=pd.DataFrame({"trial": [0]}),
        )

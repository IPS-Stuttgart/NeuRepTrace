from __future__ import annotations

import numpy as np

from neureptrace.metrics import validate_sample_weight


def _flag(value: int) -> object:
    return np.asarray(value == 1).item()

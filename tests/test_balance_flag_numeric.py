from __future__ import annotations

import numpy as np

from neureptrace._source_selection_class_balance_patch import _normalize_bool


def test_balance_flag_accepts_numeric_scalars() -> None:
    assert _normalize_bool(0.0, name="class_balance") is False
    assert _normalize_bool(np.float64(0.0), name="class_balance") is False
    assert _normalize_bool(np.array(0.0), name="class_balance") is False
    assert _normalize_bool(1.0, name="class_balance") is True
    assert _normalize_bool(np.float64(1.0), name="class_balance") is True
    assert _normalize_bool(np.array(1.0), name="class_balance") is True

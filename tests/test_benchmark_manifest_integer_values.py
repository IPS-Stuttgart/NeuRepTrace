from __future__ import annotations

import pandas as pd
import pytest

from neureptrace import benchmark


def test_benchmark_manifest_integer_values_accept_integer_like_strings():
    row = pd.Series({"n_splits": "3.0"})

    assert benchmark._int_value(row, "n_splits", 5) == 3


def test_benchmark_manifest_integer_values_use_default_for_missing_values():
    row = pd.Series({"subject": "sub-01"})

    assert benchmark._int_value(row, "n_splits", 5) == 5


def test_benchmark_manifest_integer_values_reject_fractional_strings():
    row = pd.Series({"n_splits": "2.5"})

    with pytest.raises(ValueError) as excinfo:
        benchmark._int_value(row, "n_splits", 5)

    message = str(excinfo.value)
    assert "n_splits" in message
    assert "integer-valued" in message

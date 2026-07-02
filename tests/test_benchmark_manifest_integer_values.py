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


@pytest.mark.parametrize("value", ["1", "true", "True", "t", "yes", "Y", "on"])
def test_benchmark_manifest_bool_values_accept_true_aliases(value):
    row = pd.Series({"tune_hyperparameters": value})

    assert benchmark._bool_value(row, "tune_hyperparameters") is True


@pytest.mark.parametrize("value", ["0", "false", "False", "f", "no", "N", "off"])
def test_benchmark_manifest_bool_values_accept_false_aliases(value):
    row = pd.Series({"tune_hyperparameters": value})

    assert benchmark._bool_value(row, "tune_hyperparameters", default=True) is False


def test_benchmark_manifest_bool_values_use_default_for_missing_values():
    row = pd.Series({"subject": "sub-01"})

    assert benchmark._bool_value(row, "tune_hyperparameters", default=True) is True


def test_benchmark_manifest_bool_values_reject_unknown_strings():
    row = pd.Series({"tune_hyperparameters": "tru"})

    with pytest.raises(ValueError) as excinfo:
        benchmark._bool_value(row, "tune_hyperparameters")

    message = str(excinfo.value)
    assert "tune_hyperparameters" in message
    assert "boolean value" in message

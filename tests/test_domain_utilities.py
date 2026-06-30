from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.domain_invariance import domain_risk_summary
from neureptrace.decoding.domain_subsets import domain_subsets


def test_domain_risk_summary_accepts_tuple_ids() -> None:
    summary = domain_risk_summary(
        [1.0, 3.0, 2.0, 4.0],
        [("a", "x"), ("a", "x"), ("b", "y"), ("b", "y")],
    )

    assert summary["domain_risks"][("a", "x")] == 2.0
    assert summary["domain_risks"][("b", "y")] == 3.0


def test_domain_subsets_accepts_tuple_ids() -> None:
    rows = domain_subsets(
        [("a", "x"), ("a", "x"), ("b", "y"), ("b", "y")],
        subset_size=1,
    )

    assert len(rows) == 2
    assert rows[0][0] == (("a", "x"),)
    assert rows[0][1].tolist() == [True, True, False, False]
    assert rows[1][0] == (("b", "y"),)
    assert rows[1][1].tolist() == [False, False, True, True]


def test_domain_subsets_accepts_string_integer_subset_size() -> None:
    rows = domain_subsets(["a", "a", "b", "b"], subset_size="1")

    assert len(rows) == 2


def test_domain_subsets_accepts_zero_dimensional_numeric_array_subset_size() -> None:
    rows = domain_subsets(["a", "a", "b", "b"], subset_size=np.asarray(1))

    assert len(rows) == 2


@pytest.mark.parametrize(
    "invalid_size",
    [
        True,
        False,
        1.5,
        "1.5",
        np.nan,
        np.inf,
        "all",
        np.asarray(True),
        np.asarray(False),
        np.asarray([1]),
        np.asarray([[1]]),
    ],
)
def test_domain_subsets_rejects_non_integral_subset_size(invalid_size) -> None:
    with pytest.raises(ValueError, match="subset_size"):
        domain_subsets(["a", "a", "b", "b"], subset_size=invalid_size)

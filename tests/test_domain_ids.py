from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.decoding._domain_ids import atomic_domain_vector, domain_mask, ordered_unique, values_equal


def test_atomic_domain_vector_preserves_singleton_axis_composite_rows() -> None:
    domains = np.asarray(
        [
            [["sub-01", "run-01"]],
            [["sub-02", "run-01"]],
            [["sub-01", "run-01"]],
        ],
        dtype=object,
    )

    vector = atomic_domain_vector(domains)

    assert vector.dtype == object
    assert vector.tolist() == [
        ("sub-01", "run-01"),
        ("sub-02", "run-01"),
        ("sub-01", "run-01"),
    ]
    assert ordered_unique(domains) == (("sub-01", "run-01"), ("sub-02", "run-01"))
    assert domain_mask(domains, [("sub-01", "run-01")]).tolist() == [True, False, True]


def test_atomic_domain_vector_keeps_column_vectors_as_scalar_domains() -> None:
    domains = np.asarray([["sub-01"], ["sub-02"], ["sub-01"]], dtype=object)

    vector = atomic_domain_vector(domains)

    assert vector.tolist() == ["sub-01", "sub-02", "sub-01"]
    assert ordered_unique(domains) == ("sub-01", "sub-02")


def test_missing_scalar_domain_ids_group_and_select_consistently() -> None:
    domains = np.empty(4, dtype=object)
    domains[:] = [np.nan, "sub-01", float("nan"), pd.NA]

    unique = ordered_unique(domains)

    assert len(unique) == 2
    assert values_equal(unique[0], np.nan)
    assert unique[1] == "sub-01"
    assert domain_mask(domains, [pd.NA]).tolist() == [True, False, True, True]
    assert values_equal(pd.NaT, np.datetime64("NaT"))


def test_missing_values_inside_composite_domain_ids_are_reflexive() -> None:
    domains = [(np.nan, "run-01"), ("sub-01", "run-01"), (pd.NA, "run-01")]

    unique = ordered_unique(domains)

    assert len(unique) == 2
    assert values_equal(unique[0], (pd.NA, "run-01"))
    assert unique[1] == ("sub-01", "run-01")
    assert domain_mask(domains, [(np.nan, "run-01")]).tolist() == [True, False, True]

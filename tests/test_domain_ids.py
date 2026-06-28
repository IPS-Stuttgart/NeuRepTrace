from __future__ import annotations

import numpy as np

from neureptrace.decoding._domain_ids import atomic_domain_vector, domain_mask, ordered_unique


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

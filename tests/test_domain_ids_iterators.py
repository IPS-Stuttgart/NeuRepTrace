from __future__ import annotations

import numpy as np

from neureptrace.decoding._domain_ids import atomic_domain_vector, domain_mask, hashable_domain_id, ordered_unique, values_equal


def _parts(subject: str):
    yield subject
    yield "run-01"


def test_atomic_domain_vector_materializes_outer_iterator() -> None:
    domains = (value for value in ["sub-01", "sub-02", "sub-01"])

    vector = atomic_domain_vector(domains)

    assert vector.tolist() == ["sub-01", "sub-02", "sub-01"]
    assert ordered_unique(value for value in ["sub-01", "sub-02", "sub-01"]) == ("sub-01", "sub-02")


def test_nested_iterator_domain_ids_are_stable_composite_values() -> None:
    domains = [_parts("sub-01"), _parts("sub-02"), _parts("sub-01")]

    vector = atomic_domain_vector(domains)

    assert vector.tolist() == [
        ("sub-01", "run-01"),
        ("sub-02", "run-01"),
        ("sub-01", "run-01"),
    ]
    assert ordered_unique([_parts("sub-01"), _parts("sub-02"), _parts("sub-01")]) == (
        ("sub-01", "run-01"),
        ("sub-02", "run-01"),
    )


def test_object_array_iterator_cells_match_composite_selection() -> None:
    domains = np.empty(3, dtype=object)
    domains[:] = [_parts("sub-01"), _parts("sub-02"), _parts("sub-01")]

    mask = domain_mask(domains, [_parts("sub-01")])

    assert mask.tolist() == [True, False, True]


def test_iterator_composites_compare_and_hash_by_contents() -> None:
    assert values_equal(_parts("sub-01"), ("sub-01", "run-01"))
    assert hashable_domain_id(_parts("sub-01")) == ("sub-01", "run-01")


def test_domain_mask_materializes_outer_selection_iterator_once() -> None:
    domains = [("sub-01", "run-01"), ("sub-02", "run-01"), ("sub-01", "run-01")]
    selected = (value for value in [("sub-01", "run-01")])

    mask = domain_mask(domains, selected)

    assert mask.tolist() == [True, False, True]

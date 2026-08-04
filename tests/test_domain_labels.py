from __future__ import annotations

from neureptrace.decoding._domain_labels import _as_domain_vector, _domain_mask, _unique_domain_labels


def test_as_domain_vector_preserves_expected_single_tuple_domain() -> None:
    vector = _as_domain_vector(("subject", "run"), expected_length=1)

    assert vector.dtype == object
    assert vector.tolist() == [("subject", "run")]
    assert _unique_domain_labels(vector) == (("subject", "run"),)
    assert _domain_mask(vector, ("subject", "run")).tolist() == [True]


def test_as_domain_vector_keeps_flat_sequences_as_rows_without_expected_length() -> None:
    vector = _as_domain_vector(("first", "second"))

    assert vector.tolist() == ["first", "second"]


def test_as_domain_vector_materializes_nested_one_pass_composite_labels() -> None:
    rows = (
        (part for part in ("subject-01", "run-01")),
        (part for part in ("subject-01", "run-01")),
        (part for part in ("subject-02", "run-01")),
    )

    vector = _as_domain_vector((row for row in rows), expected_length=3)

    assert vector.tolist() == [
        ("subject-01", "run-01"),
        ("subject-01", "run-01"),
        ("subject-02", "run-01"),
    ]
    assert _unique_domain_labels(vector) == (
        ("subject-01", "run-01"),
        ("subject-02", "run-01"),
    )
    assert _domain_mask(vector, ("subject-01", "run-01")).tolist() == [True, True, False]

from __future__ import annotations

from neureptrace.observations import stable_hash


def _iterator_payload():
    return {
        "pipeline": (step for step in ("filter", "pca")),
        "folds": iter((1, 2, 3)),
    }


def test_stable_hash_canonicalizes_equivalent_iterator_payloads() -> None:
    canonical = {"pipeline": ["filter", "pca"], "folds": [1, 2, 3]}

    assert stable_hash(_iterator_payload(), length=64) == stable_hash(canonical, length=64)
    assert stable_hash(_iterator_payload(), length=64) == stable_hash(canonical, length=64)


def test_stable_hash_canonicalizes_nested_iterator_values() -> None:
    payload = {
        "stages": [
            (value for value in (1, 2)),
            {"classes": map(str, (0, 1))},
        ]
    }
    canonical = {"stages": [[1, 2], {"classes": ["0", "1"]}]}

    assert stable_hash(payload, length=64) == stable_hash(canonical, length=64)

from __future__ import annotations

import hashlib
import json

from neureptrace.observations import stable_hash


def _legacy_hash(payload: object, *, length: int = 16) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:length]


def test_stable_hash_preserves_json_native_payload_hashes() -> None:
    payload = {"channels": ["MEG0111", "MEG0121"], "options": {"baseline": True, "seed": 13}}

    assert stable_hash(payload) == _legacy_hash(payload)


def test_stable_hash_canonicalizes_set_payloads() -> None:
    payload = {"channels": {"MEG0141", "MEG0121", "MEG0111", "MEG0131"}}
    canonical = {"channels": ["MEG0111", "MEG0121", "MEG0131", "MEG0141"]}
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")

    assert stable_hash(payload, length=64) == hashlib.sha256(encoded).hexdigest()


def test_stable_hash_canonicalizes_nested_frozensets() -> None:
    first = {"groups": {frozenset(("left", "index")), frozenset(("right", "middle"))}}
    second = {"groups": {frozenset(("middle", "right")), frozenset(("index", "left"))}}

    assert stable_hash(first) == stable_hash(second)

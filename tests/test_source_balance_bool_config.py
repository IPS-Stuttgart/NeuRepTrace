from __future__ import annotations

from neureptrace.decoding.source_balance import source_balance_config


def test_source_balance_normalize_weights_parses_string_false() -> None:
    assert source_balance_config(normalize_weights="false").normalize_weights is False

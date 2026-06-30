from __future__ import annotations

from neureptrace.decoding.source_rff import source_rff_config


def test_source_rff_accepts_null_random_state_string() -> None:
    cfg = source_rff_config(random_state="null")

    assert cfg.random_state is None

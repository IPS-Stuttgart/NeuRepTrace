from __future__ import annotations

from neureptrace.decoding.source_rff import source_rff_config


def test_source_rff_accepts_text_none_random_state() -> None:
    cfg = source_rff_config(random_state="NULL")

    assert cfg.random_state is None

from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_rff import SourceRFFConfig, fit_source_rff_reference, source_rff_config


@pytest.mark.parametrize("seed", ["null", "NULL", np.asarray("null")])
def test_source_rff_config_accepts_null_text_random_state(seed: object) -> None:
    cfg = source_rff_config(random_state=seed)  # type: ignore[arg-type]

    assert cfg.random_state is None


@pytest.mark.parametrize("seed", ["null", np.asarray("NULL")])
def test_source_rff_direct_config_accepts_null_text_random_state(seed: object) -> None:
    cfg = SourceRFFConfig(random_state=seed)  # type: ignore[arg-type]
    reference = fit_source_rff_reference([[0.0], [1.0]], config=cfg)

    assert cfg.random_state is None
    assert reference.config.random_state is None
    assert reference.weights.shape[0] == 1

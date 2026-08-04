from __future__ import annotations

import numpy as np
import pytest

from neureptrace.features.oscillatory import summarize_analytic_window


@pytest.mark.parametrize(
    "eps",
    [
        0.0,
        -1e-12,
        np.inf,
        np.nan,
        True,
        1e-12 + 0.0j,
        np.asarray([1e-12]),
    ],
)
def test_log_power_rejects_invalid_epsilon(eps: object) -> None:
    analytic_window = np.asarray([0.0j, 1.0 + 0.0j])

    with pytest.raises(ValueError, match="eps must be a positive finite real scalar"):
        summarize_analytic_window(
            analytic_window,
            outputs=("log_power",),
            eps=eps,  # type: ignore[arg-type]
        )


def test_log_power_accepts_numpy_real_scalar_epsilon() -> None:
    analytic_window = np.asarray([0.0j, 1.0 + 0.0j])

    summary = summarize_analytic_window(
        analytic_window,
        outputs=("log_power",),
        eps=np.float64(1e-12),
    )

    assert np.isfinite(summary["log_power"])


def test_unused_epsilon_does_not_block_other_outputs() -> None:
    summary = summarize_analytic_window(
        np.asarray([1.0 + 0.0j]),
        outputs=("mean_power",),
        eps=0.0,
    )

    assert summary == {"mean_power": 1.0}

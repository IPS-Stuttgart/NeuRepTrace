import numpy as np

from neureptrace.decoding.domain_invariance import domain_risk_summary


def test_domain_risk_summary_stays_finite_for_equal_float64_max_losses() -> None:
    maximum = np.finfo(float).max

    with np.errstate(over="raise", invalid="raise"):
        summary = domain_risk_summary(
            [maximum, maximum, maximum, maximum],
            ["a", "a", "b", "b"],
        )

    assert summary["domain_risks"] == {"a": maximum, "b": maximum}
    assert summary["mean_risk"] == maximum
    assert summary["risk_variance"] == 0.0


def test_domain_risk_summary_preserves_ordinary_results() -> None:
    summary = domain_risk_summary([1.0, 3.0, 2.0, 4.0], ["a", "a", "b", "b"])

    assert summary["domain_risks"] == {"a": 2.0, "b": 3.0}
    assert summary["mean_risk"] == 2.5
    assert summary["risk_variance"] == 0.25

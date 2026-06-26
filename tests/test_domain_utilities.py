from __future__ import annotations

from neureptrace.decoding.domain_invariance import domain_risk_summary


def test_domain_risk_summary_accepts_tuple_ids() -> None:
    summary = domain_risk_summary(
        [1.0, 3.0, 2.0, 4.0],
        [("a", "x"), ("a", "x"), ("b", "y"), ("b", "y")],
    )

    assert summary["domain_risks"][("a", "x")] == 2.0
    assert summary["domain_risks"][("b", "y")] == 3.0

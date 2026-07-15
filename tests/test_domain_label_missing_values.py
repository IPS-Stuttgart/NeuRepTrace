import numpy as np
import pytest

from neureptrace.decoding._domain_labels import _as_domain_vector, _domain_mask, _unique_domain_labels
from neureptrace.decoding.domain_invariance import domain_risk_summary


def test_missing_domain_labels_form_one_group_and_match_rows():
    domains = _as_domain_vector([float("nan"), np.float64("nan"), "site-b"], expected_length=3)

    levels = _unique_domain_labels(domains)

    assert len(levels) == 2
    np.testing.assert_array_equal(_domain_mask(domains, levels[0]), [True, True, False])
    np.testing.assert_array_equal(_domain_mask(domains, levels[1]), [False, False, True])


def test_domain_risk_summary_handles_repeated_missing_domains():
    summary = domain_risk_summary(
        [1.0, 3.0, 5.0],
        [float("nan"), np.float64("nan"), "site-b"],
    )

    assert len(summary["domain_risks"]) == 2
    assert sorted(summary["domain_risks"].values()) == pytest.approx([2.0, 5.0])
    assert summary["mean_risk"] == pytest.approx(3.5)
    assert summary["risk_variance"] == pytest.approx(2.25)

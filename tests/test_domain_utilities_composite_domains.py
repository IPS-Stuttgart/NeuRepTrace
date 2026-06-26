from __future__ import annotations

import numpy as np

from neureptrace.decoding.domain_invariance import domain_risk_summary
from neureptrace.decoding.domain_subsets import domain_subsets


def test_domain_risk_summary_treats_matrix_rows_as_composite_domains():
    losses = np.asarray([1.0, 5.0, 2.0, 8.0], dtype=float)
    domains = np.asarray(
        [
            ["subject-a", "run-1"],
            ["subject-a", "run-1"],
            ["subject-b", "run-1"],
            ["subject-b", "run-2"],
        ],
        dtype=object,
    )

    summary = domain_risk_summary(losses, domains)

    assert set(summary["domain_risks"]) == {("subject-a", "run-1"), ("subject-b", "run-1"), ("subject-b", "run-2")}
    assert np.isclose(summary["domain_risks"][("subject-a", "run-1")], 3.0)
    assert np.isclose(summary["domain_risks"][("subject-b", "run-1")], 2.0)
    assert np.isclose(summary["domain_risks"][("subject-b", "run-2")], 8.0)
    assert summary["uses_target_features"] is False
    assert summary["uses_target_labels"] is False


def test_domain_subsets_treats_matrix_rows_as_composite_domains():
    domains = np.asarray(
        [
            ["subject-a", "run-1"],
            ["subject-a", "run-2"],
            ["subject-b", "run-1"],
            ["subject-a", "run-1"],
        ],
        dtype=object,
    )

    subsets = domain_subsets(domains, subset_size=1)

    labels = tuple(selected[0] for selected, _ in subsets)
    assert labels == (("subject-a", "run-1"), ("subject-a", "run-2"), ("subject-b", "run-1"))
    masks_by_label = {selected[0]: mask for selected, mask in subsets}
    np.testing.assert_array_equal(masks_by_label[("subject-a", "run-1")], np.asarray([True, False, False, True]))
    np.testing.assert_array_equal(masks_by_label[("subject-a", "run-2")], np.asarray([False, True, False, False]))
    np.testing.assert_array_equal(masks_by_label[("subject-b", "run-1")], np.asarray([False, False, True, False]))

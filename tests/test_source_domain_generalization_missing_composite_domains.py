from __future__ import annotations

import numpy as np

from neureptrace._source_domain_generalization_composite_patch import _is_missing_domain_array


def test_missing_domain_array_reports_one_flag_per_composite_row() -> None:
    domains = np.asarray(
        [
            ["sub-01", "run-1"],
            ["sub-02", "run-2"],
            ["sub-03", "run-3"],
        ],
        dtype=object,
    )

    mask = _is_missing_domain_array(domains)

    assert mask.tolist() == [False, False, False]


def test_missing_domain_array_marks_row_with_missing_composite_component() -> None:
    domains = np.asarray(
        [
            ["sub-01", "run-1"],
            ["sub-02", np.nan],
            ["sub-03", "run-3"],
        ],
        dtype=object,
    )

    mask = _is_missing_domain_array(domains)

    assert mask.tolist() == [False, True, False]

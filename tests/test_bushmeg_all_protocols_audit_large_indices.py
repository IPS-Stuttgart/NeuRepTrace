from __future__ import annotations

import pandas as pd

from neureptrace.bushmeg_all_protocols_audit import _parse_index_set, _protocol3_prediction_overlap_failures


def _protocol3_summary(calibration_index: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protocol_category": 3,
                "method": "few_shot",
                "outer_test_subject": "sub-01",
                "fold_index": 1,
                "target_calibration_per_class": 1,
                "target_calibration_indices": str(calibration_index),
            }
        ]
    )


def _protocol3_predictions(target_row_index: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protocol_category": 3,
                "method": "few_shot",
                "outer_test_subject": "sub-01",
                "fold_index": 1,
                "target_calibration_per_class": 1,
                "target_row_index": target_row_index,
            }
        ]
    )


def test_protocol3_overlap_audit_keeps_adjacent_large_indices_distinct() -> None:
    lower_index = 2**53
    higher_index = lower_index + 1

    failures = _protocol3_prediction_overlap_failures(
        _protocol3_summary(lower_index),
        _protocol3_predictions(higher_index),
    )

    assert failures == []


def test_protocol3_overlap_audit_reports_exact_large_index() -> None:
    index = 2**53 + 1

    failures = _protocol3_prediction_overlap_failures(
        _protocol3_summary(index),
        _protocol3_predictions(index),
    )

    assert len(failures) == 1
    assert f"calibration row {index}" in failures[0]


def test_protocol3_index_parser_preserves_exact_decimal_tokens() -> None:
    index = 2**53 + 1

    parsed = _parse_index_set(f"{index}|9.007199254740993e15|1.5|nan")

    assert parsed == {index}

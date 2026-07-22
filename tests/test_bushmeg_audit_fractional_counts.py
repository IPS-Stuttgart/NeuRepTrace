from __future__ import annotations

import pandas as pd
import pytest

import neureptrace.bushmeg_all_protocols_audit as audit


def _summary(*, k_per_class, n_classes, n_target_calibration_trials) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "protocol_category": 3,
                "method": "few_shot",
                "outer_test_subject": "sub-01",
                "fold_index": 0,
                "k_per_class": k_per_class,
                "n_classes": n_classes,
                "n_target_calibration_trials": n_target_calibration_trials,
            }
        ]
    )


@pytest.mark.parametrize(
    ("k_per_class", "n_classes", "n_target_calibration_trials"),
    [
        (1.5, 2, 2),
        (1, 2.5, 2),
        (1, 2, 2.5),
        (-1, 2, -2),
        (True, 2, 2),
    ],
)
def test_protocol3_audit_rejects_malformed_calibration_counts(
    k_per_class,
    n_classes,
    n_target_calibration_trials,
) -> None:
    failures = audit._protocol3_calibration_count_failures(
        _summary(
            k_per_class=k_per_class,
            n_classes=n_classes,
            n_target_calibration_trials=n_target_calibration_trials,
        )
    )

    assert len(failures) == 1
    assert "calibration counts must be finite integers" in failures[0]


def test_protocol3_audit_accepts_integral_calibration_counts() -> None:
    failures = audit._protocol3_calibration_count_failures(
        _summary(
            k_per_class=[2, 2],
            n_classes=[3, 3],
            n_target_calibration_trials=[6, 6],
        )
    )

    assert failures == []

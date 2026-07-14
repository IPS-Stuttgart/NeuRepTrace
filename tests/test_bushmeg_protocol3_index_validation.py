from __future__ import annotations

import numpy as np
import pytest

from neureptrace.bushmeg_all_protocols import (
    select_bushmeg_target_calibration_split,
    validate_disjoint_calibration_evaluation,
    validate_protocol_input_use,
)


def test_protocol3_rejects_matrix_shaped_split_indices() -> None:
    with pytest.raises(ValueError, match="calibration_indices must be a one-dimensional index vector"):
        validate_disjoint_calibration_evaluation([[0, 1], [2, 3]], [4, 5, 6, 7])

    with pytest.raises(ValueError, match="evaluation_indices must be a one-dimensional index vector"):
        validate_protocol_input_use(
            3,
            target_features_for_fitting=True,
            target_labels_for_fitting=True,
            calibration_indices=[0, 1, 2, 3],
            evaluation_indices=np.asarray([[4, 5], [6, 7]]),
        )


def test_protocol3_split_validator_accepts_column_and_row_vectors() -> None:
    validate_disjoint_calibration_evaluation(np.asarray([[0], [1]]), np.asarray([[2, 3]]))


def test_protocol3_rejects_non_integer_split_indices() -> None:
    with pytest.raises(ValueError, match="calibration_indices must contain integer row indices"):
        validate_disjoint_calibration_evaluation([0.0, 1.5], [2, 3])

    with pytest.raises(ValueError, match="evaluation_indices must contain integer row indices, not boolean values"):
        validate_disjoint_calibration_evaluation([0, 1], [False, True])


@pytest.mark.parametrize(
    ("option_name", "option_value"),
    [
        ("per_class", np.asarray([True])),
        ("seed", [np.bool_(False)]),
        ("min_evaluation_per_class", np.asarray([[True]])),
    ],
)
def test_protocol3_target_split_rejects_boolean_array_split_options(option_name: str, option_value: object) -> None:
    kwargs = {
        "per_class": 1,
        "seed": 7,
        "min_evaluation_per_class": 1,
    }
    kwargs[option_name] = option_value

    with pytest.raises(ValueError, match=f"{option_name} must be an integer value, not a boolean value"):
        select_bushmeg_target_calibration_split(["a", "a", "b", "b"], **kwargs)


def test_protocol3_target_split_preserves_composite_tuple_labels() -> None:
    labels = [
        ("face", "left"),
        ("face", "left"),
        ("face", "right"),
        ("face", "right"),
    ]

    split = select_bushmeg_target_calibration_split(
        labels,
        per_class=1,
        seed=7,
        min_evaluation_per_class=1,
    )

    assert not split.skipped
    assert split.n_classes == 2
    assert split.n_target_calibration_trials == 2
    assert split.n_target_evaluation_trials == 2
    assert split.calibration_rows_disjoint_from_evaluation

    calibration_labels = [labels[int(index)] for index in split.calibration_indices]
    evaluation_labels = [labels[int(index)] for index in split.evaluation_indices]
    assert sorted(calibration_labels) == [("face", "left"), ("face", "right")]
    assert sorted(evaluation_labels) == [("face", "left"), ("face", "right")]


def test_protocol3_target_split_reports_composite_label_class_counts() -> None:
    labels = [("face", "left"), ("face", "left"), ("face", "right")]

    split = select_bushmeg_target_calibration_split(
        labels,
        per_class=1,
        seed=7,
        min_evaluation_per_class=1,
    )

    assert split.skipped
    assert split.skip_reason_code == "insufficient_rows_per_class"
    assert "('face', 'right')" in split.skip_reason

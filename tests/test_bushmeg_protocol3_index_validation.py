from __future__ import annotations

import numpy as np
import pytest

from neureptrace.bushmeg_all_protocols import (
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

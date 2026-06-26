from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.mcca import class_alignment_matrices, fit_class_mcca
from neureptrace.decoding.mcca_target import class_alignment_matrix


def _features_by_subject() -> dict[str, np.ndarray]:
    return {
        "a": np.array([[1.0], [2.0], [10.0], [20.0]]),
        "b": np.array([[101.0], [102.0], [110.0], [120.0]]),
    }


def _labels_by_subject() -> dict[str, np.ndarray]:
    return {
        "a": np.array([1, 1, 2, 2]),
        "b": np.array([1, 1, 2, 2]),
    }


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_class_alignment_matrices_reject_boolean_repetition_counts(value: object) -> None:
    with pytest.raises(ValueError, match="n_repetitions_per_class"):
        class_alignment_matrices(
            _features_by_subject(),
            _labels_by_subject(),
            sample_mode="class_repetition",
            n_repetitions_per_class=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_fit_class_mcca_rejects_boolean_repetition_counts(value: object) -> None:
    with pytest.raises(ValueError, match="n_repetitions_per_class"):
        fit_class_mcca(
            _features_by_subject(),
            _labels_by_subject(),
            sample_mode="class_repetition",
            n_repetitions_per_class=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_target_class_alignment_matrix_rejects_boolean_repetition_counts(value: object) -> None:
    features = np.array([[1.0], [2.0], [10.0], [20.0]])
    labels = np.array([1, 1, 2, 2])

    with pytest.raises(ValueError, match="n_repetitions_per_class"):
        class_alignment_matrix(
            features,
            labels,
            sample_mode="class_repetition",
            n_repetitions_per_class=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [True, np.bool_(True)])
def test_target_class_alignment_matrix_rejects_boolean_repetition_counts_with_selected_offsets(value: object) -> None:
    features = np.array([[1.0], [2.0], [10.0], [20.0]])
    labels = np.array([1, 1, 2, 2])

    with pytest.raises(ValueError, match="n_repetitions_per_class"):
        class_alignment_matrix(
            features,
            labels,
            sample_mode="class_repetition",
            n_repetitions_per_class=value,  # type: ignore[arg-type]
            selected_offsets_by_class={0: np.array([0]), 1: np.array([0])},
        )

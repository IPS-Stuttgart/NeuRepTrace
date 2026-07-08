from __future__ import annotations

import numpy as np
import pytest

import neureptrace  # noqa: F401  # importing the package installs runtime patches
from neureptrace.decoding.source_free import SourceFreeSubjectAdapter


class _ShortProbabilitySourceModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        n_rows = max(int(np.asarray(features).shape[0]) - 1, 0)
        return np.tile(np.asarray([[0.75, 0.25]], dtype=float), (n_rows, 1))


class _ShortDecisionSourceModel:
    classes_ = np.asarray([0, 1])

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        n_rows = max(int(np.asarray(features).shape[0]) - 1, 0)
        return np.zeros(n_rows, dtype=float)


@pytest.mark.parametrize(
    "source_model",
    [_ShortProbabilitySourceModel(), _ShortDecisionSourceModel()],
)
def test_source_free_rejects_source_model_row_count_mismatch(source_model) -> None:
    target_features = np.zeros((3, 2), dtype=float)

    with pytest.raises(ValueError, match="one row per feature row"):
        SourceFreeSubjectAdapter(source_model=source_model, max_iterations=0).fit(target_features)

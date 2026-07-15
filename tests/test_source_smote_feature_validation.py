from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_smote import augment_source_with_smote, interpolate_rows


def test_source_smote_rejects_boolean_feature_matrix() -> None:
    features = np.asarray([[True, False], [False, True]], dtype=bool)
    labels = np.asarray([0, 1])

    with pytest.raises(ValueError, match="source_features must contain numeric, non-boolean values"):
        augment_source_with_smote(features, labels)


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_source_smote_interpolation_rejects_nonfinite_rows(invalid_value: float) -> None:
    with pytest.raises(ValueError, match="content_row and partner_row must contain only finite values"):
        interpolate_rows([0.0, invalid_value], [1.0, 2.0], 0.5)


def test_source_smote_interpolation_rejects_boolean_rows() -> None:
    with pytest.raises(ValueError, match="content_row and partner_row must contain numeric, non-boolean values"):
        interpolate_rows([True, False], [False, True], 0.5)

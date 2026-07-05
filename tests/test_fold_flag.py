from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.fold_flag import ABS_FEATURE_CATEGORY, absolute_value_features


def test_absolute_value_features_returns_abs_and_metadata() -> None:
    transformed, metadata = absolute_value_features([[-1.0, 2.0], [3.0, -4.0]])

    assert np.allclose(transformed, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert metadata["abs_feature_protocol_category"] == ABS_FEATURE_CATEGORY
    assert metadata["abs_feature_has_fitted_parameters"] is False
    assert metadata["abs_feature_uses_labels"] is False
    assert metadata["abs_feature_valid_for_strict_source_only"] is True


def test_absolute_value_features_rejects_non_matrix() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        absolute_value_features([1.0, -2.0])

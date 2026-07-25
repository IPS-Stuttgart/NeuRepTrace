from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.source_knn import fit_source_knn_decoder


def test_source_knn_rejects_complex_source_features_before_float_coercion() -> None:
    source = np.asarray(
        [[0.0 + 1.0j], [1.0 + 0.0j], [10.0 + 0.0j], [11.0 + 0.0j]],
        dtype=np.complex128,
    )

    with pytest.raises(ValueError, match="source_features must contain real-valued feature values"):
        fit_source_knn_decoder(
            source_features=source,
            source_labels=["left", "left", "right", "right"],
            test_features=[[0.5]],
            config={"k": 1},
        )


def test_source_knn_rejects_complex_test_features_before_float_coercion() -> None:
    test = np.asarray([[0.5 + 1.0j]], dtype=np.complex128)

    with pytest.raises(ValueError, match="test_features must contain real-valued feature values"):
        fit_source_knn_decoder(
            source_features=[[0.0], [1.0], [10.0], [11.0]],
            source_labels=["left", "left", "right", "right"],
            test_features=test,
            config={"k": 1},
        )

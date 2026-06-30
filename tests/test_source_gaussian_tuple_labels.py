from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_gaussian import fit_source_gaussian_decoder


def test_source_gaussian_preserves_tuple_labels() -> None:
    left_label = ("visual", "left")
    right_label = ("visual", "right")
    source_features = np.asarray([[-2.0], [-1.5], [1.5], [2.0]], dtype=float)
    source_labels = [left_label, left_label, right_label, right_label]
    test_features = np.asarray([[-1.75], [1.75]], dtype=float)

    result = fit_source_gaussian_decoder(
        source_features=source_features,
        source_labels=source_labels,
        test_features=test_features,
        config={"covariance_type": "tied_spherical", "prior": "uniform"},
    )

    assert result.classes.tolist() == [left_label, right_label]
    assert result.predictions.tolist() == [left_label, right_label]
    assert result.probabilities.shape == (2, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)

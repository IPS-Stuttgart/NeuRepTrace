from __future__ import annotations

import numpy as np

from neureptrace.decoding.source_smote import augment_source_with_smote


def test_disabled_source_smote_respects_preserve_original_false() -> None:
    features = np.asarray([[1.0], [2.0]], dtype=float)
    labels = np.asarray(["a", "b"], dtype=object)

    result = augment_source_with_smote(
        features,
        labels,
        config={"synthetic_per_class": 0, "preserve_original": False},
    )

    assert result.features.shape == (0, 1)
    assert result.labels.shape == (0,)
    assert result.synthetic_mask.shape == (0,)
    assert result.content_indices.shape == (0,)
    assert result.partner_indices.shape == (0,)
    assert result.lambdas.shape == (0,)
    assert result.n_synthetic == 0
    assert result.metadata["source_smote"] is False
    assert result.metadata["source_smote_preserve_original"] is False
    assert result.metadata["source_smote_n_output_rows"] == 0

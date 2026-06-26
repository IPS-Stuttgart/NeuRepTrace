from __future__ import annotations

import numpy as np

from neureptrace.decoding.conditional_coral import fit_conditional_coral_alignment


def test_conditional_coral_smoke() -> None:
    result = fit_conditional_coral_alignment(
        source_features=np.asarray([[0.0], [0.1], [2.0], [2.1]], dtype=float),
        source_labels=["class_a", "class_a", "class_b", "class_b"],
        target_features=np.asarray([[5.0], [5.1], [8.0], [8.1]], dtype=float),
        target_pseudo_labels=["class_a", "class_a", "class_b", "class_b"],
    )
    assert result.train_features.shape == (4, 1)
    assert result.test_features.shape == (4, 1)

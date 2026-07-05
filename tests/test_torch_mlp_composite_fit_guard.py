from __future__ import annotations

import numpy as np

from neureptrace import _torch_weight_validation_patch as torch_weight_patch


def test_torch_mlp_fit_guard_dense_encodes_composite_labels_and_restores_classes() -> None:
    class TorchMLPClassifier:
        def __init__(self):
            self.class_weight = "balanced"
            self.validation_fraction = 0.5
            self.seen_labels = None

        def fit(self, features, labels):
            self.seen_labels = np.asarray(labels)
            self.classes_ = np.unique(labels)
            return self

    torch_weight_patch._install_fit_guard(TorchMLPClassifier, "class_weight")

    model = TorchMLPClassifier()
    labels = [
        ("cue", "left"),
        ("cue", "left"),
        ("cue", "right"),
        ("cue", "right"),
    ]

    result = model.fit(np.zeros((4, 1), dtype=float), labels)

    assert result is model
    assert model.seen_labels.tolist() == [0, 0, 1, 1]
    assert model.classes_.dtype == object
    assert model.classes_.tolist() == [("cue", "left"), ("cue", "right")]
    assert model.validation_fraction == 0.5

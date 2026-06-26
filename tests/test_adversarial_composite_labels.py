from __future__ import annotations

import numpy as np

from neureptrace._adversarial_composite_labels_patch import (
    _PATCH_MARKER,
    _encode_atomic_labels,
    _install_fit_wrapper,
)
from neureptrace.decoding.cdan import TorchCDANClassifier
from neureptrace.decoding.dann import TorchDANNClassifier


def test_adversarial_label_encoder_preserves_tuple_labels() -> None:
    labels = [("face", "left"), ("house", "right"), ("face", "left"), ("house", "right")]

    classes, encoded = _encode_atomic_labels(labels, expected_length=4, name="source_labels")

    assert classes.shape == (2,)
    assert classes.tolist() == [("face", "left"), ("house", "right")]
    assert encoded.tolist() == [0, 1, 0, 1]


def test_adversarial_label_encoder_keeps_numpy_tuple_rows_atomic() -> None:
    labels = np.asarray([("face", "left"), ("house", "right"), ("face", "left")], dtype=object)

    classes, encoded = _encode_atomic_labels(labels, expected_length=3, name="source_labels")

    assert classes.tolist() == [("face", "left"), ("house", "right")]
    assert encoded.tolist() == [0, 1, 0]


def test_fit_wrapper_passes_integer_labels_and_restores_classes() -> None:
    class DummyClassifier:
        def fit(self, source_features, source_labels, *, target_features):
            self.source_labels_seen_ = np.asarray(source_labels)
            self.target_features_seen_ = np.asarray(target_features)
            self.classes_ = np.unique(self.source_labels_seen_)
            return self

    _install_fit_wrapper(DummyClassifier, label_name="dummy source_labels")
    model = DummyClassifier()
    labels = [("face", "left"), ("house", "right"), ("face", "left")]

    result = model.fit(np.zeros((3, 2)), labels, target_features=np.zeros((2, 2)))

    assert result is model
    assert model.source_labels_seen_.tolist() == [0, 1, 0]
    assert model.classes_.tolist() == [("face", "left"), ("house", "right")]
    assert model.target_features_seen_.shape == (2, 2)


def test_dann_and_cdan_fit_methods_are_composite_label_wrapped() -> None:
    assert getattr(TorchDANNClassifier.fit, _PATCH_MARKER, False) is True
    assert getattr(TorchCDANClassifier.fit, _PATCH_MARKER, False) is True

from __future__ import annotations

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    ("classifier_type", "estimator_name"),
    [
        (TorchDANNClassifier, "DANN"),
        (TorchCDANClassifier, "CDAN"),
    ],
)
@pytest.mark.parametrize("feature_role", ["source", "target"])
@pytest.mark.parametrize("complex_dtype", [np.complex128, object])
def test_adversarial_fit_rejects_complex_features_before_training(
    classifier_type: type,
    estimator_name: str,
    feature_role: str,
    complex_dtype: type,
) -> None:
    source_features = np.zeros((4, 2), dtype=float)
    target_features = np.zeros((2, 2), dtype=float)
    if feature_role == "source":
        source_features = np.asarray(
            [[1.0 + 2.0j, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=complex_dtype,
        )
    else:
        target_features = np.asarray(
            [[1.0 + 2.0j, 0.0], [0.0, 1.0]],
            dtype=complex_dtype,
        )

    model = classifier_type(max_epochs=1)
    with pytest.raises(
        ValueError,
        match=rf"{estimator_name} {feature_role}_features must contain real-valued features",
    ):
        model.fit(
            source_features,
            np.asarray([0, 1, 0, 1]),
            target_features=target_features,
        )

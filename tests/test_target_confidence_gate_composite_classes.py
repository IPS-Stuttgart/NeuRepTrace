from __future__ import annotations

from neureptrace.decoding.target_confidence_gate import gate_target_probabilities_by_confidence


def test_target_confidence_gate_preserves_composite_class_labels() -> None:
    classes = [("animal", "cat"), ("animal", "dog")]

    result = gate_target_probabilities_by_confidence(
        [[0.9, 0.1], [0.1, 0.9]],
        classes=classes,
    )

    assert result.classes.shape == (2,)
    assert result.classes.tolist() == classes
    assert result.predictions.shape == (2,)
    assert result.predictions.tolist() == classes

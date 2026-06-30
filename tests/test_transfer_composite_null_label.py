from __future__ import annotations

from neureptrace.decoding import transfer


def test_append_null_class_features_preserves_tuple_null_label_atoms() -> None:
    features, labels = transfer.append_null_class_features(
        [[1.0, 0.0], [0.0, 1.0]],
        [("stimulus", "left"), ("stimulus", "right")],
        [[0.5, 0.5], [0.25, 0.75]],
        null_label=("baseline", "null"),
    )

    assert features.tolist() == [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.25, 0.75]]
    assert labels.dtype == object
    assert labels.tolist() == [
        ("stimulus", "left"),
        ("stimulus", "right"),
        ("baseline", "null"),
        ("baseline", "null"),
    ]

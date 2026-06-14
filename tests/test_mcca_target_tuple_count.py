from neureptrace.decoding.mcca_target import _count_label


def test_count_label_preserves_plain_sequence_tuple_labels():
    labels = [
        ("run-01", "stim-a"),
        ("run-01", "stim-a"),
        ("run-01", "stim-b"),
    ]

    assert _count_label(labels, ("run-01", "stim-a")) == 2
    assert _count_label(labels, ("run-01", "stim-b")) == 1

from __future__ import annotations

import numpy as np
import pytest

from neureptrace.transfer_from_config import _encode_transfer_labels


def test_encode_transfer_labels_rejects_test_only_classes() -> None:
    raw_labels = np.asarray(["face", "face", "tool"], dtype=object)
    train_mask = np.asarray([True, True, False])
    test_mask = np.asarray([False, False, True])

    with pytest.raises(ValueError, match="absent from train_filter"):
        _encode_transfer_labels(raw_labels, train_mask, test_mask)


def test_encode_transfer_labels_rejects_single_class_transfer_task() -> None:
    raw_labels = np.asarray(["face", "face", "face"], dtype=object)
    train_mask = np.asarray([True, True, False])
    test_mask = np.asarray([False, False, True])

    with pytest.raises(ValueError, match="at least two labeled classes"):
        _encode_transfer_labels(raw_labels, train_mask, test_mask)


def test_encode_transfer_labels_allows_auxiliary_unlabeled_rows_outside_split() -> None:
    raw_labels = np.asarray(["face", "tool", None, "face"], dtype=object)
    train_mask = np.asarray([True, True, False, False])
    test_mask = np.asarray([False, False, False, True])

    encoder, labels, classes = _encode_transfer_labels(raw_labels, train_mask, test_mask)

    assert list(encoder.classes_) == ["face", "tool"]
    assert labels.tolist() == [0, 1, -1, 0]
    assert classes.tolist() == [0, 1]


def test_encode_transfer_labels_accepts_training_only_classes() -> None:
    raw_labels = np.asarray(["face", "tool", "face"], dtype=object)
    train_mask = np.asarray([True, True, False])
    test_mask = np.asarray([False, False, True])

    encoder, labels, classes = _encode_transfer_labels(raw_labels, train_mask, test_mask)

    assert list(encoder.classes_) == ["face", "tool"]
    assert labels.tolist() == [0, 1, 0]
    assert classes.tolist() == [0, 1]

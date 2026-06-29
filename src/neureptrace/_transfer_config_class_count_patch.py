"""Reject single-class configured-transfer tasks before decoder fitting."""

from __future__ import annotations

import importlib
from functools import wraps

import numpy as np

_PATCH_MARKER = "_neureptrace_transfer_config_class_count_patch_installed"


def install() -> None:
    """Install explicit class-count validation for configured-transfer labels."""

    transfer_from_config = importlib.import_module("neureptrace.transfer_from_config")
    original_encode = transfer_from_config._encode_transfer_labels
    if getattr(original_encode, _PATCH_MARKER, False):
        return

    @wraps(original_encode)
    def _encode_transfer_labels(raw_labels, train_mask, test_mask):
        encoder, labels, classes = original_encode(raw_labels, train_mask, test_mask)
        if len(classes) < 2:
            raise ValueError("transfer train_filter/test_filter must select at least two labeled classes.")

        train_mask_array = np.asarray(train_mask, dtype=bool)
        train_labels = labels[train_mask_array]
        train_classes = np.unique(train_labels[train_labels >= 0])
        if len(train_classes) < 2:
            raise ValueError("transfer train_filter must contain at least two labeled classes.")
        return encoder, labels, classes

    setattr(_encode_transfer_labels, _PATCH_MARKER, True)
    transfer_from_config._encode_transfer_labels = _encode_transfer_labels


__all__ = ["install"]

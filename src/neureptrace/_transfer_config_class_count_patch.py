"""Validate configured-transfer sections and class counts."""

from __future__ import annotations

import importlib
from functools import wraps

import numpy as np

_CLASS_COUNT_PATCH_MARKER = "_neureptrace_transfer_config_class_count_patch_installed"
_SECTION_PATCH_MARKER = "_neureptrace_transfer_section_precedence_patch_installed"


def install() -> None:
    """Install configured-transfer section and class-count validation."""

    transfer_from_config = importlib.import_module("neureptrace.transfer_from_config")

    original_section = transfer_from_config._transfer_section
    if not getattr(original_section, _SECTION_PATCH_MARKER, False):

        @wraps(original_section)
        def _transfer_section(config):
            if "transfer" in config:
                transfer = config["transfer"]
            else:
                transfer = config.get("workflow", {})
            if transfer is None:
                transfer = {}
            if not isinstance(transfer, dict):
                raise ValueError("Config section 'transfer' must be a mapping.")
            return dict(transfer)

        setattr(_transfer_section, _SECTION_PATCH_MARKER, True)
        transfer_from_config._transfer_section = _transfer_section

    original_encode = transfer_from_config._encode_transfer_labels
    if not getattr(original_encode, _CLASS_COUNT_PATCH_MARKER, False):

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

        setattr(_encode_transfer_labels, _CLASS_COUNT_PATCH_MARKER, True)
        transfer_from_config._encode_transfer_labels = _encode_transfer_labels


__all__ = ["install"]

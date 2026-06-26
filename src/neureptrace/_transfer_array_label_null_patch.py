"""Repair transfer null-class replacement for array-valued labels."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from neureptrace._object_label_utils import replace_null_class_predictions as _replace_object_label_null_predictions

_INSTALLED = False
_ORIGINAL_REPLACE_NULL_CLASS_PREDICTIONS = None


def install() -> None:
    """Install the transfer null-label replacement wrapper once."""

    global _INSTALLED, _ORIGINAL_REPLACE_NULL_CLASS_PREDICTIONS
    if _INSTALLED:
        return

    from neureptrace.decoding import transfer

    _ORIGINAL_REPLACE_NULL_CLASS_PREDICTIONS = transfer.replace_null_class_predictions

    def _replace_null_class_predictions(
        predictions: Sequence | np.ndarray,
        *,
        null_label: object = 0,
        fallback_label: object = 1,
    ) -> np.ndarray:
        """Replace predicted null labels while preserving atomic composite labels."""

        return _replace_object_label_null_predictions(
            transfer._prediction_vector(predictions),
            null_label=null_label,
            fallback_label=fallback_label,
        )

    _replace_null_class_predictions.__name__ = _ORIGINAL_REPLACE_NULL_CLASS_PREDICTIONS.__name__
    _replace_null_class_predictions.__doc__ = _ORIGINAL_REPLACE_NULL_CLASS_PREDICTIONS.__doc__
    transfer.replace_null_class_predictions = _replace_null_class_predictions
    _INSTALLED = True

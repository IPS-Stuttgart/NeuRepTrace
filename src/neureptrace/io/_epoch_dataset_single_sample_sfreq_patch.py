"""Reject unidentifiable sampling-rate inference for single-sample epochs."""

from __future__ import annotations

import importlib
from functools import wraps

_PATCH_MARKER = "_neureptrace_epoch_dataset_single_sample_sfreq_patch_installed"


def install() -> None:
    """Require at least two time samples before inferring a sampling rate."""

    dataset_module = importlib.import_module("neureptrace.io.dataset")
    if getattr(dataset_module, _PATCH_MARKER, False):
        return

    epoch_dataset = dataset_module.EpochDataset
    original_infer_sampling_frequency = epoch_dataset.infer_sampling_frequency

    @wraps(original_infer_sampling_frequency)
    def infer_sampling_frequency(self) -> float:
        if len(self.times) < 2:
            raise ValueError(
                "EpochDataset.times must contain at least two time samples to infer sampling frequency."
            )
        return original_infer_sampling_frequency(self)

    epoch_dataset.infer_sampling_frequency = infer_sampling_frequency
    setattr(dataset_module, _PATCH_MARKER, True)


__all__ = ["install"]

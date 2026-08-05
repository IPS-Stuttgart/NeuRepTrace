"""Runtime guard for explicit EpochDataset-to-MNE sampling rates."""

from __future__ import annotations

from functools import wraps

import numpy as np

_SFREQ_ERROR = "sfreq must be a positive finite scalar."
_SFREQ_MISMATCH_ERROR = "sfreq must match the uniformly sampled EpochDataset.times axis."


def _validate_sampling_frequency(value: object) -> float:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating, np.ndarray)):
        raise ValueError(_SFREQ_ERROR)
    try:
        sampling_frequency = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(_SFREQ_ERROR) from exc
    if not np.isfinite(sampling_frequency) or sampling_frequency <= 0.0:
        raise ValueError(_SFREQ_ERROR)
    return sampling_frequency


def install() -> None:
    """Reject explicit sampling rates that would change the dataset time axis."""
    from neureptrace.io.dataset import EpochDataset

    if getattr(EpochDataset.to_mne_epochs, "_epoch_dataset_sfreq_patched", False):
        return

    original_to_mne_epochs = EpochDataset.to_mne_epochs

    @wraps(original_to_mne_epochs)
    def to_mne_epochs(
        self: EpochDataset,
        *,
        channel_type: str | list[str] = "mag",
        sfreq: float | None = None,
    ):
        if sfreq is None:
            return original_to_mne_epochs(self, channel_type=channel_type, sfreq=None)

        validated_sfreq = _validate_sampling_frequency(sfreq)
        if len(self.times) >= 2:
            inferred_sfreq = self.infer_sampling_frequency()
            if not np.isclose(validated_sfreq, inferred_sfreq, rtol=1e-6, atol=1e-12):
                raise ValueError(_SFREQ_MISMATCH_ERROR)

        return original_to_mne_epochs(
            self,
            channel_type=channel_type,
            sfreq=validated_sfreq,
        )

    to_mne_epochs._epoch_dataset_sfreq_patched = True  # type: ignore[attr-defined]
    EpochDataset.to_mne_epochs = to_mne_epochs


__all__ = ["install"]

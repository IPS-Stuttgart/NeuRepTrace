"""Trial-set regularization for sequence-level calibrated decoding.

Some repeated-event tasks contain a known set of unique classes in each labeled
training trial, while the globally shared class head contains additional classes.
This module adds a soft class-mass objective without requiring a square
(event-by-class) assignment matrix and without accepting evaluation labels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from neureptrace.decoding._progressive_sequence_core import _torch
from neureptrace.decoding._progressive_sequence_model import (
    TorchProgressiveSequenceClassifier,
)

TRIAL_SET_SEQUENCE_PROTOCOL = "progressive_sequence_trial_set_regularization"


def _nonnegative_weight(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("trial_set_loss_weight must be a non-negative finite value.")
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "trial_set_loss_weight must be a non-negative finite value."
        ) from exc
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("trial_set_loss_weight must be a non-negative finite value.")
    return weight


def trial_set_mass_loss(logits, labels):
    """Return mean squared event-to-class mass error for labeled trials.

    ``logits`` must have shape ``(trials, events, classes)`` and ``labels`` must
    have shape ``(trials, events)``. The target mass for a class is its number of
    labeled occurrences in that trial. For the Katja variable-press trials this
    is one for four physical fingers and zero for the participant's fixed finger.
    """

    torch = _torch()
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels must have shapes (trials, events, classes) and "
            "(trials, events)."
        )
    if logits.shape[2] < 2:
        raise ValueError("trial-set loss requires at least two classes.")
    if labels.numel() and (
        bool(torch.any(labels < 0)) or bool(torch.any(labels >= logits.shape[2]))
    ):
        raise ValueError("labels contain an out-of-range class index.")
    probabilities = torch.softmax(logits, dim=2)
    predicted_mass = probabilities.sum(dim=1)
    target_mass = torch.nn.functional.one_hot(
        labels,
        num_classes=logits.shape[2],
    ).to(dtype=logits.dtype).sum(dim=1)
    return torch.mean((predicted_mass - target_mass) ** 2)


class TorchTrialSetSequenceClassifier(TorchProgressiveSequenceClassifier):
    """Progressive sequence classifier with a soft labeled trial-set loss."""

    def __init__(self, *args: Any, trial_set_loss_weight: float = 0.1, **kwargs: Any):
        self.trial_set_loss_weight = _nonnegative_weight(trial_set_loss_weight)
        super().__init__(*args, **kwargs)

    def _classification_loss(self, logits, labels, *, domains=None, include_vrex: bool):
        loss = super()._classification_loss(
            logits,
            labels,
            domains=domains,
            include_vrex=include_vrex,
        )
        if self.trial_set_loss_weight > 0.0:
            loss = loss + self.trial_set_loss_weight * trial_set_mass_loss(
                logits,
                labels,
            )
        return loss

    def metadata(self) -> dict[str, Any]:
        metadata = super().metadata()
        metadata.update(
            {
                "trial_set_sequence_protocol": TRIAL_SET_SEQUENCE_PROTOCOL,
                "trial_set_loss_weight": float(self.trial_set_loss_weight),
                "trial_set_uses_training_labels": True,
                "trial_set_uses_evaluation_labels": False,
                "trial_set_hard_constraint_at_inference": False,
            }
        )
        return metadata


__all__ = (
    "TRIAL_SET_SEQUENCE_PROTOCOL",
    "TorchTrialSetSequenceClassifier",
    "trial_set_mass_loss",
)

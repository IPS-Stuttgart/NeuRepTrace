from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.trial_set_sequence import (
    TorchTrialSetSequenceClassifier,
    trial_set_mass_loss,
)

torch = pytest.importorskip("torch")


def test_trial_set_mass_loss_prefers_unique_trial_classes() -> None:
    labels = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    unique_logits = torch.full((1, 4, 5), -4.0)
    duplicate_logits = torch.full((1, 4, 5), -4.0)
    for event, class_index in enumerate((0, 1, 2, 3)):
        unique_logits[0, event, class_index] = 4.0
    duplicate_logits[0, :, 0] = 4.0

    unique_loss = trial_set_mass_loss(unique_logits, labels)
    duplicate_loss = trial_set_mass_loss(duplicate_logits, labels)

    assert float(unique_loss) < float(duplicate_loss)
    assert float(unique_loss) < 1e-4


def test_trial_set_mass_loss_supports_rectangular_event_class_layout() -> None:
    labels = torch.tensor([[0, 1, 3, 4], [1, 2, 3, 4]], dtype=torch.long)
    logits = torch.randn(2, 4, 5, generator=torch.Generator().manual_seed(13))

    loss = trial_set_mass_loss(logits, labels)

    assert loss.ndim == 0
    assert bool(torch.isfinite(loss))


def test_trial_set_classifier_rejects_invalid_weight() -> None:
    for value in (-0.1, np.inf, True):
        with pytest.raises(ValueError, match="trial_set_loss_weight"):
            TorchTrialSetSequenceClassifier(trial_set_loss_weight=value)


def test_trial_set_metadata_declares_no_evaluation_labels() -> None:
    model = TorchTrialSetSequenceClassifier(trial_set_loss_weight=0.1)
    metadata = model.metadata()

    assert metadata["trial_set_loss_weight"] == 0.1
    assert metadata["trial_set_uses_training_labels"] is True
    assert metadata["trial_set_uses_evaluation_labels"] is False
    assert metadata["trial_set_hard_constraint_at_inference"] is False

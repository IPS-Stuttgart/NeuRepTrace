from __future__ import annotations

import numpy as np
import pytest

from neureptrace.decoding.exact_permutation import (
    exact_permutation_decode,
    torch_exact_permutation_nll,
)
from neureptrace.decoding.monotone_exact_sequence import (
    MONOTONE_EXACT_SEQUENCE_PROTOCOL,
    TorchMonotoneExactSequenceClassifier,
)


def test_exact_permutation_decode_returns_doubly_stochastic_marginals() -> None:
    probabilities = np.asarray(
        [
            [
                [0.80, 0.10, 0.05, 0.05],
                [0.10, 0.80, 0.05, 0.05],
                [0.05, 0.05, 0.80, 0.10],
                [0.05, 0.05, 0.10, 0.80],
            ]
        ]
    )

    result = exact_permutation_decode(probabilities)

    np.testing.assert_array_equal(result.map_assignments, [[0, 1, 2, 3]])
    np.testing.assert_allclose(result.marginals.sum(axis=1), np.ones((1, 4)))
    np.testing.assert_allclose(result.marginals.sum(axis=2), np.ones((1, 4)))
    np.testing.assert_allclose(
        result.permutation_probabilities.sum(axis=1),
        np.ones(1),
    )


torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def test_exact_permutation_nll_prefers_the_true_assignment() -> None:
    labels = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    good = torch.full((1, 4, 4), -2.0, dtype=torch.float32)
    bad = good.clone()
    good[0, torch.arange(4), labels[0]] = 3.0
    bad[0, torch.arange(4), torch.tensor([1, 0, 3, 2])] = 3.0
    good.requires_grad_()

    good_loss = torch_exact_permutation_nll(good, labels)
    bad_loss = torch_exact_permutation_nll(bad, labels)

    assert float(good_loss) < float(bad_loss)
    good_loss.backward()
    assert good.grad is not None


def _toy_sequence_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_classes = 4
    n_features = 8
    class_vectors = rng.normal(size=(n_classes, n_features))
    source_x = []
    source_y = []
    source_subjects = []
    for subject_index, shift in enumerate((0.0, 0.25, -0.2)):
        for _ in range(12):
            permutation = rng.permutation(n_classes)
            source_y.append(permutation)
            source_x.append(
                class_vectors[permutation]
                + shift
                + rng.normal(scale=0.25, size=(n_classes, n_features))
            )
            source_subjects.append(f"s{subject_index}")
    target_x = []
    target_y = []
    target_strata = []
    target_shift = rng.normal(scale=0.35, size=n_features)
    for stratum in range(4):
        for _ in range(4):
            permutation = rng.permutation(n_classes)
            target_y.append(permutation)
            target_x.append(
                class_vectors[permutation]
                + target_shift
                + rng.normal(scale=0.25, size=(n_classes, n_features))
            )
            target_strata.append(stratum)
    return (
        np.asarray(source_x, dtype=np.float32),
        np.asarray(source_y),
        np.asarray(source_subjects),
        np.asarray(target_x, dtype=np.float32),
        np.asarray(target_y),
        np.asarray(target_strata),
    )


def _small_model(**overrides):
    kwargs = {
        "hidden_units": 16,
        "num_layers": 1,
        "num_heads": 4,
        "adapter_rank": 2,
        "source_max_epochs": 1,
        "meta_epochs": 0,
        "adapter_steps": 1,
        "last_block_steps": 0,
        "full_finetune_steps": 0,
        "batch_size": 16,
        "patience": 1,
        "dropout": 0.0,
        "feature_noise_std": 0.0,
        "feature_dropout": 0.0,
        "validation_fraction": 0.25,
        "random_state": 17,
    }
    kwargs.update(overrides)
    return TorchMonotoneExactSequenceClassifier(**kwargs)


def test_worsening_target_stage_is_rejected_and_rolled_back(monkeypatch) -> None:
    source_x, source_y, subjects, target_x, target_y, strata = (
        _toy_sequence_problem(seed=2)
    )
    model = _small_model(refit_target_on_all=True)
    model.fit_source(source_x, source_y, source_subjects=subjects)
    source_state = {
        name: value.detach().cpu().clone()
        for name, value in model.source_state_.items()
    }
    losses = iter((1.0, 2.0))
    monkeypatch.setattr(
        model,
        "_validation_loss",
        lambda *_args, **_kwargs: next(losses),
    )

    model.adapt_target(target_x[:12], target_y[:12], target_strata=strata[:12])

    history = model.adaptation_stage_history_[0]
    assert history["stage_accepted"] is False
    assert history["best_step"] == 0
    assert history["refit_on_all_calibration"] is False
    for name, value in model.model_.state_dict().items():
        torch.testing.assert_close(value.detach().cpu(), source_state[name])


def test_selected_target_step_is_refit_on_all_calibration(monkeypatch) -> None:
    source_x, source_y, subjects, target_x, target_y, strata = (
        _toy_sequence_problem(seed=3)
    )
    model = _small_model(refit_target_on_all=True)
    model.fit_source(source_x, source_y, source_subjects=subjects)
    losses = iter((2.0, 1.0))
    monkeypatch.setattr(
        model,
        "_validation_loss",
        lambda *_args, **_kwargs: next(losses),
    )
    seen_sizes: list[int] = []
    original_update = model._adaptation_update

    def recording_update(*args, **kwargs):
        seen_sizes.append(int(np.asarray(kwargs["target_indices"]).size))
        return original_update(*args, **kwargs)

    monkeypatch.setattr(model, "_adaptation_update", recording_update)

    model.adapt_target(target_x[:12], target_y[:12], target_strata=strata[:12])

    history = model.adaptation_stage_history_[0]
    assert history["stage_accepted"] is True
    assert history["best_step"] == 1
    assert history["refit_on_all_calibration"] is True
    assert seen_sizes[0] < 12
    assert seen_sizes[-1] == 12
    metadata = model.metadata()
    assert (
        metadata["monotone_exact_sequence_protocol"]
        == MONOTONE_EXACT_SEQUENCE_PROTOCOL
    )
    assert metadata["uses_evaluation_labels"] is False

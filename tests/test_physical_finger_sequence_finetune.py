import numpy as np
import pytest

from neureptrace.decoding.physical_finger_sequence_finetune import (
    TorchPhysicalFingerSequenceClassifier,
)


def _synthetic_source(seed: int = 3):
    rng = np.random.default_rng(seed)
    physical_vectors = rng.normal(size=(5, 12)).astype(np.float32)
    rows = []
    labels = []
    subjects = []
    variable_sets = {
        "s1": np.array([0, 1, 2, 3]),
        "s2": np.array([0, 1, 2, 4]),
        "s3": np.array([0, 1, 3, 4]),
    }
    for subject, physical_codes in variable_sets.items():
        for _ in range(10):
            order = rng.permutation(physical_codes)
            features = physical_vectors[order] + 0.05 * rng.normal(size=(4, 12))
            rows.append(features.astype(np.float32))
            labels.append(order)
            subjects.append(subject)
    return np.stack(rows), np.stack(labels), np.asarray(subjects, dtype=object), physical_vectors


def _synthetic_target(physical_vectors, seed: int = 7):
    rng = np.random.default_rng(seed)
    codes = np.array([0, 1, 3, 4])
    local_by_physical = {physical: local for local, physical in enumerate(codes.tolist())}
    features = []
    local_labels = []
    physical_labels = []
    strata = []
    for trial in range(16):
        order = rng.permutation(codes)
        features.append(
            (physical_vectors[order] + 0.05 * rng.normal(size=(4, physical_vectors.shape[1]))).astype(np.float32)
        )
        physical_labels.append(order)
        local_labels.append(np.asarray([local_by_physical[int(value)] for value in order]))
        strata.append(trial % 4)
    return np.stack(features), np.stack(local_labels), np.stack(physical_labels), np.asarray(strata)


def _classifier():
    return TorchPhysicalFingerSequenceClassifier(
        hidden_units=16,
        num_layers=1,
        num_heads=2,
        adapter_rank=4,
        source_max_epochs=3,
        adapter_steps=2,
        last_block_steps=1,
        full_finetune_steps=1,
        meta_epochs=0,
        batch_size=16,
        validation_fraction=0.2,
        patience=2,
        min_trials_for_last_block=2,
        min_trials_for_full_finetune=2,
        feature_noise_std=0.0,
        feature_dropout=0.0,
        sinkhorn_loss_weight=0.1,
        assignment_loss_weight=0.05,
        source_replay_weight=0.1,
        random_state=13,
        device="cpu",
    )


def test_physical_head_selects_target_columns_and_keeps_source_replay_view():
    source_x, source_physical, source_subjects, physical_vectors = _synthetic_source()
    target_x, target_local, target_physical, strata = _synthetic_target(physical_vectors)

    model = _classifier().fit_source(
        source_x,
        source_physical,
        source_subjects=source_subjects,
    )
    assert model.n_physical_classes_ == 5

    model.adapt_target(
        target_x[:8],
        target_local[:8],
        target_calibration_physical_labels=target_physical[:8],
        target_strata=strata[:8],
    )

    probabilities = model.predict_proba(target_x[8:])
    assert probabilities.shape == (8, 4, 4)
    np.testing.assert_allclose(probabilities.sum(axis=2), 1.0, atol=1e-6)
    np.testing.assert_array_equal(model.target_physical_codes_, np.array([0, 1, 3, 4]))

    torch = pytest.importorskip("torch")
    tensor = torch.as_tensor(target_x[8:10], dtype=torch.float32)
    with torch.no_grad():
        target_logits = model.model_(tensor, use_adapter=True)
        physical_logits = model.model_(tensor, use_adapter=False)
    assert target_logits.shape == (2, 4, 4)
    assert physical_logits.shape == (2, 4, 5)
    assert model.metadata()["physical_finger_source_pretraining"] is True


def test_target_mapping_must_be_calibration_consistent():
    source_x, source_physical, source_subjects, physical_vectors = _synthetic_source()
    target_x, target_local, target_physical, strata = _synthetic_target(physical_vectors)
    target_physical = target_physical.copy()
    target_physical[1, target_local[1] == 0] = 2

    model = _classifier().fit_source(
        source_x,
        source_physical,
        source_subjects=source_subjects,
    )
    with pytest.raises(ValueError, match="exactly one physical finger"):
        model.adapt_target(
            target_x[:8],
            target_local[:8],
            target_calibration_physical_labels=target_physical[:8],
            target_strata=strata[:8],
        )

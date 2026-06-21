import numpy as np
import pytest


def _small_source_target_problem():
    rng = np.random.default_rng(7)
    source_features = []
    source_labels = []
    source_groups = []
    for group in range(3):
        shift = group * 0.05
        for label, center in [(0, -1.0), (1, 1.0)]:
            for _ in range(4):
                source_features.append([center + shift + rng.normal(0.0, 0.05), rng.normal(0.0, 0.05)])
                source_labels.append(label)
                source_groups.append(f"s{group}")
    target_features = np.array(
        [
            [-0.95, 0.40],
            [-1.05, 0.35],
            [-0.90, 0.45],
            [-1.10, 0.42],
            [0.95, 0.40],
            [1.05, 0.35],
            [0.90, 0.45],
            [1.10, 0.42],
        ],
        dtype=float,
    )
    target_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=object)
    return np.asarray(source_features, dtype=float), np.asarray(source_labels, dtype=object), np.asarray(source_groups, dtype=object), target_features, target_labels


def test_semi_supervised_lora_few_shot_does_not_use_evaluation_labels_with_fixed_split():
    pytest.importorskip("torch")
    from neureptrace.decoding.few_shot import FewShotTargetCalibrationSplit
    from neureptrace.decoding.semi_supervised_lora_few_shot import (
        SEMI_SUPERVISED_LORA_FEW_SHOT_CATEGORY,
        SEMI_SUPERVISED_LORA_FEW_SHOT_PROTOCOL,
        fit_semi_supervised_lora_few_shot_decoder,
    )

    source_features, source_labels, source_groups, target_features, target_labels = _small_source_target_problem()
    split = FewShotTargetCalibrationSplit(calibration_indices=np.array([0, 4]), evaluation_indices=np.array([1, 2, 3, 5, 6, 7]))
    poisoned_labels = target_labels.copy()
    poisoned_labels[split.evaluation_indices] = np.array(["x", "y", "z", "u", "v", "w"], dtype=object)

    common_kwargs = dict(
        source_features=source_features,
        source_labels=source_labels,
        source_groups=source_groups,
        target_features=target_features,
        split=split,
        per_class=1,
        seed=23,
        hidden_units=8,
        lora_rank=2,
        source_pretrain_epochs=4,
        meta_epochs=2,
        meta_inner_steps=1,
        target_adaptation_steps=3,
        batch_size=8,
        learning_rate=0.02,
        adapter_learning_rate=0.02,
        entropy_loss_weight=0.0,
        consistency_loss_weight=0.0,
        use_evaluation_features_unlabeled=False,
        device="cpu",
    )

    clean = fit_semi_supervised_lora_few_shot_decoder(target_labels=target_labels, **common_kwargs)
    poisoned = fit_semi_supervised_lora_few_shot_decoder(target_labels=poisoned_labels, **common_kwargs)

    np.testing.assert_allclose(clean.probabilities, poisoned.probabilities, atol=1e-6)
    np.testing.assert_array_equal(clean.calibration_indices, np.array([0, 4]))
    np.testing.assert_array_equal(clean.evaluation_indices, np.array([1, 2, 3, 5, 6, 7]))
    assert clean.metadata["few_shot_protocol"] == SEMI_SUPERVISED_LORA_FEW_SHOT_PROTOCOL
    assert clean.metadata["few_shot_protocol_category"] == SEMI_SUPERVISED_LORA_FEW_SHOT_CATEGORY
    assert clean.metadata["semi_supervised_lora_uses_target_evaluation_labels"] is False
    assert clean.metadata["semi_supervised_lora_valid_for_strict_source_only"] is False
    assert clean.metadata["semi_supervised_lora_meta_learning_enabled"] is True
    assert clean.metadata["semi_supervised_lora_meta_episodes"] > 0


def test_semi_supervised_lora_few_shot_marks_transductive_unlabeled_target_features():
    pytest.importorskip("torch")
    from neureptrace.decoding.semi_supervised_lora_few_shot import fit_semi_supervised_lora_few_shot_decoder

    source_features, source_labels, source_groups, target_features, target_labels = _small_source_target_problem()

    result = fit_semi_supervised_lora_few_shot_decoder(
        source_features=source_features,
        source_labels=source_labels,
        source_groups=source_groups,
        target_features=target_features,
        target_labels=target_labels,
        per_class=1,
        seed=31,
        hidden_units=8,
        lora_rank=2,
        source_pretrain_epochs=2,
        meta_epochs=1,
        meta_inner_steps=1,
        target_adaptation_steps=2,
        batch_size=8,
        learning_rate=0.02,
        adapter_learning_rate=0.02,
        entropy_loss_weight=0.01,
        use_evaluation_features_unlabeled=True,
        device="cpu",
    )

    assert result.probabilities.shape[0] == result.evaluation_indices.shape[0]
    assert result.probabilities.shape[1] == 2
    assert result.metadata["semi_supervised_lora_uses_unlabeled_target_features"] is True
    assert result.metadata["semi_supervised_lora_transductive_evaluation_features"] is True
    assert result.metadata["semi_supervised_lora_unlabeled_target_rows"] == result.evaluation_indices.shape[0]
    assert result.metadata["few_shot_n_target_calibration_rows"] == 2
    assert result.metadata["few_shot_n_target_evaluation_rows"] == 6

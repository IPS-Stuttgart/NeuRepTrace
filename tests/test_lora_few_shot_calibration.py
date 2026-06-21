import numpy as np
import pytest

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from neureptrace.decoding.lora_few_shot import (  # noqa: E402
    LORA_FEW_SHOT_CALIBRATION_CATEGORY,
    LORA_FEW_SHOT_CALIBRATION_PROTOCOL,
    LoRAFewShotTargetCalibrationSplit,
    fit_lora_few_shot_target_calibrated_decoder,
    select_lora_few_shot_target_calibration_split,
)


def _toy_problem(seed=0):
    rng = np.random.default_rng(seed)
    source_features = np.vstack(
        [
            rng.normal(-1.0, 0.15, size=(10, 3)),
            rng.normal(1.0, 0.15, size=(10, 3)),
            rng.normal(-0.8, 0.15, size=(10, 3)),
            rng.normal(0.8, 0.15, size=(10, 3)),
        ]
    )
    source_labels = np.array([0] * 10 + [1] * 10 + [0] * 10 + [1] * 10)
    source_subjects = np.array(["s1"] * 20 + ["s2"] * 20)
    target_features = np.vstack([rng.normal(-0.9, 0.15, size=(4, 3)), rng.normal(0.9, 0.15, size=(4, 3))])
    target_labels = np.array([0] * 4 + [1] * 4)
    split = LoRAFewShotTargetCalibrationSplit(calibration_indices=np.array([0, 4]), evaluation_indices=np.array([1, 2, 3, 5, 6, 7]))
    return source_features, source_labels, source_subjects, target_features, target_labels, split


def test_select_lora_few_shot_target_calibration_split_is_balanced_and_disjoint():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

    split_a = select_lora_few_shot_target_calibration_split(labels, per_class=1, seed=7, context=("fold", 1))
    split_b = select_lora_few_shot_target_calibration_split(labels, per_class=1, seed=7, context=("fold", 1))

    np.testing.assert_array_equal(split_a.calibration_indices, split_b.calibration_indices)
    np.testing.assert_array_equal(split_a.evaluation_indices, split_b.evaluation_indices)
    assert np.intersect1d(split_a.calibration_indices, split_a.evaluation_indices).size == 0
    assert split_a.calibration_indices.size == 3
    assert split_a.evaluation_indices.size == 6
    assert {int(label): int(np.count_nonzero(labels[split_a.calibration_indices] == label)) for label in np.unique(labels)} == {0: 1, 1: 1, 2: 1}


def test_lora_few_shot_uses_only_calibration_target_labels_not_evaluation_labels():
    source_features, source_labels, _subjects, target_features, target_labels, split = _toy_problem(seed=1)
    common_kwargs = dict(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        split=split,
        hidden_units=8,
        lora_rank=2,
        source_max_epochs=1,
        adaptation_steps=2,
        batch_size=8,
        dropout=0.0,
        seed=19,
    )

    result = fit_lora_few_shot_target_calibrated_decoder(target_labels=target_labels, **common_kwargs)
    perturbed_labels = target_labels.copy()
    perturbed_labels[split.evaluation_indices] = 1 - perturbed_labels[split.evaluation_indices]
    perturbed_result = fit_lora_few_shot_target_calibrated_decoder(target_labels=perturbed_labels, **common_kwargs)

    assert result.probabilities.shape == (6, 2)
    np.testing.assert_allclose(result.probabilities, perturbed_result.probabilities)
    assert result.metadata["lora_few_shot_protocol"] == LORA_FEW_SHOT_CALIBRATION_PROTOCOL
    assert result.metadata["lora_few_shot_protocol_category"] == LORA_FEW_SHOT_CALIBRATION_CATEGORY
    assert result.metadata["lora_few_shot_uses_target_labels"] is True
    assert result.metadata["lora_few_shot_valid_for_strict_source_only"] is False
    assert result.metadata["lora_few_shot_n_target_calibration_rows"] == 2
    assert result.metadata["lora_few_shot_n_target_evaluation_rows"] == 6


def test_lora_few_shot_meta_learning_and_unlabeled_target_features_are_reported():
    source_features, source_labels, source_subjects, target_features, target_labels, split = _toy_problem(seed=2)

    result = fit_lora_few_shot_target_calibrated_decoder(
        source_features=source_features,
        source_labels=source_labels,
        source_subjects=source_subjects,
        target_features=target_features,
        target_labels=target_labels,
        target_unlabeled_features=target_features[[2, 6]],
        split=split,
        hidden_units=8,
        lora_rank=2,
        source_max_epochs=1,
        adaptation_steps=2,
        meta_epochs=1,
        meta_inner_steps=1,
        batch_size=8,
        entropy_loss_weight=0.01,
        consistency_loss_weight=0.01,
        source_replay_weight=0.01,
        dropout=0.0,
        seed=23,
    )

    assert result.metadata["lora_few_shot_meta_learning"] is True
    assert result.metadata["lora_few_shot_meta_episodes_run"] > 0
    assert result.metadata["lora_few_shot_uses_unlabeled_target_features"] is True
    assert result.metadata["lora_few_shot_target_unlabeled_rows"] == 2
    assert result.metadata["lora_few_shot_uses_evaluation_features_as_unlabeled"] is False
    trainable_names = set(result.metadata["lora_few_shot_trainable_parameter_names"])
    assert {"lora_a.weight", "lora_b.weight"}.issubset(trainable_names)
    assert "base.weight" not in trainable_names


def test_lora_few_shot_flags_transductive_evaluation_features_when_requested():
    source_features, source_labels, _subjects, target_features, target_labels, split = _toy_problem(seed=3)

    result = fit_lora_few_shot_target_calibrated_decoder(
        source_features=source_features,
        source_labels=source_labels,
        target_features=target_features,
        target_labels=target_labels,
        split=split,
        hidden_units=8,
        lora_rank=2,
        source_max_epochs=1,
        adaptation_steps=2,
        batch_size=8,
        use_evaluation_features_as_unlabeled=True,
        entropy_loss_weight=0.01,
        dropout=0.0,
        seed=29,
    )

    assert result.metadata["lora_few_shot_uses_unlabeled_target_features"] is True
    assert result.metadata["lora_few_shot_uses_evaluation_features_as_unlabeled"] is True
    assert result.metadata["lora_few_shot_transductive_evaluation_features"] is True
    assert result.metadata["lora_few_shot_target_unlabeled_rows"] == result.evaluation_indices.size

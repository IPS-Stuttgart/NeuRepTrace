import numpy as np
import pytest

from neureptrace.decoding.progressive_sequence_finetune import (
    PROGRESSIVE_SEQUENCE_CATEGORY,
    PROGRESSIVE_SEQUENCE_PROTOCOL,
    NestedTrialCalibrationSplit,
    TorchProgressiveSequenceClassifier,
    fit_progressive_sequence_target_calibrated_decoder,
    pack_complete_trial_events,
    permutation_constrained_decode,
    select_nested_trial_calibration_splits,
)


def test_pack_complete_trial_events_sorts_rows_and_preserves_back_mapping():
    features = np.arange(24, dtype=float).reshape(8, 3)
    trial_ids = np.array(["b", "a", "b", "a", "b", "a", "b", "a"], dtype=object)
    positions = np.array([4, 4, 2, 2, 1, 1, 3, 3])
    labels = np.array([3, 3, 1, 1, 0, 0, 2, 2])

    packed = pack_complete_trial_events(
        features,
        trial_ids,
        positions,
        labels=labels,
        expected_events=4,
        require_permutation_labels=True,
    )

    assert packed.features.shape == (2, 4, 3)
    assert packed.trial_ids.tolist() == ["b", "a"]
    np.testing.assert_array_equal(packed.press_positions, np.tile(np.arange(1, 5), (2, 1)))
    np.testing.assert_array_equal(packed.labels, np.tile(np.arange(4), (2, 1)))
    np.testing.assert_array_equal(packed.features, features[packed.row_indices])


def test_nested_trial_calibration_reserves_max_pool_first_and_keeps_evaluation_fixed():
    strata = np.repeat(np.array([0, 1, 2, 3]), 8)
    splits_a = select_nested_trial_calibration_splits(
        strata,
        calibration_counts=(1, 3, 5),
        max_per_stratum=5,
        seed=29,
        context=("target", "s05"),
    )
    splits_b = select_nested_trial_calibration_splits(
        strata,
        calibration_counts=(1, 3, 5),
        max_per_stratum=5,
        seed=29,
        context=("target", "s05"),
    )

    np.testing.assert_array_equal(splits_a[1].evaluation_indices, splits_a[5].evaluation_indices)
    np.testing.assert_array_equal(splits_a[1].calibration_pool_indices, splits_a[5].calibration_pool_indices)
    np.testing.assert_array_equal(splits_a[3].calibration_indices, splits_b[3].calibration_indices)
    assert set(splits_a[1].calibration_indices).issubset(set(splits_a[3].calibration_indices))
    assert set(splits_a[3].calibration_indices).issubset(set(splits_a[5].calibration_indices))
    assert splits_a[5].calibration_indices.size == 20
    assert splits_a[5].evaluation_indices.size == 12


def test_permutation_constrained_decode_assigns_each_class_once_and_improves_joint_map():
    probabilities = np.array(
        [
            [
                [0.45, 0.44, 0.06, 0.05],
                [0.43, 0.42, 0.10, 0.05],
                [0.05, 0.05, 0.80, 0.10],
                [0.05, 0.05, 0.10, 0.80],
            ]
        ]
    )

    result = permutation_constrained_decode(probabilities)

    np.testing.assert_array_equal(np.sort(result.assignments[0]), np.arange(4))
    np.testing.assert_allclose(result.probabilities.sum(axis=1), np.ones((1, 4)), atol=1e-6)
    np.testing.assert_allclose(result.probabilities.sum(axis=2), np.ones((1, 4)), atol=1e-6)
    assert result.assignments[0, 0] != result.assignments[0, 1]


torch = pytest.importorskip("torch")
torch.set_num_threads(1)


def _toy_sequence_problem(seed=0):
    rng = np.random.default_rng(seed)
    classes = 4
    features = 8
    class_vectors = rng.normal(size=(classes, features))
    source_x = []
    source_y = []
    source_subjects = []
    for subject_index, shift in enumerate((0.0, 0.25, -0.2)):
        for _ in range(20):
            permutation = rng.permutation(classes)
            source_y.append(permutation)
            source_x.append(class_vectors[permutation] + shift + rng.normal(scale=0.25, size=(classes, features)))
            source_subjects.append(f"s{subject_index}")
    target_x = []
    target_y = []
    target_strata = []
    target_shift = rng.normal(scale=0.35, size=features)
    for stratum in range(4):
        for _ in range(5):
            permutation = rng.permutation(classes)
            target_y.append(permutation)
            target_x.append(class_vectors[permutation] + target_shift + rng.normal(scale=0.25, size=(classes, features)))
            target_strata.append(stratum)
    return (
        np.asarray(source_x, dtype=np.float32),
        np.asarray(source_y),
        np.asarray(source_subjects),
        np.asarray(target_x, dtype=np.float32),
        np.asarray(target_y),
        np.asarray(target_strata),
    )


def test_progressive_sequence_classifier_runs_all_stages_and_returns_permutations():
    source_x, source_y, subjects, target_x, target_y, strata = _toy_sequence_problem(seed=2)
    model = TorchProgressiveSequenceClassifier(
        hidden_units=16,
        num_layers=1,
        num_heads=4,
        adapter_rank=2,
        source_max_epochs=2,
        meta_epochs=0,
        adapter_steps=2,
        last_block_steps=2,
        full_finetune_steps=2,
        batch_size=16,
        patience=2,
        dropout=0.0,
        feature_noise_std=0.0,
        feature_dropout=0.0,
        min_trials_for_last_block=4,
        min_trials_for_full_finetune=8,
        random_state=17,
    )

    model.fit_source(source_x, source_y, source_subjects=subjects)
    model.adapt_target(target_x[:12], target_y[:12], target_strata=strata[:12])
    probabilities = model.predict_proba(target_x[12:])
    predictions = model.predict(target_x[12:], constrained=True)

    assert probabilities.shape == (8, 4, 4)
    np.testing.assert_allclose(probabilities.sum(axis=2), np.ones((8, 4)), atol=1e-6)
    for row in predictions:
        assert sorted(row.tolist()) == [0, 1, 2, 3]
    metadata = model.metadata()
    assert metadata["progressive_sequence_protocol"] == PROGRESSIVE_SEQUENCE_PROTOCOL
    assert metadata["progressive_sequence_protocol_category"] == PROGRESSIVE_SEQUENCE_CATEGORY
    assert metadata["progressive_sequence_adaptation_stages"] == ("adapter", "last_block", "full")


def test_progressive_helper_does_not_use_evaluation_labels_for_fitting():
    source_x, source_y, subjects, target_x, target_y, strata = _toy_sequence_problem(seed=4)
    split = NestedTrialCalibrationSplit(
        calibration_indices=np.arange(12),
        evaluation_indices=np.arange(12, 20),
        calibration_pool_indices=np.arange(12),
        per_stratum=3,
        max_per_stratum=3,
        seed=13,
    )
    kwargs = dict(
        source_features=source_x,
        source_labels=source_y,
        source_subjects=subjects,
        target_features=target_x,
        split=split,
        target_strata=strata,
        hidden_units=16,
        num_layers=1,
        num_heads=4,
        adapter_rank=2,
        source_max_epochs=1,
        meta_epochs=0,
        adapter_steps=1,
        last_block_steps=1,
        full_finetune_steps=1,
        batch_size=16,
        patience=1,
        dropout=0.0,
        feature_noise_std=0.0,
        feature_dropout=0.0,
        min_trials_for_last_block=4,
        min_trials_for_full_finetune=8,
        random_state=31,
    )

    result = fit_progressive_sequence_target_calibrated_decoder(target_labels=target_y, **kwargs)
    perturbed = target_y.copy()
    perturbed[split.evaluation_indices] = np.roll(perturbed[split.evaluation_indices], shift=1, axis=1)
    perturbed_result = fit_progressive_sequence_target_calibrated_decoder(target_labels=perturbed, **kwargs)

    np.testing.assert_allclose(result.probabilities, perturbed_result.probabilities, atol=1e-7)
    np.testing.assert_array_equal(result.predictions, perturbed_result.predictions)
    assert result.metadata["progressive_sequence_n_target_calibration_trials"] == 12
    assert result.metadata["progressive_sequence_n_target_evaluation_trials"] == 8


def test_source_subject_meta_episodes_initialize_the_target_adapter():
    source_x, source_y, subjects, _target_x, _target_y, _strata = _toy_sequence_problem(seed=7)
    model = TorchProgressiveSequenceClassifier(
        hidden_units=16,
        num_layers=1,
        num_heads=4,
        adapter_rank=2,
        source_max_epochs=1,
        meta_epochs=1,
        meta_support_trials=2,
        meta_query_trials=2,
        meta_inner_steps=1,
        adapter_steps=1,
        last_block_steps=0,
        full_finetune_steps=0,
        batch_size=16,
        patience=1,
        dropout=0.0,
        feature_noise_std=0.0,
        feature_dropout=0.0,
        random_state=41,
    )

    model.fit_source(source_x, source_y, source_subjects=subjects)

    assert model.meta_episodes_run_ == 3
    assert model.meta_episodes_accepted_ >= 1
    assert float(torch.linalg.vector_norm(model.model_.adapter_up.weight).detach().cpu()) > 0.0

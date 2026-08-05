import numpy as np

from neureptrace.katja_physical_finger_sequence_benchmark import (
    _participant_variable_codes,
    run_katja_physical_finger_sequence_benchmark,
)


def _cache(seed: int = 12):
    rng = np.random.default_rng(seed)
    physical_vectors = rng.normal(size=(5, 10)).astype(np.float32)
    variable_sets = {
        "s1": np.array([0, 1, 2, 3]),
        "s2": np.array([0, 1, 2, 4]),
        "s3": np.array([0, 2, 3, 4]),
        "s4": np.array([0, 1, 3, 4]),
    }
    features = []
    subjects = []
    trials = []
    positions = []
    sequence_ids = []
    physical_codes = []
    for subject, codes in variable_sets.items():
        for trial in range(16):
            order = rng.permutation(codes)
            for event, physical in enumerate(order, start=2):
                features.append(
                    physical_vectors[physical]
                    + 0.03 * rng.normal(size=physical_vectors.shape[1])
                )
                subjects.append(subject)
                trials.append(trial)
                positions.append(event)
                sequence_ids.append(trial % 4)
                physical_codes.append(physical)
    return {
        "features": np.asarray(features, dtype=np.float32),
        "subjects": np.asarray(subjects),
        "trial_ids": np.asarray(trials),
        "press_positions": np.asarray(positions),
        "sequence_ids": np.asarray(sequence_ids),
        "finger_codes": np.asarray(physical_codes),
        "correct_order": np.ones(len(features), dtype=bool),
    }


def test_participant_variable_code_audit_is_stable():
    cache = _cache()
    codes = _participant_variable_codes(
        cache["subjects"],
        cache["finger_codes"],
    )
    assert codes["s1"] == (0, 1, 2, 3)
    assert codes["s4"] == (0, 1, 3, 4)


def test_physical_benchmark_reports_all_three_endpoints():
    cache = _cache()
    per_seed, per_target, summary, metadata = run_katja_physical_finger_sequence_benchmark(
        cache,
        participants=("s1", "s2", "s3", "s4"),
        target_participants=("s4",),
        calibration_counts=(1,),
        calibration_seeds=(13,),
        source_selection_mode="all",
        pca_components=8,
        model_repeats=1,
        model_kwargs={
            "hidden_units": 16,
            "num_layers": 1,
            "num_heads": 2,
            "adapter_rank": 4,
            "source_max_epochs": 2,
            "adapter_steps": 1,
            "last_block_steps": 1,
            "full_finetune_steps": 1,
            "meta_epochs": 0,
            "batch_size": 16,
            "validation_fraction": 0.2,
            "patience": 1,
            "min_trials_for_last_block": 2,
            "min_trials_for_full_finetune": 2,
            "feature_noise_std": 0.0,
            "feature_dropout": 0.0,
            "source_replay_weight": 0.1,
            "random_state": 13,
            "device": "cpu",
        },
    )
    assert per_seed.shape[0] == 1
    assert per_target.shape[0] == 1
    assert summary.shape[0] == 1
    assert {
        "independent_accuracy",
        "soft_assignment_accuracy",
        "permutation_accuracy",
    }.issubset(per_seed.columns)
    assert metadata["source_objective"] == "global_physical_finger_identity"
    assert metadata["target_physical_mapping_source"] == "target_calibration_rows_only"

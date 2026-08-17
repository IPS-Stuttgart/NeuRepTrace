from __future__ import annotations

import argparse
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from neureptrace.katja_julia_window_benchmark import (
    JULIA_SUBJECTS,
    _append_row,
    aggregate_benchmark_shards,
    paired_common_cohort_statistics,
    prepare_fold_features,
    relabel_minimum_overlap,
    run_benchmark,
    select_nested_trial_splits,
    summarize_results,
)
from neureptrace.decoding.progressive_temporal_window_finetune import (
    TorchProgressiveTemporalWindowClassifier,
)


def test_minimum_overlap_relabel_only_demotes_labels() -> None:
    raw = np.asarray([0, 1, 2, 3, 0])
    overlap = np.asarray([0.9, 0.1, 0.2, 0.8, 0.0])

    result = relabel_minimum_overlap(raw, overlap, 0.2)

    assert result.tolist() == [0, 0, 2, 3, 0]
    assert raw.tolist() == [0, 1, 2, 3, 0]


def test_incremental_csv_rejects_schema_drift(tmp_path) -> None:
    path = tmp_path / "partial.csv"
    _append_row(path, {"target": "s05", "accuracy": 0.5})
    with pytest.raises(ValueError, match="different CSV schema"):
        _append_row(path, {"target": "s05", "accuracy": 0.6, "extra": 1})
    frame = pd.read_csv(path)
    assert frame.to_dict("records") == [{"target": "s05", "accuracy": 0.5}]


def test_nested_trial_split_uses_k_complete_trials_per_sequence() -> None:
    sequence = np.repeat([0, 1], 12)
    trials = np.repeat(np.arange(12), 2)

    splits = select_nested_trial_splits(
        sequence,
        trials,
        k_values=(1, 2, 4),
        seed=3,
        context="s05",
    )

    previous: set[int] = set()
    for k, split in splits.items():
        calibration_trials = set(int(value) for value in split.calibration_trials)
        assert len(calibration_trials) == 2 * k
        assert previous.issubset(calibration_trials)
        assert not np.intersect1d(split.calibration_rows, split.evaluation_rows).size
        assert set(trials[split.calibration_rows]) == calibration_trials
        assert set(trials[split.evaluation_rows]).isdisjoint(calibration_trials)
        previous = calibration_trials


def test_fold_preprocessing_does_not_fit_on_target_rows() -> None:
    rng = np.random.default_rng(5)
    features = rng.normal(size=(36, 8)).astype(np.float32)
    subjects = np.repeat([0, 1, 2], 12)
    perturbed = features.copy()
    perturbed[subjects == 2] += 10000.0

    first = prepare_fold_features(
        features,
        subjects,
        source_subjects=np.asarray([0, 1]),
        target_subject=2,
        pca_components=4,
        pca_fit_max_windows=20,
        seed=7,
    )
    second = prepare_fold_features(
        perturbed,
        subjects,
        source_subjects=np.asarray([0, 1]),
        target_subject=2,
        pca_components=4,
        pca_fit_max_windows=20,
        seed=7,
    )

    np.testing.assert_allclose(first[subjects != 2], second[subjects != 2], atol=1e-6)
    assert not np.allclose(first[subjects == 2], second[subjects == 2])


def test_summary_averages_seeds_before_subject_sem() -> None:
    rows = pd.DataFrame(
        {
            "method": ["m"] * 4,
            "k_trials_per_sequence": [1] * 4,
            "target": ["s05", "s05", "s06", "s06"],
            "seed": [0, 1, 0, 1],
            "accuracy_raw_labels": [0.2, 0.4, 0.6, 0.8],
            "balanced_accuracy_raw_labels": [0.2, 0.4, 0.6, 0.8],
            "accuracy_tau_labels": [0.2, 0.4, 0.6, 0.8],
            "press_only_finger_accuracy": [0.2, 0.4, 0.6, 0.8],
            "rest_recall": [0.2, 0.4, 0.6, 0.8],
            "press_detection_accuracy": [0.2, 0.4, 0.6, 0.8],
            "trial_macro_accuracy_raw_labels": [0.2, 0.4, 0.6, 0.8],
            "log_loss_raw_labels": [2.0, 1.8, 1.2, 1.0],
        }
    )

    subject, subject_summary, julia_style = summarize_results(rows)

    assert subject.loc[subject["target"] == "s05", "accuracy_raw_labels"].item() == pytest.approx(0.3)
    assert subject_summary["mean_accuracy_raw_labels"].item() == pytest.approx(0.5)
    assert subject_summary["sem_accuracy_raw_labels"].item() == pytest.approx(0.2)
    assert julia_style["mean_accuracy_raw_labels"].item() == pytest.approx(0.5)
    assert julia_style["sd_accuracy_raw_labels"].item() == pytest.approx(np.std([0.2, 0.4, 0.6, 0.8], ddof=1))


def test_paired_common_cohort_statistics_use_subject_as_unit() -> None:
    values = {
        ("s05", "source_only", 1): 0.30,
        ("s06", "source_only", 1): 0.40,
        ("s05", "source_only", 20): 0.30,
        ("s06", "source_only", 20): 0.40,
        ("s05", "adapter_only", 1): 0.35,
        ("s06", "adapter_only", 1): 0.45,
        ("s05", "adapter_only", 20): 0.50,
        ("s06", "adapter_only", 20): 0.68,
        ("s05", "progressive_full", 1): 0.40,
        ("s06", "progressive_full", 1): 0.50,
        ("s05", "progressive_full", 20): 0.60,
        ("s06", "progressive_full", 20): 0.80,
    }
    subject_rows = pd.DataFrame(
        [
            {
                "target": target,
                "method": method,
                "k_trials_per_sequence": k,
                "accuracy_raw_labels": accuracy,
            }
            for (target, method, k), accuracy in values.items()
        ]
    )

    result = paired_common_cohort_statistics(subject_rows).set_index("contrast")

    assert result.loc["progressive dose response", "n_subjects"] == 2
    assert result.loc["progressive dose response", "mean_delta_accuracy"] == pytest.approx(0.25)
    assert result.loc["progressive versus source at maximum k", "mean_delta_accuracy"] == pytest.approx(0.35)
    assert result.loc["progressive versus adapter at maximum k", "mean_delta_accuracy"] == pytest.approx(0.11)


def test_shard_aggregation_rebuilds_top_level_outputs(tmp_path) -> None:
    base_rows = {
        "seed": [0],
        "k_trials_per_sequence": [1],
        "method": ["source_only"],
        "accuracy_raw_labels": [0.3],
        "balanced_accuracy_raw_labels": [0.3],
        "accuracy_tau_labels": [0.3],
        "press_only_finger_accuracy": [0.3],
        "rest_recall": [0.3],
        "press_detection_accuracy": [0.3],
        "trial_macro_accuracy_raw_labels": [0.3],
        "log_loss_raw_labels": [1.8],
        "majority_class_accuracy": [0.28],
    }
    shards = []
    for target in ("s05", "s06"):
        shard = tmp_path / target
        shard.mkdir()
        pd.DataFrame({**base_rows, "target": [target]}).to_csv(shard / "fold_results.csv", index=False)
        shards.append(shard)

    output = aggregate_benchmark_shards(tuple(shards), tmp_path / "combined")

    assert pd.read_csv(output / "fold_results.csv")["target"].tolist() == ["s05", "s06"]
    assert (output / "summary_subject_sem.csv").exists()
    assert (output / "summary_common_subject_sem.csv").exists()
    assert (output / "paired_common_cohort_statistics.csv").exists()
    assert (output / "comparison_scope.json").exists()
    assert (output / "summary_julia_fold_sd.csv").exists()
    assert (output / "summary_julia_50fold_sd.csv").exists()
    assert (output / "katja_julia_window_comparison.pdf").exists()
    assert "task- and data-matched" in (output / "comparison_to_julia.md").read_text()
    scope = json.loads((output / "comparison_scope.json").read_text())
    assert scope["claim_level"] == "task_data_and_split_convention_matched_not_model_identical"
    validation = json.loads((output / "validation.json").read_text())
    assert validation["all_required_checks_pass"] is True


def _write_synthetic_window_cache(path) -> None:
    rng = np.random.default_rng(13)
    windows = []
    labels = []
    overlaps = []
    subjects = []
    sequences = []
    trials = []
    for subject in range(len(JULIA_SUBJECTS)):
        trial_number = 0
        for sequence in range(2):
            for _repeat in range(3):
                for window in range(6):
                    label = window
                    sample = rng.normal(scale=0.2, size=(4, 3)).astype(np.float32)
                    sample[:, label % 3] += 1.0
                    windows.append(sample)
                    labels.append(label)
                    overlaps.append(0.8 if label else 0.0)
                    subjects.append(subject)
                    sequences.append(sequence)
                    trials.append(trial_number)
                trial_number += 1
    np.savez(
        path,
        meg_windows=np.asarray(windows, dtype=np.float32),
        finger_ids=np.asarray(labels, dtype=np.int64),
        press_overlap_fraction=np.asarray(overlaps, dtype=np.float32),
        subject_indices=np.asarray(subjects, dtype=np.int64),
        sequence_id=np.asarray(sequences, dtype=np.int64),
        trial_id=np.asarray(trials, dtype=np.int64),
    )


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_temporal_window_model_runs_progressive_stages() -> None:
    rng = np.random.default_rng(4)
    windows = rng.normal(size=(72, 6, 4)).astype(np.float32)
    finger = np.tile(np.arange(6), 12)
    sequence = np.tile(np.arange(4), 18)
    order = finger.copy()
    overlap = (finger > 0).astype(np.float32) * 0.6
    source = np.arange(60)
    source_domains = np.repeat(np.arange(3), 24)
    sensor_mean = windows[source].mean(axis=(0, 1))
    sensor_std = windows[source].std(axis=(0, 1))
    model = TorchProgressiveTemporalWindowClassifier(
        hidden_units=8,
        num_blocks=1,
        adapter_rank=2,
        source_epochs=1,
        adapter_steps=1,
        last_block_steps=1,
        full_finetune_steps=1,
        batch_size=16,
        random_state=3,
        device="cpu",
    ).fit_source(
        windows,
        source_indices=source,
        source_domains=source_domains,
        finger_labels=finger,
        sequence_labels=sequence,
        order_labels=order,
        overlap_targets=overlap,
        sensor_mean=sensor_mean,
        sensor_std=sensor_std,
    )

    adapted = model.clone_source(random_state=5).adapt_target_indices(
        np.arange(60, 66),
        n_calibration_trials=12,
        mode="progressive_full",
    )
    probabilities = adapted.predict_proba_indices(np.arange(66, 72))

    assert [row["stage"] for row in adapted.adaptation_history_] == [
        "adapter",
        "last_block",
        "full",
    ]
    assert probabilities.shape == (6, 6)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
    assert model.source_validation_mode_ == "heldout_source_subject"
    assert model.source_validation_domain_ in {0, 1, 2}
    assert model.best_source_epoch_ == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch is optional")
def test_tiny_end_to_end_run_writes_matched_artifacts(tmp_path) -> None:
    cache = tmp_path / "cache.npz"
    output = tmp_path / "result"
    _write_synthetic_window_cache(cache)
    args = argparse.Namespace(
        cache=str(cache),
        out_dir=str(output),
        feature_mode="dct",
        feature_cache=None,
        raw_window_cache=None,
        subjects=",".join(JULIA_SUBJECTS),
        targets="s05",
        k_values="1",
        seeds="0",
        methods="source_only",
        minimum_overlap=0.2,
        temporal_coefficients=2,
        feature_batch_size=64,
        pca_components=4,
        pca_fit_max_windows=100,
        preprocessing_seed=13,
        hidden_units=8,
        num_layers=1,
        num_blocks=1,
        adapter_rank=2,
        source_epochs=1,
        adapter_steps=1,
        last_block_steps=1,
        full_finetune_steps=1,
        batch_size=32,
        sequence_loss_weight=0.15,
        order_loss_weight=0.30,
        overlap_loss_weight=0.30,
        source_model_seed=13,
        source_model_per_seed=False,
        device="cpu",
        max_folds=1,
        resume=False,
    )

    run_benchmark(args)

    rows = pd.read_csv(output / "fold_results.csv")
    assert rows[["target", "seed", "k_trials_per_sequence", "method"]].to_dict("records") == [{"target": "s05", "seed": 0, "k_trials_per_sequence": 1, "method": "source_only"}]
    assert rows["calibration_evaluation_disjoint"].all()
    assert (output / "subject_seed_averages.csv").exists()
    assert (output / "summary_julia_50fold_sd.csv").exists()
    assert (output / "katja_julia_window_comparison.png").exists()
    provenance = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["aggregation_collaborator"] == "mean and SD over subject-by-seed folds"

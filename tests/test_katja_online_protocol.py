from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.katja_online_protocol import (
    aggregate_fold_scores,
    build_trial_split_manifest,
    score_window_predictions,
    validate_fold_registry,
)


def _trial_table():
    rows = []
    for subject in ("s1", "s2"):
        for sequence in range(4):
            for trial in range(6):
                rows.append(
                    {
                        "subject": subject,
                        "trial_id": f"{sequence}-{trial}",
                        "sequence_id": sequence,
                    }
                )
    return pd.DataFrame(rows)


def test_nested_rest_selects_k_per_sequence_and_changes_evaluation_size():
    manifest, metadata = build_trial_split_manifest(
        _trial_table(), calibration_counts=(1, 2), seeds=(0,), mode="nested_rest"
    )
    assert metadata["exact_julia_split_status"].startswith("pending")
    for subject in ("s1", "s2"):
        one = manifest[(manifest.subject == subject) & (manifest.k == 1)]
        two = manifest[(manifest.subject == subject) & (manifest.k == 2)]
        assert (
            one[one.split_role == "calibration"].groupby("sequence_id").size()
            == 1
        ).all()
        assert (
            two[two.split_role == "calibration"].groupby("sequence_id").size()
            == 2
        ).all()
        assert set(one.loc[one.split_role == "calibration", "trial_id"]).issubset(
            set(two.loc[two.split_role == "calibration", "trial_id"])
        )
        assert (one.split_role == "evaluation").sum() > (
            two.split_role == "evaluation"
        ).sum()


def test_fixed_max_complement_keeps_evaluation_trials_constant():
    manifest, _ = build_trial_split_manifest(
        _trial_table(),
        calibration_counts=(1, 2),
        seeds=(0,),
        mode="fixed_max_complement",
    )
    one = manifest[(manifest.subject == "s1") & (manifest.k == 1)]
    two = manifest[(manifest.subject == "s1") & (manifest.k == 2)]
    assert set(one.loc[one.split_role == "evaluation", "trial_id"]) == set(
        two.loc[two.split_role == "evaluation", "trial_id"]
    )
    assert (one.split_role == "reserved_unused").sum() == 4


def test_reporting_produces_fold_sd_and_subject_sem():
    rows = []
    for subject, base in [("s1", 0), ("s2", 1)]:
        for seed in (0, 1):
            for index, true in enumerate(["null", "a", "b", "null"]):
                predicted = true
                if subject == "s2" and seed == 1 and index == 2:
                    predicted = "a"
                rows.append(
                    {
                        "subject": subject,
                        "seed": seed,
                        "k": 1,
                        "trial_id": base * 10 + index // 2,
                        "true_label": true,
                        "predicted_label": predicted,
                    }
                )
    folds = score_window_predictions(pd.DataFrame(rows))
    assert folds.shape[0] == 4
    registry = validate_fold_registry(
        folds,
        expected_subjects=("s1", "s2"),
        expected_seeds=(0, 1),
        expected_k=(1,),
    )
    assert registry["n_folds"] == 4
    julia, subjects, population = aggregate_fold_scores(folds)
    overall = julia[julia.metric == "overall_accuracy"].iloc[0]
    assert np.isclose(overall["mean"], 15 / 16)
    assert overall["n_folds"] == 4
    assert subjects.shape[0] == 2
    assert not population[population.metric == "overall_accuracy"].empty


def test_registry_rejects_missing_fold():
    folds = pd.DataFrame(
        {"subject": ["s1"], "seed": [0], "k": [1], "overall_accuracy": [1.0]}
    )
    with pytest.raises(ValueError, match="Incomplete"):
        validate_fold_registry(
            folds,
            expected_subjects=("s1",),
            expected_seeds=(0, 1),
            expected_k=(1,),
        )

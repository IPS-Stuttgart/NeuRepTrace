import numpy as np
import pytest

from neureptrace.decoding.alignment_benchmark import (
    AlignmentBenchmarkConfig,
    AlignmentSubjectData,
    evaluate_alignment_loso,
    rank_metrics,
    target_calibration_mask,
)


def _synthetic_subjects(seed=0, n_subjects=4, n_classes=3, repetitions=5):
    rng = np.random.default_rng(seed)
    labels = np.tile(np.arange(n_classes), repetitions)
    prototypes = np.eye(n_classes)
    subjects = []
    for subject_id in range(n_subjects):
        mixing = rng.normal(size=(n_classes, 7))
        features = prototypes[labels] @ mixing + 0.02 * rng.normal(size=(labels.size, 7))
        subjects.append(
            AlignmentSubjectData(
                subject_id=subject_id,
                features=features,
                labels=labels,
            )
        )
    return subjects


def test_hyperalignment_loso_with_heldout_target_calibration_returns_fold_artifacts():
    subjects = _synthetic_subjects()
    config = AlignmentBenchmarkConfig(
        method="hyperalignment",
        n_components=2,
        hyperalignment_iterations=2,
        target_calibration_mode="heldout_trials",
        target_calibration_trials_per_class=1,
        classifier="correlation-prototype",
        components_pca=float("inf"),
        signflip_permutations=0,
    )

    result = evaluate_alignment_loso(subjects, config=config)

    assert len(result["outer"]) == 4
    assert len(result["predictions"]) == 4 * 3 * 4
    assert len(result["group_summary"]) == 1
    assert {row["target_transform"] for row in result["outer"]} == {"target_calibrated"}
    assert all(row["n_target_calibration_trials"] == 3 for row in result["outer"])
    assert all(row["alignment_actual_components"] == 2 for row in result["outer"])
    assert result["folds"][0].alignment_model.n_components == 2


def test_mcca_loso_with_independent_alignment_data_scores_all_decode_rows():
    rng = np.random.default_rng(1)
    subjects = []
    for base in _synthetic_subjects(seed=1):
        subjects.append(
            AlignmentSubjectData(
                subject_id=base.subject_id,
                features=base.features,
                labels=base.labels,
                alignment_features=base.features + 0.01 * rng.normal(size=base.features.shape),
                alignment_labels=base.labels,
            )
        )
    config = AlignmentBenchmarkConfig(
        method="mcca",
        n_components=2,
        target_calibration_mode="alignment_data",
        classifier="correlation-prototype",
        components_pca=float("inf"),
        signflip_permutations=0,
    )

    result = evaluate_alignment_loso(subjects, config=config)

    assert len(result["outer"]) == 4
    assert len(result["predictions"]) == 4 * 3 * 5
    assert {row["target_transform"] for row in result["outer"]} == {"alignment_data_calibrated"}
    assert all(row["n_target_calibration_trials"] == 15 for row in result["outer"])
    assert all(row["alignment_method"] == "mcca" for row in result["outer"])


def test_positive_target_calibration_trials_imply_heldout_calibration_mode():
    config = AlignmentBenchmarkConfig(
        method="mcca",
        n_components=2,
        target_calibration_trials_per_class=1,
        classifier="correlation-prototype",
        components_pca=float("inf"),
        signflip_permutations=0,
    )

    result = evaluate_alignment_loso(_synthetic_subjects(), config=config, outer_subjects=[0])

    assert result["outer"][0]["target_calibration_mode"] == "heldout_trials"
    assert result["outer"][0]["target_transform"] == "target_calibrated"


def test_target_calibration_mask_requires_one_scored_row_per_class():
    with pytest.raises(ValueError, match="plus at least one scored trial"):
        target_calibration_mask(np.array([1, 1, 2, 2]), trials_per_class=2)


def test_rank_metrics_reports_topk_and_per_row_ranks():
    summary, rows = rank_metrics(
        np.array([1, 2]),
        np.array([[0.1, 0.9, 0.0], [0.3, 0.2, 0.1]]),
        np.array([0, 1, 2]),
    )

    assert summary["top2_accuracy"] == 0.5
    assert summary["top3_accuracy"] == 1.0
    assert rows[0]["true_label_rank"] == 1.0
    assert rows[1]["true_label_rank"] == 3.0


def test_alignment_loso_requires_at_least_three_subjects():
    with pytest.raises(ValueError, match="at least three subjects"):
        evaluate_alignment_loso(_synthetic_subjects(n_subjects=2))

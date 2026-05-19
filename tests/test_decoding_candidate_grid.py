from dataclasses import dataclass

import pytest

from neureptrace.decoding import CandidateGrid, candidate_grid_rows, expand_candidate_grid


@dataclass(frozen=True)
class CandidateConfig:
    window_center: float
    classifier: str
    normalization: str = "none"


def test_expand_candidate_grid_preserves_order_and_fixed_values():
    candidates = expand_candidate_grid(
        {"window_center": [0.15, 0.175], "classifier": ["svm", "lda"]},
        fixed={"normalization": "subject_baseline_z"},
    )

    assert candidates == (
        {"normalization": "subject_baseline_z", "window_center": 0.15, "classifier": "svm"},
        {"normalization": "subject_baseline_z", "window_center": 0.15, "classifier": "lda"},
        {"normalization": "subject_baseline_z", "window_center": 0.175, "classifier": "svm"},
        {"normalization": "subject_baseline_z", "window_center": 0.175, "classifier": "lda"},
    )


def test_expand_candidate_grid_constructs_dataclass_configs():
    candidates = expand_candidate_grid(
        {"window_center": [0.15, 0.175], "classifier": "svm"},
        fixed={"normalization": "subject_z"},
        factory=CandidateConfig,
    )

    assert candidates == (
        CandidateConfig(window_center=0.15, classifier="svm", normalization="subject_z"),
        CandidateConfig(window_center=0.175, classifier="svm", normalization="subject_z"),
    )


def test_candidate_grid_rows_adds_stable_indices():
    rows = candidate_grid_rows({"classifier": ["svm", "lda"]}, fixed={"window_center": 0.15}, start_index=1)

    assert rows == (
        {"candidate_index": 1, "window_center": 0.15, "classifier": "svm"},
        {"candidate_index": 2, "window_center": 0.15, "classifier": "lda"},
    )


def test_candidate_grid_dataclass_normalizes_once():
    grid = CandidateGrid({"classifier": ["svm"]}, fixed={"normalization": "none"})

    assert grid.rows() == ({"candidate_index": 1, "normalization": "none", "classifier": "svm"},)


def test_candidate_grid_validates_empty_dimensions_and_overlaps():
    with pytest.raises(ValueError, match="at least one value"):
        expand_candidate_grid({"classifier": []})
    with pytest.raises(ValueError, match="both fixed and varied"):
        expand_candidate_grid({"classifier": ["svm"]}, fixed={"classifier": "lda"})
    with pytest.raises(ValueError, match="non-empty"):
        candidate_grid_rows({"classifier": ["svm"]}, index_key="")

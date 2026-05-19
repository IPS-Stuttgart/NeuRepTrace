import numpy as np
import pandas as pd
import pytest

from neureptrace.features import FeatureDataset, SubjectFeatureSet


def test_subject_feature_set_validates_and_exposes_aliases():
    feature_set = SubjectFeatureSet(
        subject="sub-01",
        features=[[1.0, 2.0], [3.0, 4.0]],
        labels=["face", "scrambled"],
        groups=["run-1", "run-2"],
        trial_index=[10, 11],
        metadata=pd.DataFrame({"condition": ["a", "b"]}),
        feature_names=["MEG0111", "MEG0112"],
    )

    assert feature_set.n_trials == 2
    assert feature_set.n_features == 2
    assert feature_set.X.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert feature_set.y.tolist() == ["face", "scrambled"]
    assert feature_set.feature_names == ("MEG0111", "MEG0112")

    frame = feature_set.to_trial_frame()
    assert frame.to_dict("list") == {
        "subject": ["sub-01", "sub-01"],
        "label": ["face", "scrambled"],
        "group": ["run-1", "run-2"],
        "trial_index": [10, 11],
        "condition": ["a", "b"],
    }


def test_subject_feature_set_rejects_inconsistent_shapes():
    with pytest.raises(ValueError, match="2D"):
        SubjectFeatureSet(subject="sub-01", features=[1.0, 2.0], labels=["a", "b"])
    with pytest.raises(ValueError, match="labels has length"):
        SubjectFeatureSet(subject="sub-01", features=np.ones((2, 3)), labels=["a"])
    with pytest.raises(ValueError, match="groups has length"):
        SubjectFeatureSet(subject="sub-01", features=np.ones((2, 3)), labels=["a", "b"], groups=["run-1"])
    with pytest.raises(ValueError, match="metadata has 1 rows"):
        SubjectFeatureSet(subject="sub-01", features=np.ones((2, 3)), labels=["a", "b"], metadata=pd.DataFrame({"condition": ["a"]}))
    with pytest.raises(ValueError, match="feature_names has length"):
        SubjectFeatureSet(subject="sub-01", features=np.ones((2, 3)), labels=["a", "b"], feature_names=["x"])


def test_feature_dataset_indexes_subjects_and_stacks_uniform_features():
    sub01 = SubjectFeatureSet("sub-01", [[1.0, 2.0], [3.0, 4.0]], [1, 2], groups=["run-1", "run-1"], trial_index=[0, 1])
    sub02 = SubjectFeatureSet("sub-02", [[5.0, 6.0]], [2], groups=["run-2"], trial_index=[0])
    dataset = FeatureDataset([sub01, sub02], name="demo")

    assert dataset.subject_ids == ("sub-01", "sub-02")
    assert dataset.n_subjects == 2
    assert dataset.n_trials == 3
    assert dataset.n_features == 2
    assert dataset.get_subject("sub-02") is sub02

    stacked = dataset.stack()
    assert stacked.X.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert stacked.y.tolist() == [1, 2, 2]
    assert stacked.subjects.tolist() == ["sub-01", "sub-01", "sub-02"]
    assert stacked.groups.tolist() == ["run-1", "run-1", "run-2"]
    assert stacked.trial_index.tolist() == [0, 1, 0]


def test_feature_dataset_selection_preserves_requested_order():
    sub01 = SubjectFeatureSet("sub-01", [[1.0]], [1])
    sub02 = SubjectFeatureSet("sub-02", [[2.0]], [2])
    dataset = FeatureDataset([sub01, sub02])

    selected = dataset.select_subjects(["sub-02", "sub-01"])

    assert selected.subject_ids == ("sub-02", "sub-01")
    assert selected.stack().X.ravel().tolist() == [2.0, 1.0]


def test_feature_dataset_rejects_duplicate_subject_ids_and_nonuniform_stacking():
    sub01 = SubjectFeatureSet("sub-01", np.ones((2, 2)), [1, 2])
    duplicate = SubjectFeatureSet("sub-01", np.ones((1, 2)), [1])
    sub02 = SubjectFeatureSet("sub-02", np.ones((1, 3)), [1])

    with pytest.raises(ValueError, match="duplicates"):
        FeatureDataset([sub01, duplicate])

    dataset = FeatureDataset([sub01, sub02])
    assert dataset.has_uniform_feature_count is False
    with pytest.raises(ValueError, match="non-uniform"):
        _ = dataset.n_features
    with pytest.raises(ValueError, match="same number of feature columns"):
        dataset.stack()


def test_trial_frame_rejects_metadata_column_collisions():
    feature_set = SubjectFeatureSet(
        subject="sub-01",
        features=np.ones((1, 2)),
        labels=["a"],
        metadata=pd.DataFrame({"label": ["metadata-label"]}),
    )

    with pytest.raises(ValueError, match="overlap"):
        feature_set.to_trial_frame()


def test_feature_dataset_treats_tuple_identifiers_as_scalar_values():
    sub01 = SubjectFeatureSet(("study", "sub-01"), [[1.0], [2.0]], [("face", 1), ("object", 2)])
    dataset = FeatureDataset([sub01])

    stacked = dataset.stack()

    assert stacked.subjects.tolist() == [("study", "sub-01"), ("study", "sub-01")]
    assert stacked.labels.tolist() == [("face", 1), ("object", 2)]

import numpy as np
import pandas as pd
import pytest

from neureptrace.decoding.temporal_generalization import TemporalFeatureWindow
from neureptrace.loaders import (
    FeatureBlock,
    FeatureDataset,
    SubjectFeatureSet,
    load_feature_dataset,
)


def _window(center=0.1, values=None, labels=None):
    if values is None:
        values = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]]
    if labels is None:
        labels = ["a", "b", "a"]
    return TemporalFeatureWindow(
        center=center,
        start=center - 0.05,
        stop=center + 0.05,
        features=np.asarray(values, dtype=float),
        labels=np.asarray(labels),
        metadata={"source": "toy"},
    )


def test_feature_block_validates_and_exposes_temporal_windows():
    block = FeatureBlock(
        "main",
        [_window(0.0), _window(0.1)],
        metadata=pd.DataFrame({"trial": [0, 1, 2]}),
        sample_ids=["s0", "s1", "s2"],
        groups=["run1", "run1", "run2"],
    )

    assert block.name == "main"
    assert block.role == "analysis"
    assert block.n_samples == 3
    assert block.n_windows == 2
    assert block.window_centers == (0.0, 0.1)
    assert block.labels.tolist() == ["a", "b", "a"]


def test_feature_block_rejects_feature_label_mismatch():
    with pytest.raises(ValueError, match="feature rows must match labels"):
        FeatureBlock("bad", [_window(values=[[0.0], [1.0]], labels=[0, 1, 0])])


def test_feature_block_rejects_different_window_label_order():
    with pytest.raises(ValueError, match="labels must match the first window order"):
        FeatureBlock("bad", [_window(labels=[0, 1, 0]), _window(center=0.2, labels=[0, 0, 1])])


def test_feature_dataset_exposes_subject_blocks_and_calibration_role():
    main = FeatureBlock("main", [_window(0.0)])
    cue = FeatureBlock("cue", [_window(0.0)], role="calibration")
    subject = SubjectFeatureSet("S01", {"main": main, "cue": cue}, default_block="main")
    dataset = FeatureDataset([subject], metadata={"dataset": "toy"})

    assert dataset.subject_ids == ("S01",)
    assert dataset.get_subject("S01").get_block().name == "main"
    assert [block.name for block in dataset.get_subject("S01").calibration_blocks] == ["cue"]
    assert [(subject_id, block.name) for subject_id, block in dataset.iter_blocks(role="calibration")] == [("S01", "cue")]


def test_load_feature_dataset_accepts_dataset_object_loader_object_and_callable():
    dataset = FeatureDataset([SubjectFeatureSet("S01", {"main": FeatureBlock("main", [_window()])})])

    class Loader:
        def load(self):
            return dataset

    assert load_feature_dataset(dataset) is dataset
    assert load_feature_dataset(Loader()) is dataset
    assert load_feature_dataset(lambda: dataset) is dataset

    with pytest.raises(TypeError, match="expected FeatureDataset"):
        load_feature_dataset(lambda: "not-a-dataset")

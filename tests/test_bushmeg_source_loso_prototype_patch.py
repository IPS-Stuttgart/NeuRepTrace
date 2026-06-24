from __future__ import annotations

import numpy as np
import pandas as pd

from neureptrace.bushmeg_source_loso import (
    CandidateSpec,
    FeatureCache,
    PROTOTYPE_BASE_FEATURE_KINDS,
    PROTOTYPE_FEATURE_KINDS,
    SubjectEpochs,
    WindowSpec,
    _prepare_window_train_test_features,
    _window_evoked_dct_features,
)


def _dct_shape_subjects() -> dict[str, SubjectEpochs]:
    times = np.array([0.10, 0.20, 0.30, 0.40])
    labels = np.array([0, 0, 1, 1])
    subjects: dict[str, SubjectEpochs] = {}
    for subject_idx in range(3):
        data = np.zeros((4, 1, 4), dtype=np.float32)
        data[labels == 0, 0, :] = np.array([1.0, -1.0, 1.0, -1.0]) + subject_idx * 0.01
        data[labels == 1, 0, :] = np.array([-1.0, 1.0, -1.0, 1.0]) + subject_idx * 0.01
        subjects[str(subject_idx)] = SubjectEpochs(
            subject=str(subject_idx),
            data=data,
            times=times,
            metadata=pd.DataFrame(),
            labels=labels,
        )
    return subjects


def test_evoked_dct_prototype_is_mapped_to_dct_base_features() -> None:
    subjects = _dct_shape_subjects()
    window = WindowSpec(center=0.25, width=0.30)
    candidate = CandidateSpec(
        name="dct_proto",
        decoder="logistic",
        emission_mode="uncalibrated",
        feature_preprocessor="none",
        pca_components=None,
        classifier_param=1.0,
        temporal_bins=2,
        windows=(window,),
        feature_kind="evoked_dct_prototype",
    )

    train_features, _test_features = _prepare_window_train_test_features(
        subjects=subjects,
        cache=FeatureCache(subjects),
        candidate=candidate,
        train_subjects=["0", "1"],
        test_subject="2",
        window=window,
        n_classes=2,
    )
    expected_base = np.vstack(
        [
            _window_evoked_dct_features(subjects["0"].data, subjects["0"].times, window, temporal_bins=2),
            _window_evoked_dct_features(subjects["1"].data, subjects["1"].times, window, temporal_bins=2),
        ]
    )

    np.testing.assert_allclose(train_features[:, : expected_base.shape[1]], expected_base)


def test_all_non_xdawn_prototype_feature_kinds_have_explicit_base_mapping() -> None:
    missing = PROTOTYPE_FEATURE_KINDS.difference(PROTOTYPE_BASE_FEATURE_KINDS, {"xdawn_prototype"})

    assert not missing

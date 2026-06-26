from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neureptrace.bushmeg_cue_source_weights import CueSubjectData, _evoked_bin_means, _evoked_gfp_bins, cue_subject_feature_vector


def _cue_data_with_two_response_samples() -> tuple[np.ndarray, np.ndarray]:
    data = np.array(
        [
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]],
            [[2.0, 4.0, 6.0], [1.0, 3.0, 5.0]],
        ],
        dtype=np.float32,
    )
    times = np.array([-0.1, 0.1, 0.2], dtype=float)
    return data, times


def test_cue_evoked_helpers_reject_empty_temporal_bins() -> None:
    data, times = _cue_data_with_two_response_samples()

    with pytest.raises(ValueError, match="temporal_bins.*must not exceed"):
        _evoked_bin_means(data, times, (0.1, 0.2), temporal_bins=3)

    with pytest.raises(ValueError, match="temporal_bins.*must not exceed"):
        _evoked_gfp_bins(data, times, (0.1, 0.2), temporal_bins=3)


def test_cue_subject_feature_vector_rejects_empty_temporal_bins() -> None:
    data, times = _cue_data_with_two_response_samples()
    subject = CueSubjectData(subject="s1", data=data, times=times, metadata=pd.DataFrame())

    with pytest.raises(ValueError, match="temporal_bins.*must not exceed"):
        cue_subject_feature_vector(
            subject,
            feature_kinds=["evoked_mean"],
            response_window=(0.1, 0.2),
            temporal_bins=3,
        )

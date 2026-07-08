from __future__ import annotations

import numpy as np
import pytest

import neureptrace.bushmeg_all_protocols as all_protocols


def test_protocol3_prediction_output_rejects_negative_probability_entries() -> None:
    with pytest.raises(ValueError, match="negative probability"):
        all_protocols._coerce_protocol3_prediction_output(
            {"probabilities": np.asarray([[1.1, -0.1]])},
            n_evaluation_rows=1,
            classes=np.asarray(["face", "object"], dtype=object),
        )


def test_protocol3_prediction_output_clips_roundoff_negative_probability_entries() -> None:
    probabilities, predicted_indices, predicted_values = all_protocols._coerce_protocol3_prediction_output(
        {"probabilities": np.asarray([[1.0 + 1.0e-14, -1.0e-14]])},
        n_evaluation_rows=1,
        classes=np.asarray(["face", "object"], dtype=object),
    )

    assert probabilities.shape == (1, 2)
    assert np.all(probabilities >= 0.0)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert predicted_indices.tolist() == [0]
    assert predicted_values.tolist() == ["face"]

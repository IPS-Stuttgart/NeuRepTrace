import numpy as np

from neureptrace.decoding.source_roll import augment_source_with_feature_roll


def test_disabled_source_roll_drops_original_rows_when_requested():
    result = augment_source_with_feature_roll(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        np.array(["left", "right"]),
        config={"synthetic_per_class": 0, "preserve_original": False},
    )

    assert result.features.shape == (0, 2)
    assert result.labels.shape == (0,)
    assert result.synthetic_mask.shape == (0,)
    assert result.content_indices.shape == (0,)
    assert result.shifts.shape == (0,)
    assert result.metadata["source_feature_roll_n_output_rows"] == 0

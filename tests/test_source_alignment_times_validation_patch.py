import numpy as np
import pytest

from neureptrace.decoding.source_alignment import source_alignment_config


@pytest.mark.parametrize(
    "times",
    [
        True,
        False,
        np.bool_(True),
        [True],
        [0.088, False],
        (np.bool_(False),),
    ],
)
def test_source_alignment_times_reject_boolean_values(times):
    with pytest.raises(ValueError, match="alignment_times.*booleans"):
        source_alignment_config(method="procrustes", times=times)


def test_source_alignment_times_patch_preserves_numeric_sequences():
    config = source_alignment_config(method="procrustes", times=[0.088, "0.136"])

    assert config.same_decode_window is False
    assert config.times == (0.088, 0.136)

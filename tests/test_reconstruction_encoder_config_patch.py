import numpy as np
import pytest

from neureptrace.decoding.reconstruction_encoder import reconstruction_encoder_config


@pytest.mark.parametrize("value", ["false", "False", "0", "off", "no", 0, False, np.bool_(False)])
def test_reconstruction_encoder_standardize_false_aliases(value):
    assert reconstruction_encoder_config(standardize=value).standardize is False


@pytest.mark.parametrize("value", ["true", "True", "1", "on", "yes", 1, True, np.bool_(True)])
def test_reconstruction_encoder_standardize_true_aliases(value):
    assert reconstruction_encoder_config(standardize=value).standardize is True


@pytest.mark.parametrize("value", ["", "maybe", 2, -1])
def test_reconstruction_encoder_standardize_rejects_ambiguous_values(value):
    with pytest.raises(ValueError, match="standardize must be a boolean value"):
        reconstruction_encoder_config(standardize=value)

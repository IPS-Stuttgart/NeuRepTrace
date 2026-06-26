import numpy as np
import pytest

import neureptrace  # noqa: F401
from neureptrace.decoding import lora_few_shot
from neureptrace.decoding import semi_supervised_lora_few_shot


@pytest.mark.parametrize(
    ("validator", "kwargs"),
    [
        (lora_few_shot._positive_float, {"name": "lora_positive"}),
        (lora_few_shot._nonnegative_float, {"name": "lora_nonnegative"}),
        (lora_few_shot._bounded_float, {"name": "lora_bounded", "lower": 0.0, "upper": 1.0}),
        (semi_supervised_lora_few_shot._positive_float, {"name": "semi_positive"}),
        (semi_supervised_lora_few_shot._nonnegative_float, {"name": "semi_nonnegative"}),
        (semi_supervised_lora_few_shot._bounded_float, {"name": "semi_bounded", "lower": 0.0, "upper": 1.0}),
    ],
)
@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_lora_float_validators_reject_boolean_scalars(validator, kwargs, value):
    with pytest.raises(ValueError):
        validator(value, **kwargs)


def test_lora_float_validators_still_accept_numeric_values():
    assert lora_few_shot._positive_float("0.5", "lora_positive") == 0.5
    assert lora_few_shot._nonnegative_float(0.0, "lora_nonnegative") == 0.0
    assert lora_few_shot._bounded_float(1.0, "lora_bounded", lower=0.0, upper=1.0) == 1.0
    assert semi_supervised_lora_few_shot._positive_float("0.5", name="semi_positive") == 0.5
    assert semi_supervised_lora_few_shot._nonnegative_float(0.0, name="semi_nonnegative") == 0.0
    assert semi_supervised_lora_few_shot._bounded_float(1.0, name="semi_bounded", lower=0.0, upper=1.0) == 1.0

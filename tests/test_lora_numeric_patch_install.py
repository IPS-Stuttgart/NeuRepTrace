from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest


def test_lora_numeric_patch_is_installed_by_package_import() -> None:
    # Force a fresh target-module import while keeping the package initializer's
    # patch registration in effect.  Without automatic installation, this module
    # import leaves the LoRA float validators and probability normalizer unwrapped.
    sys.modules.pop("neureptrace.decoding.lora_few_shot", None)

    import neureptrace  # noqa: F401

    lora_few_shot = importlib.import_module("neureptrace.decoding.lora_few_shot")

    with pytest.raises(ValueError, match="positive finite"):
        lora_few_shot._positive_float(True, "lora_alpha")
    with pytest.raises(ValueError, match="non-negative finite"):
        lora_few_shot._nonnegative_float(np.bool_(False), "entropy_loss_weight")
    with pytest.raises(ValueError, match="finite two-dimensional rows"):
        lora_few_shot._normalize_probability_rows(np.asarray([0.2, 0.8]))

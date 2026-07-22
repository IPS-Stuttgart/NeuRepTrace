from __future__ import annotations

import numpy as np
import pandas as pd

import neureptrace  # noqa: F401  # installs runtime compatibility patches
from neureptrace.openneuro_alignment_compare import _first_nonempty


def test_alignment_compare_first_nonempty_accepts_structured_values() -> None:
    assert _first_nonempty(["ds-test"], "fallback") == "['ds-test']"
    assert _first_nonempty({"alignment": "strict_source_only"}, "fallback") == "{'alignment': 'strict_source_only'}"


def test_alignment_compare_first_nonempty_keeps_scalar_fallbacks() -> None:
    assert _first_nonempty("", None, "fallback") == "fallback"


def test_alignment_compare_first_nonempty_skips_scalar_missing_values() -> None:
    assert _first_nonempty(np.nan, np.float64(np.nan), pd.NA, np.datetime64("NaT"), "fallback") == "fallback"

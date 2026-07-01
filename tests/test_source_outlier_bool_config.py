from __future__ import annotations

import pytest

from neureptrace.decoding.source_outlier import source_outlier_config


def test_source_outlier_config_parses_string_boolean_controls() -> None:
    assert source_outlier_config(use_diagonal_scale="false").use_diagonal_scale is False
    assert source_outlier_config(use_diagonal_scale="0").use_diagonal_scale is False
    assert source_outlier_config(use_diagonal_scale="yes").use_diagonal_scale is True

    with pytest.raises(ValueError, match="use_diagonal_scale"):
        source_outlier_config(use_diagonal_scale="maybe")

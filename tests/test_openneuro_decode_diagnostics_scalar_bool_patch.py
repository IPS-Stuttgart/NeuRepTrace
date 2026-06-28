from __future__ import annotations

import pytest

from neureptrace import openneuro_decode_diagnostics as diagnostics


def test_optional_unique_bool_accepts_repeated_list_tokens() -> None:
    assert diagnostics._optional_unique_bool(["true", "yes", "1"], column="alignment_valid_for_benchmark") is True
    assert diagnostics._optional_unique_bool(["false", "no", "0"], column="alignment_valid_for_benchmark") is False


def test_optional_unique_bool_rejects_mixed_list_tokens_cleanly() -> None:
    with pytest.raises(ValueError, match="Inconsistent boolean provenance"):
        diagnostics._optional_unique_bool(["true", "false"], column="alignment_valid_for_benchmark")


def test_as_bool_rejects_mixed_list_tokens_cleanly() -> None:
    with pytest.raises(ValueError, match="Inconsistent boolean provenance"):
        diagnostics._as_bool(["true", "false"])


def test_provenance_value_accepts_list_manifest_values() -> None:
    value = diagnostics._provenance_value(
        {"alignment_valid_for_benchmark": ["true", "true"]},
        {"alignment_valid_for_benchmark": "false"},
        "alignment_valid_for_benchmark",
    )

    assert value == ["true", "true"]

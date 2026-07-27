from __future__ import annotations

import pytest

from neureptrace.decode_from_config import _decode_kwargs


def _config(*, include_decoding: bool, decoding_value=None):
    config = {
        "dataset": {"name": "demo"},
        "workflow": {
            "label_column": "legacy_condition",
            "classifier": "legacy_classifier",
        },
        "preprocessing": {},
        "outputs": {"summary_csv": "summary.csv"},
    }
    if include_decoding:
        config["decoding"] = decoding_value
    return config


@pytest.mark.parametrize("decoding_value", [{}, None])
def test_explicit_empty_decoding_does_not_fall_back_to_workflow(tmp_path, decoding_value):
    with pytest.raises(ValueError, match="decoding.label_column"):
        _decode_kwargs(
            _config(include_decoding=True, decoding_value=decoding_value),
            config_dir=tmp_path,
        )


def test_absent_decoding_keeps_legacy_workflow_fallback(tmp_path):
    kwargs = _decode_kwargs(
        _config(include_decoding=False),
        config_dir=tmp_path,
    )

    assert kwargs["label_column"] == "legacy_condition"
    assert kwargs["decoder"] == "legacy_classifier"

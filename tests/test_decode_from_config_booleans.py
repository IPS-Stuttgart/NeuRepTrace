from __future__ import annotations

from pathlib import Path

import pytest

from neureptrace.decode_from_config import _bool_value, _decode_kwargs


def _minimal_decode_config(tmp_path: Path, **decoding_overrides):
    return {
        "dataset": {"name": "synthetic"},
        "preprocessing": {},
        "decoding": {"label_column": "condition", **decoding_overrides},
        "outputs": {"base_dir": tmp_path.as_posix()},
    }


@pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", "none", "null", ""])
def test_bool_value_parses_false_like_strings(value: str) -> None:
    assert _bool_value(value, name="example.flag") is False


@pytest.mark.parametrize("value", ["true", "True", "1", "yes", "on"])
def test_bool_value_parses_true_like_strings(value: str) -> None:
    assert _bool_value(value, name="example.flag") is True


def test_bool_value_rejects_ambiguous_strings() -> None:
    with pytest.raises(ValueError, match="example.flag"):
        _bool_value("ambiguous", name="example.flag")


def test_decode_kwargs_keeps_quoted_false_switches_disabled(tmp_path: Path) -> None:
    kwargs = _decode_kwargs(
        _minimal_decode_config(
            tmp_path,
            tune_hyperparameters="false",
            label_shuffle_control="false",
        ),
        config_dir=tmp_path,
    )

    assert kwargs["tune_hyperparameters"] is False
    assert kwargs["label_shuffle_control"] is False


def test_decode_kwargs_accepts_quoted_true_switches(tmp_path: Path) -> None:
    kwargs = _decode_kwargs(
        _minimal_decode_config(
            tmp_path,
            tune_hyperparameters="true",
            label_shuffle_control="yes",
        ),
        config_dir=tmp_path,
    )

    assert kwargs["tune_hyperparameters"] is True
    assert kwargs["label_shuffle_control"] is True


def test_decode_kwargs_rejects_ambiguous_boolean_switches(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="decoding.label_shuffle_control"):
        _decode_kwargs(
            _minimal_decode_config(tmp_path, label_shuffle_control="ambiguous"),
            config_dir=tmp_path,
        )

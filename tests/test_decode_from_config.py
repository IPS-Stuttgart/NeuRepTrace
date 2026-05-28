from __future__ import annotations

from neureptrace.decode_from_config import _decode_kwargs


def test_decode_from_config_passes_label_shuffle_controls(tmp_path):
    kwargs = _decode_kwargs(
        {
            "dataset": {"name": "demo"},
            "decoding": {
                "label_column": "condition",
                "label_shuffle_control": True,
                "label_shuffle_seed": 29,
            },
            "preprocessing": {},
            "outputs": {"summary_csv": "summary.csv"},
        },
        config_dir=tmp_path,
    )

    assert kwargs["label_shuffle_control"] is True
    assert kwargs["label_shuffle_seed"] == 29

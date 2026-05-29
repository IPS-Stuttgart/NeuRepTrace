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
                "decode_window": [0.12, 0.248],
                "class_prior_correction": "train-uniform",
            },
            "preprocessing": {},
            "outputs": {"summary_csv": "summary.csv"},
        },
        config_dir=tmp_path,
    )

    assert kwargs["label_shuffle_control"] is True
    assert kwargs["label_shuffle_seed"] == 29
    assert kwargs["decode_window"] == (0.12, 0.248)
    assert kwargs["class_prior_correction"] == "train-uniform"


def test_decode_from_config_passes_ensemble_controls_for_ensemble_decoder(tmp_path):
    kwargs = _decode_kwargs(
        {
            "dataset": {"name": "demo"},
            "decoding": {
                "label_column": "condition",
                "classifier": "logistic-svm-ensemble",
                "ensemble_weights": "0.7,0.3",
                "ensemble_baseline_window": "none",
                "ensemble_baseline_group_columns": "subject,fold",
                "ensemble_min_probability": "1e-9",
            },
            "preprocessing": {},
            "outputs": {"summary_csv": "summary.csv"},
        },
        config_dir=tmp_path,
    )

    assert kwargs["ensemble_weights"] == (0.7, 0.3)
    assert kwargs["ensemble_baseline_window"] is None
    assert kwargs["ensemble_baseline_group_columns"] == ("subject", "fold")
    assert kwargs["ensemble_min_probability"] == 1e-9


def test_decode_from_config_ignores_ensemble_controls_for_regular_decoder(tmp_path):
    kwargs = _decode_kwargs(
        {
            "dataset": {"name": "demo"},
            "decoding": {
                "label_column": "condition",
                "classifier": "logistic",
                "ensemble_weights": [0.7, 0.3],
                "ensemble_baseline_window": [-0.35, -0.05],
            },
            "preprocessing": {},
            "outputs": {"summary_csv": "summary.csv"},
        },
        config_dir=tmp_path,
    )

    assert "ensemble_weights" not in kwargs
    assert "ensemble_baseline_window" not in kwargs

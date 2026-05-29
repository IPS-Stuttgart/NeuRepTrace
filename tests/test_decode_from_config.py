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


def test_decode_from_config_passes_outer_test_groups(tmp_path):
    kwargs = _decode_kwargs(
        {
            "dataset": {"name": "demo"},
            "decoding": {
                "label_column": "condition",
                "group_column": "subject",
                "outer_test_groups": "[sub-01,sub-07]",
            },
            "preprocessing": {},
            "outputs": {"summary_csv": "summary.csv"},
        },
        config_dir=tmp_path,
    )

    assert kwargs["outer_test_groups"] == ("sub-01", "sub-07")


def test_decode_from_config_passes_ensemble_controls_for_ensemble_decoder(tmp_path):
    kwargs = _decode_kwargs(
        {
            "dataset": {"name": "demo"},
            "decoding": {
                "label_column": "condition",
                "classifier": "logistic-svm-ensemble",
                "ensemble_weights": "0.7,0.3",
                "ensemble_source_decoders": "multinomial-logistic-weighted,linear_svm",
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
    assert kwargs["ensemble_source_decoders"] == ("multinomial-logistic-weighted", "linear_svm")
    assert kwargs["ensemble_baseline_window"] is None
    assert kwargs["ensemble_baseline_group_columns"] == ("subject", "fold")
    assert kwargs["ensemble_min_probability"] == 1e-9


def test_decode_from_config_accepts_workflow_style_unquoted_string_lists(tmp_path):
    kwargs = _decode_kwargs(
        {
            "dataset": {"name": "demo"},
            "decoding": {
                "label_column": "condition",
                "classifier": "logistic-svm-ensemble",
                "ensemble_source_decoders": "[multinomial-logistic-weighted,linear_svm,shrinkage_lda]",
            },
            "preprocessing": {},
            "outputs": {"summary_csv": "summary.csv"},
        },
        config_dir=tmp_path,
    )

    assert kwargs["ensemble_source_decoders"] == (
        "multinomial-logistic-weighted",
        "linear_svm",
        "shrinkage_lda",
    )


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

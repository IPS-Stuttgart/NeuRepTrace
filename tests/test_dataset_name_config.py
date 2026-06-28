from pathlib import Path

from neureptrace import bushmeg_source_loso, config_workflow, decode_from_config


def test_decode_from_config_unnamed_dataset_mapping_uses_scalar_name():
    kwargs = decode_from_config._decode_kwargs(
        {
            "dataset": {"epochs_path": "epochs.fif"},
            "decoding": {"label_column": "condition"},
        },
        config_dir=Path("/tmp"),
    )

    assert kwargs["dataset_name"] == ""


def test_decode_from_config_named_dataset_mapping_keeps_name():
    kwargs = decode_from_config._decode_kwargs(
        {
            "dataset": {"name": "ds006629", "epochs_path": "epochs.fif"},
            "decoding": {"label_column": "condition"},
        },
        config_dir=Path("/tmp"),
    )

    assert kwargs["dataset_name"] == "ds006629"


def test_decode_from_config_null_dataset_name_uses_template_token(tmp_path):
    config = {
        "dataset": {"name": None, "epochs_path": "epochs.fif"},
        "decoding": {"label_column": "condition"},
        "outputs": {
            "base_dir": "results/{dataset}",
            "metrics_csv": "{dataset}_metrics.csv",
        },
    }

    assert decode_from_config._resolve_output(
        config,
        config_dir=tmp_path,
        key="metrics_csv",
    ) == tmp_path / "results" / "dataset" / "dataset_metrics.csv"


def test_bushmeg_source_loso_null_dataset_name_uses_template_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = {
        "dataset": {"name": None, "type": "fieldtrip_mat"},
        "outputs": {
            "base_dir": "results/{dataset}",
            "summary_csv": "{dataset}_summary.csv",
        },
    }

    assert bushmeg_source_loso._resolve_output(
        config,
        config_dir=tmp_path,
        key="summary_csv",
        default="{dataset}_source_loso_summary.csv",
    ) == tmp_path / "results" / "dataset" / "dataset_summary.csv"


def test_legacy_config_workflow_unnamed_dataset_mapping_uses_scalar_name():
    kwargs = config_workflow._decode_kwargs(
        {
            "dataset": {"epochs": "epochs.fif"},
            "outputs": {"metrics_csv": "metrics.csv"},
            "decoding": {"label_column": "condition"},
        },
        config_path=Path("/tmp/workflow.yml"),
    )

    assert kwargs["dataset_name"] == ""


def test_legacy_config_workflow_named_dataset_mapping_keeps_name():
    kwargs = config_workflow._decode_kwargs(
        {
            "dataset": {"name": "bushmeg", "epochs": "epochs.fif"},
            "outputs": {"metrics_csv": "metrics.csv"},
            "decoding": {"label_column": "condition"},
        },
        config_path=Path("/tmp/workflow.yml"),
    )

    assert kwargs["dataset_name"] == "bushmeg"

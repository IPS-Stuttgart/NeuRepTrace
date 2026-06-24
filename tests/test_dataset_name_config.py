from pathlib import Path

from neureptrace import config_workflow, decode_from_config


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

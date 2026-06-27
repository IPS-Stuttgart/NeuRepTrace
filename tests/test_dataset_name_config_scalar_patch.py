from pathlib import Path

from neureptrace import config_workflow, decode_from_config


def test_decode_from_config_container_dataset_name_uses_empty_scalar():
    kwargs = decode_from_config._decode_kwargs(
        {
            "dataset": {"name": ["not", "scalar"], "epochs_path": "epochs.fif"},
            "decoding": {"label_column": "condition"},
        },
        config_dir=Path("/tmp"),
    )

    assert kwargs["dataset_name"] == ""


def test_legacy_config_workflow_container_dataset_name_uses_empty_scalar():
    kwargs = config_workflow._decode_kwargs(
        {
            "dataset": {"name": ["not", "scalar"], "epochs": "epochs.fif"},
            "outputs": {"metrics_csv": "metrics.csv"},
            "decoding": {"label_column": "condition"},
        },
        config_path=Path("/tmp/workflow.yml"),
    )

    assert kwargs["dataset_name"] == ""


def test_decode_from_config_numeric_dataset_name_is_stringified():
    kwargs = decode_from_config._decode_kwargs(
        {
            "dataset": {"name": 123, "epochs_path": "epochs.fif"},
            "decoding": {"label_column": "condition"},
        },
        config_dir=Path("/tmp"),
    )

    assert kwargs["dataset_name"] == "123"


def test_legacy_config_workflow_numeric_dataset_name_is_stringified():
    kwargs = config_workflow._decode_kwargs(
        {
            "dataset": {"name": 123, "epochs": "epochs.fif"},
            "outputs": {"metrics_csv": "metrics.csv"},
            "decoding": {"label_column": "condition"},
        },
        config_path=Path("/tmp/workflow.yml"),
    )

    assert kwargs["dataset_name"] == "123"

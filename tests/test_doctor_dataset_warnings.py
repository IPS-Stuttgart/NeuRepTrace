from __future__ import annotations

import json

from neureptrace.doctor import main as doctor_main


def test_doctor_reports_dataset_config_warnings(tmp_path, capsys):
    config = tmp_path / "dataset.yml"
    config.write_text(
        "\n".join(
            [
                "schema_version: neureptrace.dataset.v1",
                "dataset:",
                "  type: mne_epochs",
                "  epochs_file: sub-01_epo.fif",
            ]
        ),
        encoding="utf-8",
    )

    assert doctor_main(["--json", "--skip-optional", "--dataset-config", str(config)]) == 0
    payload = json.loads(capsys.readouterr().out)
    matching = [check for check in payload["checks"] if check["name"] == f"dataset-config:{config}"]
    assert matching
    assert matching[0]["status"] == "warning"
    assert "No decoding.label_column" in matching[0]["details"]
    assert payload["summary"]["warning"] >= 1

    assert doctor_main(["--json", "--skip-optional", "--fail-on-warnings", "--dataset-config", str(config)]) == 1

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from neureptrace import cli
from neureptrace.doctor import main as doctor_main
from neureptrace.doctor import run_checks, summarize_checks


def test_doctor_smoke_json_output(capsys):
    assert doctor_main(["--json", "--skip-optional"]) == 0

    payload = json.loads(capsys.readouterr().out)
    names = {check["name"] for check in payload["checks"]}
    assert "python" in names
    assert "neureptrace" in names
    assert payload["summary"]["error"] == 0


def test_doctor_required_module_failure(capsys):
    code = doctor_main(["--json", "--skip-optional", "--require-module", "definitely_missing_neureptrace_dependency"])

    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    failing = [check for check in payload["checks"] if check["name"] == "module:definitely_missing_neureptrace_dependency"]
    assert failing
    assert failing[0]["status"] == "error"


def test_doctor_validates_dataset_config(tmp_path, capsys):
    config = tmp_path / "dataset.yml"
    config.write_text(
        "\n".join(
            [
                "schema_version: neureptrace.dataset.v1",
                "dataset:",
                "  type: mne_epochs",
                "  epochs_file: sub-01_epo.fif",
                "decoding:",
                "  label_column: condition",
            ]
        ),
        encoding="utf-8",
    )

    assert doctor_main(["--json", "--skip-optional", "--dataset-config", str(config)]) == 0
    payload = json.loads(capsys.readouterr().out)
    matching = [check for check in payload["checks"] if check["name"] == f"dataset-config:{config}"]
    assert matching
    assert matching[0]["status"] == "ok"


def test_doctor_reports_invalid_dataset_config(tmp_path, capsys):
    config = tmp_path / "dataset.yml"
    config.write_text("dataset:\n  type: unsupported\n", encoding="utf-8")

    assert doctor_main(["--json", "--skip-optional", "--dataset-config", str(config)]) == 1
    payload = json.loads(capsys.readouterr().out)
    matching = [check for check in payload["checks"] if check["name"] == f"dataset-config:{config}"]
    assert matching
    assert matching[0]["status"] == "error"


def test_doctor_summary_counts_statuses():
    checks = run_checks(include_optional=False, required_modules=["sys"])
    summary = summarize_checks(checks)

    assert summary["ok"] >= 1
    assert summary["error"] == 0


def test_grouped_cli_exposes_doctor():
    assert cli.COMMAND_MODULES["doctor"] == "neureptrace.doctor"
    assert cli.COMMAND_MODULES["env"] == "neureptrace.doctor"
    assert cli.COMMAND_MODULES["check"] == "neureptrace.doctor"


def test_poetry_scripts_expose_doctor():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["tool"]["poetry"]["scripts"]

    assert scripts["neureptrace-doctor"] == "neureptrace.doctor:main"

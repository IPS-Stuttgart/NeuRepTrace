from __future__ import annotations

import json
import tomllib
from pathlib import Path

from neureptrace import cli
import neureptrace.doctor as doctor
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


def test_doctor_can_validate_entry_points(capsys):
    code = doctor_main(["--json", "--skip-optional", "--check-entry-points"])

    payload = json.loads(capsys.readouterr().out)
    names = {check["name"] for check in payload["checks"]}
    assert "grouped-command:doctor" in names
    assert "grouped-command:mne-time-decode" in names
    assert any(name.startswith("entry-point:neureptrace") for name in names)
    assert payload["summary"]["error"] == 0
    assert code in {0, 1}
    if payload["summary"]["warning"] == 0:
        assert code == 0


def test_run_checks_can_validate_entry_points_directly():
    checks = run_checks(include_optional=False, check_entry_points=True)
    names = {check.name for check in checks}

    assert "grouped-command:doctor" in names
    assert "grouped-command:mne-time-decode" in names
    assert not [check for check in checks if check.status == "error"]


def test_grouped_cli_exposes_doctor():
    assert cli.COMMAND_MODULES["doctor"] == "neureptrace.doctor"
    assert cli.COMMAND_MODULES["env"] == "neureptrace.doctor"
    assert cli.COMMAND_MODULES["check"] == "neureptrace.doctor"


def test_poetry_scripts_expose_doctor():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["tool"]["poetry"]["scripts"]

    assert scripts["neureptrace-doctor"] == "neureptrace.doctor:main"


def test_import_module_for_diagnostics_reports_import_time_failure(monkeypatch):
    def fake_import_module(module_name: str):
        raise ImportError(f"{module_name} failed to load native extension")

    monkeypatch.setattr(doctor.importlib, "import_module", fake_import_module)

    available, error = doctor._import_module_for_diagnostics("broken_backend")

    assert available is False
    assert error == "ImportError: broken_backend failed to load native extension"


def test_check_dependency_reports_installed_but_broken_required_module(monkeypatch):
    monkeypatch.setattr(doctor, "_distribution_version", lambda distribution_name: "1.2.3")
    monkeypatch.setattr(
        doctor,
        "_import_module_for_diagnostics",
        lambda module_name: (False, "ImportError: libexample.so: cannot open shared object file"),
    )

    check = doctor._check_dependency("example-dist", "example", required=True)

    assert check.name == "dependency:example-dist"
    assert check.status == "error"
    assert check.required is True
    assert "required module 'example' is not importable" in check.details
    assert "libexample.so" in check.details


def test_check_dependency_reports_installed_but_broken_optional_module_as_warning(monkeypatch):
    monkeypatch.setattr(doctor, "_distribution_version", lambda distribution_name: "1.2.3")
    monkeypatch.setattr(
        doctor,
        "_import_module_for_diagnostics",
        lambda module_name: (False, "RuntimeError: ABI mismatch"),
    )

    check = doctor._check_dependency("optional-dist", "optional_backend", required=False)

    assert check.name == "dependency:optional-dist"
    assert check.status == "warning"
    assert check.required is False
    assert "optional module 'optional_backend' is not importable" in check.details
    assert "ABI mismatch" in check.details


def test_check_required_module_reports_import_time_failure(monkeypatch):
    monkeypatch.setattr(
        doctor,
        "_import_module_for_diagnostics",
        lambda module_name: (False, "ValueError: incompatible binary interface"),
    )

    check = doctor._check_required_module("project_adapter")

    assert check.name == "module:project_adapter"
    assert check.status == "error"
    assert "Required module 'project_adapter' is not importable" in check.details
    assert "incompatible binary interface" in check.details

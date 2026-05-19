from __future__ import annotations

import json

import pytest

from neureptrace.workflow import main as workflow_main
from neureptrace.workflow import validate_main
from neureptrace.workflow_spec import WorkflowSpecError, check_workflow_files, load_workflow_spec, parse_workflow_spec, workflow_spec_to_dict


def _workflow_mapping() -> dict[str, object]:
    return {
        "version": 1,
        "workflow": {"kind": "cross_subject_decoding", "name": "example"},
        "dataset": {
            "id": "pymegdec_main_cue",
            "root": "data",
            "participants": [1, 2],
            "files": {
                "main": {"loader": "pymegdec.fieldtrip_mat", "pattern": "Part{participant}Data.mat", "metadata": "stimulus"},
                "cue": {"loader": "pymegdec.fieldtrip_mat", "pattern": "Part{participant}CueData.mat", "metadata": "stimulus"},
            },
            "metadata": {"stimulus": {"path": "stimulus_metadata.csv", "key": "stimulus_id"}},
        },
        "features": {"label": "stimulus_id"},
        "model": {"classifier": "multiclass-svm", "params": {"pca_components": 100}},
        "evaluation": {"split": "leave_one_subject_out", "train": "main", "test": "cue"},
        "outputs": {"root": "outputs", "tables": {"predictions": "predictions.csv"}},
    }


def test_parse_workflow_spec_normalizes_aliases() -> None:
    spec = parse_workflow_spec(_workflow_mapping())

    assert spec.workflow.kind == "cross_subject_decoding"
    assert spec.workflow.name == "example"
    assert spec.dataset.files["main"].loader == "pymegdec.fieldtrip_mat"
    assert spec.model is not None
    assert spec.model.name == "multiclass-svm"
    assert spec.evaluation is not None
    assert spec.evaluation.scheme == "leave_one_subject_out"
    assert "source" not in workflow_spec_to_dict(spec)


def test_load_workflow_spec_from_json(tmp_path) -> None:
    config = tmp_path / "workflow.json"
    config.write_text(json.dumps(_workflow_mapping()), encoding="utf-8")

    spec = load_workflow_spec(config)

    assert spec.source == config
    assert spec.dataset.id == "pymegdec_main_cue"


def test_rejects_file_role_without_loader() -> None:
    mapping = _workflow_mapping()
    mapping["dataset"]["files"]["main"].pop("loader")

    with pytest.raises(WorkflowSpecError, match="dataset.files.main.loader"):
        parse_workflow_spec(mapping)


def test_check_workflow_files_expands_participant_patterns(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    for name in ["Part1Data.mat", "Part2Data.mat", "Part1CueData.mat", "Part2CueData.mat", "stimulus_metadata.csv"]:
        (data / name).write_text("", encoding="utf-8")
    config = tmp_path / "workflow.json"
    config.write_text(json.dumps(_workflow_mapping()), encoding="utf-8")
    spec = load_workflow_spec(config)

    assert check_workflow_files(spec) == []

    (data / "Part2CueData.mat").unlink()
    assert "Part2CueData.mat" in " | ".join(check_workflow_files(spec))


def test_validate_workflow_cli_reports_ok(tmp_path, capsys) -> None:
    config = tmp_path / "workflow.json"
    config.write_text(json.dumps(_workflow_mapping()), encoding="utf-8")

    assert validate_main([str(config)]) == 0

    out = capsys.readouterr().out
    assert "ok" in out
    assert "cross_subject_decoding" in out


def test_workflow_schema_command_prints_schema(capsys) -> None:
    assert workflow_main(["schema"]) == 0

    schema = json.loads(capsys.readouterr().out)
    assert schema["properties"]["version"]["const"] == 1

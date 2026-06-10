from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from neureptrace.dataset_config import iter_dataset_files, load_config, validate_dataset_config
from neureptrace.openneuro_meg import (
    DATASET_SPECS,
    RunFiles,
    _derive_metadata,
    _drop_non_epochable_metadata,
    _filter_metadata,
    download_selected_files,
    expected_relative_files,
    invalid_raw_fif_files,
    parse_runs,
    parse_subjects,
    run_files,
    selected_download_includes,
    subject_label,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
OPENNEURO_DECODE_CONFIGS = {
    "ds000117": "ds000117_face_recognition.yml",
    "ds004276": "ds004276_words.yml",
    "ds006629": "ds006629_singsing.yml",
    "ds004330": "ds004330_object_drawing.yml",
}


def test_openneuro_decode_configs_match_staged_output_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Keep staging specs, decode configs, and LOSO split counts in sync."""

    monkeypatch.setenv("NEUREPTRACE_OPENNEURO_STAGED_DIR", str(tmp_path))

    assert set(OPENNEURO_DECODE_CONFIGS) == set(DATASET_SPECS)
    for dataset_id, config_name in OPENNEURO_DECODE_CONFIGS.items():
        spec = DATASET_SPECS[dataset_id]
        config_path = REPO_ROOT / "configs" / "openneuro" / config_name
        config = load_config(config_path)

        assert validate_dataset_config(config, base_dir=config_path.parent, check_files=False) == []
        participants = tuple(parse_subjects(spec, config["participants"]["ids"]))
        assert participants == spec.default_subjects
        assert config["decoding"]["label_column"] == "condition"
        assert config["decoding"]["group_column"] == "subject"
        assert config["decoding"]["n_splits"] == len(spec.default_subjects)

        expected_epochs = [
            tmp_path / dataset_id / f"{subject_label(spec, subject)}_{dataset_id}_{spec.name}_epo.fif"
            for subject in spec.default_subjects
        ]
        assert iter_dataset_files(config, base_dir=config_path.parent) == expected_epochs


@pytest.mark.parametrize("dataset_id", sorted(DATASET_SPECS))
def test_expected_relative_files_cover_default_subjects_runs_and_sidecars(dataset_id: str):
    spec = DATASET_SPECS[dataset_id]
    includes = expected_relative_files(dataset_id)
    expected_run_files = len(spec.default_subjects) * len(spec.runs)

    assert sum(path.endswith("_meg.fif") for path in includes) == expected_run_files
    assert sum(path.endswith("_events.tsv") for path in includes) == expected_run_files
    if dataset_id == "ds004276":
        assert sum("/beh/" in path and path.endswith("_beh.tsv") for path in includes) == len(spec.default_subjects)
    else:
        assert all("/beh/" not in path for path in includes)


def test_openneuro_workflow_exposes_every_configured_dataset():
    workflow = (REPO_ROOT / ".github" / "workflows" / "openneuro-meg-loso.yml").read_text(encoding="utf-8")

    assert "run-name: OpenNeuro MEG LOSO" in workflow
    assert "default: github-hosted" in workflow
    assert '"ds000117": ("1,2,3", "01,02")' in workflow
    assert '"ds004276": ("1-3", "all")' in workflow
    assert '"ds006629": ("1,2,4", "0")' in workflow
    assert '"ds004330": ("1,2,4", "01,02,03")' in workflow
    assert "GitHub-hosted starts immediately" in workflow
    assert "Cache OpenNeuro raw files on GitHub-hosted runners" in workflow
    assert "Resolve GitHub-hosted OpenNeuro cache keys" in workflow
    assert "safe_cache_token" in workflow
    assert "Cache staged OpenNeuro epochs on GitHub-hosted runners" in workflow
    assert "download-selected" in workflow
    assert "--batch-size 24" in workflow
    assert "--max-concurrent-downloads 2" in workflow
    assert "NeuRepTrace LOSO decode still running" in workflow
    assert "Check selected staged epochs" in workflow
    assert "raw-file check/download can be skipped" in workflow
    assert "steps.staged_check.outputs.ready != 'true'" in workflow
    assert "steps.openneuro_cache_keys.outputs.subjects" in workflow
    assert "steps.openneuro_cache_keys.outputs.runs" in workflow
    assert "steps.openneuro_cache_keys.outputs.cap" in workflow
    assert "steps.openneuro_cache_keys.outputs.stage_seed" in workflow
    assert "stage-seed-${{ steps.openneuro_cache_keys.outputs.stage_seed }}" in workflow
    assert "CONFIG_OVERRIDES_CACHE_INPUT" in workflow
    assert "workflow.stage_seed" in workflow
    assert "STAGE_SEED_INPUT" in workflow
    assert '--seed "$STAGE_SEED_INPUT"' in workflow
    assert "subjects-${{ inputs.subjects }}" not in workflow
    assert "stage_overwrite" in workflow
    assert '"stage_overwrite": runner_environment == "self-hosted" and not no_download' in workflow
    assert 'runner_environment == "github-hosted" or no_download' in workflow
    assert 'if [[ "$RUNNER_ENVIRONMENT" == "self-hosted" && "$NO_DOWNLOAD" != "true" ]]; then' in workflow
    assert "stage_args+=(--overwrite)" in workflow
    assert "OPENNEURO_ARTIFACT_SUFFIX" in workflow
    assert "OPENNEURO_OUTPUT_SUFFIX" in workflow
    assert "run_manifest.json" in workflow
    assert "workflow_quality_summary.csv" in workflow
    assert "source_calibration" in workflow
    assert "decoding.source_calibration=$SOURCE_CALIBRATION" in workflow
    assert "temporal_smoothing" in workflow
    assert "temporal_smoothing_mode" in workflow
    assert "poststimulus_forward_only" in workflow
    assert '--mode "$TEMPORAL_SMOOTHING_MODE"' in workflow
    assert "ensemble_source_temperatures" in workflow
    assert "decoding.ensemble_source_temperatures" in workflow
    assert "python -m neureptrace.temporal_smoothing" in workflow
    assert "decode/temporal_smoothing/diagnostics" in workflow
    assert "strict_source_oof_nonnegative" in workflow
    assert "strict_source_oof_classwise_nonnegative" in workflow
    assert "strict_source_oof_logit_stacker" in workflow
    assert "response_window_poststimulus_forward" in workflow
    assert "decoder_source_oof_nonnegative" in workflow
    assert "ensemble_source_observations.csv" in workflow
    assert "decoding.source_time_selection=source_oof_time_weighted_logits" in workflow
    assert "decoding.source_time_selection=source_oof_classwise_time_weighted_logits" in workflow
    assert "decoding.source_time_selection=source_oof_logit_stacker" in workflow
    assert "decoding.source_time_selection_times=[$RESPONSE_WINDOW_TIMES]" in workflow
    assert "$response_window_mode is produced inside each outer decode fold" in workflow
    assert "resolve-matrix" in workflow
    assert "workflow.outer_test_group_shards_json" in workflow
    assert "outer_test_groups: ${{ fromJSON(needs.resolve-matrix.outputs.outer_test_group_shards_json) }}" in workflow
    assert "OUTER_TEST_GROUPS_SHARD" in workflow
    assert "aggregate-openneuro-shards" in workflow
    assert "needs.openneuro-meg-loso.result != 'cancelled'" in workflow
    assert "actions/download-artifact@v8" in workflow
    assert "merge-multiple: false" in workflow
    assert "OPENNEURO_SHARD_MATRIX_RESULT" in workflow
    assert "OUTER_TEST_GROUP_SHARDS_JSON" in workflow
    assert "shard_aggregation_status.json" in workflow
    assert 'artifact_name.endswith("-shard-aggregate")' in workflow
    assert 'artifact_name.endswith(f"-shard-{shard}")' in workflow
    assert "--aggregate-out \"$OPENNEURO_AGGREGATE_OUTPUT_DIR\"" in workflow
    assert "-shard-aggregate" in workflow
    assert "OPENNEURO_CONFIG" in workflow
    assert "artifact_name" in workflow
    assert "outer_test_groups_shard" in workflow
    assert "outer_test_groups" in workflow
    assert "decoding.outer_test_groups=[sub-01]" in workflow
    assert "decoding.outer_test_groups=" in workflow
    assert 'args+=(--set "decoding.outer_test_groups=[$OUTER_TEST_GROUPS_SHARD]")' in workflow
    assert "python -m neureptrace.openneuro_decode_diagnostics" in workflow
    assert "openneuro-meg-${{ inputs.dataset }}-${{ inputs.mode }}${{ env.OPENNEURO_ARTIFACT_SUFFIX }}" in workflow
    assert "format('-shard-{0}', matrix.outer_test_groups)" in workflow
    assert "outputs/openneuro_${{ inputs.dataset }}_${{ inputs.mode }}*/**" in workflow
    for dataset_id, config_name in OPENNEURO_DECODE_CONFIGS.items():
        assert f"- {dataset_id}" in workflow
        assert f'"{dataset_id}": "{config_name}"' in workflow


def test_openneuro_auxiliary_workflows_use_alignment_valid_smoke_cohorts():
    safe_cache = (REPO_ROOT / ".github" / "workflows" / "openneuro-meg-loso-safe-cache.yml").read_text(encoding="utf-8")
    decode_only = (REPO_ROOT / ".github" / "workflows" / "openneuro-meg-decode-only.yml").read_text(encoding="utf-8")

    for workflow in (safe_cache, decode_only):
        assert '"ds004276":' in workflow
        assert "1-3" in workflow
        assert "1,2,4" in workflow


def test_openneuro_safe_cache_workflow_reuses_complete_no_download_staging():
    workflow = (REPO_ROOT / ".github" / "workflows" / "openneuro-meg-loso-safe-cache.yml").read_text(encoding="utf-8")

    assert '"stage_overwrite": runner_environment == "self-hosted" and not no_download' in workflow
    assert 'runner_environment == "github-hosted" or no_download' in workflow
    assert "steps.staged_check.outputs.ready != 'true'" in workflow
    assert 'if [[ "$RUNNER_ENVIRONMENT" == "self-hosted" && "$NO_DOWNLOAD" != "true" ]]; then' in workflow


def test_expected_relative_files_include_singsing_raw_and_events():
    assert expected_relative_files("ds006629", subjects="1,2", runs="0") == [
        "sub-01/meg/sub-01_task-MMNHCS_run-0_meg.fif",
        "sub-01/meg/sub-01_task-MMNHCS_run-0_events.tsv",
        "sub-02/meg/sub-02_task-MMNHCS_run-0_meg.fif",
        "sub-02/meg/sub-02_task-MMNHCS_run-0_events.tsv",
    ]


def test_expected_relative_files_include_ds000117_session_raw_and_events():
    assert expected_relative_files("ds000117", subjects="1", runs="1,2") == [
        "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_meg.fif",
        "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-01_events.tsv",
        "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_meg.fif",
        "sub-01/ses-meg/meg/sub-01_ses-meg_task-facerecognition_run-02_events.tsv",
    ]


def test_selected_download_includes_skip_existing_files(tmp_path: Path):
    existing = tmp_path / "sub-01" / "meg" / "sub-01_task-MMNHCS_run-0_events.tsv"
    existing.parent.mkdir(parents=True)
    existing.write_text("onset\tduration\ttrial_type\n", encoding="utf-8")

    includes = selected_download_includes("ds006629", bids_root=tmp_path, subjects="1", runs="0")

    assert includes == ("sub-01/meg/sub-01_task-MMNHCS_run-0_meg.fif",)


def test_download_selected_files_batches_openneuro_includes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is True
        commands.append(list(command))
        target_dir = Path(command[command.index("--target-dir") + 1])
        include_indices = [index + 1 for index, token in enumerate(command) if token == "--include"]
        for index in include_indices:
            path = target_dir / command[index]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"ok")

    monkeypatch.setattr("neureptrace.openneuro_meg.subprocess.run", fake_run)
    manifest = tmp_path / "includes.txt"

    missing = download_selected_files(
        "ds006629",
        bids_root=tmp_path,
        subjects="1,2",
        runs="0",
        include_manifest=manifest,
        batch_size=3,
        max_attempts=1,
        max_concurrent_downloads=2,
    )

    assert missing == ()
    assert len(commands) == 2
    assert all(command[:4] == ["openneuro-py", "download", "--dataset", "ds006629"] for command in commands)
    assert all(command[command.index("--max-concurrent-downloads") + 1] == "2" for command in commands)
    assert [command.count("--include") for command in commands] == [3, 1]
    assert manifest.read_text(encoding="utf-8").splitlines() == expected_relative_files("ds006629", subjects="1,2", runs="0")


def test_download_selected_files_splits_partial_successful_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    commands: list[list[str]] = []

    def fake_run(command, *, check):
        assert check is True
        commands.append(list(command))
        target_dir = Path(command[command.index("--target-dir") + 1])
        include_indices = [index + 1 for index, token in enumerate(command) if token == "--include"]
        path = target_dir / command[include_indices[-1]]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")

    monkeypatch.setattr("neureptrace.openneuro_meg.subprocess.run", fake_run)

    missing = download_selected_files(
        "ds006629",
        bids_root=tmp_path,
        subjects="1,2",
        runs="0",
        batch_size=4,
        max_attempts=1,
    )

    assert missing == ()
    assert [command.count("--include") for command in commands] == [4, 1, 1, 1]
    for relative_path in expected_relative_files("ds006629", subjects="1,2", runs="0"):
        assert (tmp_path / relative_path).is_file()


def test_ds004276_word_metadata_joins_behavior_file(tmp_path: Path):
    behavior = pd.DataFrame(
        {
            "Event_Type": ["Sound", "Sound", "Picture"],
            "Code": ["cat", "elephant", "probe"],
            "Trial": [1, 2, 3],
            "Stim_Type": ["other", "other", "other"],
        }
    )
    behavior_path = tmp_path / "sub-001_task-words_beh.tsv"
    behavior.to_csv(behavior_path, sep="\t", index=False)
    events = pd.DataFrame({"onset": [0.1, 0.2], "duration": [0.0, 0.0], "trial_type": ["item", "item"]})

    metadata = _derive_metadata(
        DATASET_SPECS["ds004276"],
        RunFiles(
            subject="sub-001",
            run=None,
            raw_path=tmp_path / "sub-001_task-words_meg.fif",
            events_path=tmp_path / "sub-001_task-words_events.tsv",
            behavior_path=behavior_path,
        ),
        events,
    )
    filtered = _filter_metadata(
        metadata,
        label_column="word_length_binary",
        include_labels=None,
        max_events_per_label=None,
        selection="random",
        seed=13,
    )

    assert filtered["word"].tolist() == ["cat", "elephant"]
    assert filtered["condition"].tolist() == ["short", "long"]


def test_ds004276_word_metadata_ignores_probe_events(tmp_path: Path):
    behavior = pd.DataFrame(
        {
            "Event_Type": ["Sound", "Sound"],
            "Code": ["cat", "elephant"],
            "Trial": [1, 2],
            "Stim_Type": ["other", "other"],
        }
    )
    behavior_path = tmp_path / "sub-001_task-words_beh.tsv"
    behavior.to_csv(behavior_path, sep="\t", index=False)
    events = pd.DataFrame(
        {
            "onset": [0.1, 0.2, 0.3],
            "duration": [0.0, 0.0, 0.0],
            "trial_type": ["item", "probe", "item_post_probe"],
        }
    )

    metadata = _derive_metadata(
        DATASET_SPECS["ds004276"],
        RunFiles(
            subject="sub-001",
            run=None,
            raw_path=tmp_path / "sub-001_task-words_meg.fif",
            events_path=tmp_path / "sub-001_task-words_events.tsv",
            behavior_path=behavior_path,
        ),
        events,
    )

    assert metadata["trial_type"].tolist() == ["item", "item_post_probe"]
    assert metadata["word"].tolist() == ["cat", "elephant"]


def test_drop_non_epochable_metadata_removes_out_of_bounds_events():
    info = mne.create_info(["MEG001"], sfreq=10.0, ch_types=["mag"])
    raw = mne.io.RawArray(np.zeros((1, 100)), info, verbose="error")
    metadata = pd.DataFrame(
        {
            "onset": [0.1, 1.0, 9.9],
            "condition": ["a", "b", "c"],
        }
    )

    filtered = _drop_non_epochable_metadata(raw, metadata, label_column="condition", tmin=-0.2, tmax=0.5)

    assert filtered["condition"].tolist() == ["b"]


def test_ds004330_derives_stimulus_form_and_id():
    metadata = _derive_metadata(
        DATASET_SPECS["ds004330"],
        RunFiles(
            subject="sub-01",
            run="01",
            raw_path=Path("sub-01_ses-01_task-main_run-01_meg.fif"),
            events_path=Path("sub-01_ses-01_task-main_run-01_events.tsv"),
        ),
        pd.DataFrame({"onset": [1.0], "duration": [0.45], "trial_type": ["Drawing_26"]}),
    )

    assert metadata.loc[0, "stimulus_form"] == "Drawing"
    assert metadata.loc[0, "stimulus_id"] == "26"
    assert metadata.loc[0, "stimulus_modality"] == "drawing"


def test_ds000117_uses_face_stim_type_labels():
    metadata = _derive_metadata(
        DATASET_SPECS["ds000117"],
        RunFiles(
            subject="sub-01",
            run="01",
            raw_path=Path("sub-01_ses-meg_task-facerecognition_run-01_meg.fif"),
            events_path=Path("sub-01_ses-meg_task-facerecognition_run-01_events.tsv"),
        ),
        pd.DataFrame(
            {
                "onset": [24.2, 36.5, 46.0],
                "stim_type": ["Unfamiliar", "Famous", "Scrambled"],
                "trigger": [13, 5, 17],
                "stim_file": ["meg/u032.bmp", "meg/f123.bmp", "meg/s150.bmp"],
            }
        ),
    )
    filtered = _filter_metadata(
        metadata,
        label_column=DATASET_SPECS["ds000117"].default_label_column,
        include_labels=None,
        max_events_per_label=None,
        selection="first",
        seed=13,
    )

    assert filtered["condition"].tolist() == ["Unfamiliar", "Famous", "Scrambled"]
    assert filtered["stimulus_file"].tolist() == ["meg/u032.bmp", "meg/f123.bmp", "meg/s150.bmp"]
    assert filtered["stimulus_id"].tolist() == ["u032", "f123", "s150"]
    assert filtered["event_code"].tolist() == ["13", "5", "17"]


def test_openneuro_subject_and_path_formatting():
    assert parse_subjects(DATASET_SPECS["ds004276"], "1-2") == (1, 2)
    assert parse_runs(DATASET_SPECS["ds004330"], "1,2,3") == ("01", "02", "03")
    assert parse_runs(DATASET_SPECS["ds000117"], "1,2") == ("01", "02")
    files = run_files(DATASET_SPECS["ds004276"], Path("root"), 1, None)
    assert files.raw_path == Path("root/sub-001/meg/sub-001_task-words_meg.fif")
    assert files.behavior_path == Path("root/sub-001/beh/sub-001_task-words_beh.tsv")


def test_invalid_raw_fif_files_reports_unreadable_cache_entry(tmp_path: Path):
    files = run_files(DATASET_SPECS["ds006629"], tmp_path, 1, "0")
    files.raw_path.parent.mkdir(parents=True)
    files.raw_path.write_bytes(b"not a fif file")

    invalid = invalid_raw_fif_files("ds006629", bids_root=tmp_path, subjects="1", runs="0")

    assert [path for path, _reason in invalid] == [files.raw_path]

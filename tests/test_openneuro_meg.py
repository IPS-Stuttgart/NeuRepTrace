from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neureptrace.openneuro_meg import (
    DATASET_SPECS,
    RunFiles,
    _derive_metadata,
    _drop_non_epochable_metadata,
    _filter_metadata,
    expected_relative_files,
    invalid_raw_fif_files,
    parse_runs,
    parse_subjects,
    run_files,
)


def test_expected_relative_files_include_singsing_raw_and_events():
    assert expected_relative_files("ds006629", subjects="1,2", runs="0") == [
        "sub-01/meg/sub-01_task-MMNHCS_run-0_meg.fif",
        "sub-01/meg/sub-01_task-MMNHCS_run-0_events.tsv",
        "sub-02/meg/sub-02_task-MMNHCS_run-0_meg.fif",
        "sub-02/meg/sub-02_task-MMNHCS_run-0_events.tsv",
    ]


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
            "Event_Type": ["Sound", "Picture", "Response", "Sound"],
            "Code": ["cat", "probe_cat", "1", "elephant"],
            "Trial": [1, 2, 2, 3],
            "Stim_Type": ["other", "other", pd.NA, "other"],
        }
    )
    behavior_path = tmp_path / "sub-001_task-words_beh.tsv"
    behavior.to_csv(behavior_path, sep="\t", index=False)
    events = pd.DataFrame(
        {
            "onset": [0.1, 0.2, 0.3],
            "duration": [0.0, 0.0, 0.0],
            "trial_type": ["item", "yes_probe", "item_post_probe"],
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
    assert metadata["onset"].tolist() == [0.1, 0.3]
    assert metadata["word"].tolist() == ["cat", "elephant"]
    assert metadata["word_length_binary"].tolist() == ["short", "long"]


def test_drop_non_epochable_metadata_removes_out_of_bounds_events():
    info = mne.create_info(["MEG0111"], sfreq=100.0, ch_types=["mag"])
    raw = mne.io.RawArray(np.zeros((1, 100)), info, verbose="error")
    metadata = pd.DataFrame(
        {
            "onset": [0.5, 2.0],
            "duration": [0.0, 0.0],
            "condition": ["a", "a"],
        }
    )

    filtered = _drop_non_epochable_metadata(
        raw,
        metadata,
        label_column="condition",
        tmin=-0.1,
        tmax=0.2,
    )

    assert filtered["onset"].tolist() == [0.5]


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


def test_openneuro_subject_and_path_formatting():
    assert parse_subjects(DATASET_SPECS["ds004276"], "1-2") == (1, 2)
    assert parse_runs(DATASET_SPECS["ds004330"], "1,2,3") == ("01", "02", "03")
    files = run_files(DATASET_SPECS["ds004276"], Path("root"), 1, None)
    assert files.raw_path == Path("root/sub-001/meg/sub-001_task-words_meg.fif")
    assert files.behavior_path == Path("root/sub-001/beh/sub-001_task-words_beh.tsv")


def test_invalid_raw_fif_files_reports_unreadable_cache_entry(tmp_path: Path):
    files = run_files(DATASET_SPECS["ds006629"], tmp_path, 1, "0")
    files.raw_path.parent.mkdir(parents=True)
    files.raw_path.write_bytes(b"not a fif file")

    invalid = invalid_raw_fif_files("ds006629", bids_root=tmp_path, subjects="1", runs="0")

    assert [path for path, _reason in invalid] == [files.raw_path]

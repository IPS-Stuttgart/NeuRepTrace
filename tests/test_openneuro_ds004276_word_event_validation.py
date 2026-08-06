from pathlib import Path

import pandas as pd
import pytest

from neureptrace.openneuro_meg import DATASET_SPECS, RunFiles, _derive_metadata


def _run_files(tmp_path: Path) -> RunFiles:
    behavior_path = tmp_path / "sub-001_task-words_beh.tsv"
    pd.DataFrame(
        {
            "Event_Type": ["Sound", "Sound"],
            "Code": ["cat", "elephant"],
            "Trial": [1, 2],
            "Stim_Type": ["other", "other"],
        }
    ).to_csv(behavior_path, sep="\t", index=False)
    return RunFiles(
        subject="sub-001",
        run=None,
        raw_path=tmp_path / "sub-001_task-words_meg.fif",
        events_path=tmp_path / "sub-001_task-words_events.tsv",
        behavior_path=behavior_path,
    )


def test_ds004276_rejects_unrecognized_event_rows_even_when_counts_match(tmp_path: Path):
    events = pd.DataFrame(
        {
            "onset": [0.1, 0.2],
            "duration": [0.0, 0.0],
            "trial_type": ["probe", "fixation"],
        }
    )

    with pytest.raises(ValueError, match="no recognized auditory word rows"):
        _derive_metadata(DATASET_SPECS["ds004276"], _run_files(tmp_path), events)


def test_ds004276_requires_trial_type_for_behavior_alignment(tmp_path: Path):
    events = pd.DataFrame(
        {
            "onset": [0.1, 0.2],
            "duration": [0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="must contain a 'trial_type' column"):
        _derive_metadata(DATASET_SPECS["ds004276"], _run_files(tmp_path), events)

from __future__ import annotations

from neureptrace.dataset_manifest import manifest_from_dataset_config


def test_manifest_from_dataset_config_accepts_single_string_run_name():
    frame = manifest_from_dataset_config(
        {
            "dataset": {"root": "data"},
            "participants": {"include": [1, 2]},
            "files": {"main": "sub-{participant}_epo.fif", "cue": "sub-{participant}_cue-epo.fif"},
            "runs": [{"name": "main", "file": "main"}, {"name": "cue", "file": "cue"}],
        },
        run_names="cue",
    )

    assert frame["subject"].tolist() == ["Part1", "Part2"]
    assert frame["variant"].tolist() == ["cue", "cue"]
    assert frame["epochs"].tolist() == ["data/sub-1_cue-epo.fif", "data/sub-2_cue-epo.fif"]

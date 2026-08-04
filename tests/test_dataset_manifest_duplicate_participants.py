from __future__ import annotations

from neureptrace.dataset_manifest import manifest_from_dataset_config


def test_manifest_deduplicates_overlapping_participant_specs_in_first_seen_order():
    frame = manifest_from_dataset_config(
        {
            "dataset": {"root": "data"},
            "participants": {
                "include": ["3-1", "2-4", 3],
                "exclude": [4],
            },
            "files": {"main": "sub-{participant}_epo.fif"},
        }
    )

    assert frame["subject"].tolist() == ["Part3", "Part2", "Part1"]
    assert frame["epochs"].tolist() == [
        "data/sub-3_epo.fif",
        "data/sub-2_epo.fif",
        "data/sub-1_epo.fif",
    ]

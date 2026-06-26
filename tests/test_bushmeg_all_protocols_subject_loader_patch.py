from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import neureptrace.bushmeg_all_protocols as all_protocols
import neureptrace.bushmeg_source_loso as source_loso


def test_protocol3_subject_loader_accepts_source_loader_without_progress_callback(monkeypatch) -> None:
    all_protocols._PROTOCOL3_SUBJECT_CACHE.clear()
    try:
        subjects = {
            "s1": SimpleNamespace(labels=[0, 1]),
            "s2": SimpleNamespace(labels=[1, 0]),
            "s3": SimpleNamespace(labels=[0, 1]),
        }
        encoder = SimpleNamespace(classes_=[0, 1])
        calls = []

        def fake_load_subjects_from_config(config, *, config_dir):
            calls.append((config, config_dir))
            return subjects, encoder

        monkeypatch.setattr(source_loso, "_load_subjects_from_config", fake_load_subjects_from_config)
        config = {"dataset": {"root": "data"}, "participants": {"ids": "1,2,3"}}
        progress_events = []

        loaded_subjects, loaded_encoder = all_protocols._load_protocol3_subjects_cached(
            config,
            config_dir=Path.cwd(),
            progress_callback=lambda stage, **fields: progress_events.append((stage, fields)),
        )

        assert loaded_subjects is subjects
        assert loaded_encoder is encoder
        assert len(calls) == 1
        assert progress_events == []

        cached_subjects, cached_encoder = all_protocols._load_protocol3_subjects_cached(
            config,
            config_dir=Path.cwd(),
            progress_callback=lambda stage, **fields: progress_events.append((stage, fields)),
        )

        assert cached_subjects is subjects
        assert cached_encoder is encoder
        assert len(calls) == 1
        assert progress_events == [("loading_subjects", {"cache_hit": True, "n_subject_files": 3})]
    finally:
        all_protocols._PROTOCOL3_SUBJECT_CACHE.clear()

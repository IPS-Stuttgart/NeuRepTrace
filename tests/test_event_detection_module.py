from __future__ import annotations

import runpy
import sys
import types


def test_event_detection_module_exits_with_main_status(monkeypatch):
    fake_public = types.ModuleType("neureptrace._stimulus_detection_public")
    for constant_name in (
        "CONFLICT_RESOLUTION_MODES",
        "DEFAULT_GROUP_COLUMNS",
        "DEFAULT_STREAM_FALLBACKS",
        "DEFAULT_THRESHOLD_QUANTILE",
        "DEFAULT_THRESHOLD_WINDOW",
        "EVENT_COLUMNS",
        "SCORE_MODES",
        "THRESHOLD_METHODS",
    ):
        setattr(fake_public, constant_name, ())
    for function_name in (
        "detect_stimulus_events",
        "detect_stimulus_events_from_csvs",
        "fit_stimulus_detection_thresholds",
        "match_stimulus_annotations",
        "read_stimulus_probability_observations",
        "summarize_stimulus_events",
    ):
        setattr(fake_public, function_name, lambda *args, **kwargs: None)

    def fake_main():
        return 7

    fake_public.main = fake_main

    fake_streaming = types.ModuleType("neureptrace.streaming_stimulus_detection")
    fake_streaming.StimulusDetectionConfig = type("StimulusDetectionConfig", (), {})
    fake_streaming.StreamingStimulusDetector = type("StreamingStimulusDetector", (), {})

    monkeypatch.setitem(sys.modules, "neureptrace._stimulus_detection_public", fake_public)
    monkeypatch.setitem(sys.modules, "neureptrace.streaming_stimulus_detection", fake_streaming)

    try:
        runpy.run_module("neureptrace.event_detection", run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 7
    else:
        assert False, "event_detection did not exit with main status"

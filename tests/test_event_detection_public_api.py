from __future__ import annotations


def test_event_detection_facade_exports_offline_and_streaming_api() -> None:
    import neureptrace.event_detection as event_detection

    assert callable(event_detection.detect_stimulus_events)
    assert callable(event_detection.detect_stimulus_events_from_csvs)
    assert callable(event_detection.fit_stimulus_detection_thresholds)
    assert callable(event_detection.match_stimulus_annotations)
    assert callable(event_detection.summarize_stimulus_events)
    assert event_detection.StimulusDetectionConfig is not None
    assert event_detection.StreamingStimulusDetector is not None
    assert "detect_stimulus_events" in event_detection.__all__
    assert "StreamingStimulusDetector" in event_detection.__all__


def test_event_detection_cli_aliases_are_registered() -> None:
    from neureptrace.cli import COMMAND_MODULES

    assert COMMAND_MODULES["event-detect"] == "neureptrace.event_detection"
    assert COMMAND_MODULES["event-detection"] == "neureptrace.event_detection"

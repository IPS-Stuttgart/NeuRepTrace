"""Patch public detection CLI main so tests and callers can pass argv explicitly."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_PATCH_MARKER = "_neureptrace_cli_argv_patch_installed"


def _build_parser(public: Any) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect one or more stimulus events in long probability streams.")
    parser.add_argument("observation_csv", nargs="+", help="Observation CSVs or glob patterns with time and prob_class_* columns.")
    parser.add_argument("--annotations", "--annotations-csv", dest="annotations_csv", type=Path)
    parser.add_argument("--thresholds-csv", type=Path)
    parser.add_argument("--threshold-window", nargs=2, type=float, default=public.DEFAULT_THRESHOLD_WINDOW, metavar=("START", "STOP"))
    parser.add_argument("--threshold-quantile", type=float, default=public.DEFAULT_THRESHOLD_QUANTILE)
    parser.add_argument("--threshold-method", choices=public.THRESHOLD_METHODS, default="point")
    parser.add_argument("--score-mode", choices=public.SCORE_MODES, default="class_probability")
    parser.add_argument("--target-class", action="append", dest="target_classes")
    parser.add_argument("--group-column", action="append", dest="group_columns")
    parser.add_argument("--stream-column", action="append", dest="stream_columns")
    parser.add_argument("--detection-window", nargs=2, type=float, metavar=("START", "STOP"))
    parser.add_argument("--min-consecutive", type=int, default=1)
    parser.add_argument("--min-duration", type=float)
    parser.add_argument("--merge-gap", type=float)
    parser.add_argument("--refractory", type=float)
    parser.add_argument("--conflict-resolution", choices=public.CONFLICT_RESOLUTION_MODES, default="none")
    parser.add_argument("--match-tolerance", type=float, default=0.1)
    parser.add_argument("--out-events", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--out-thresholds", type=Path)
    return parser


def install() -> None:
    from neureptrace import (
        _onset_sensitivity_setting_id_patch,
        _onset_signed_probability_labels_patch,
        _onset_workflow_plot_optional_columns_patch,
    )

    _onset_sensitivity_setting_id_patch.install()
    _onset_signed_probability_labels_patch.install()
    _onset_workflow_plot_optional_columns_patch.install()
    public = importlib.import_module("neureptrace._stimulus_detection_public")
    if getattr(public.main, _PATCH_MARKER, False):
        return

    def main(argv: Sequence[str] | None = None) -> int:
        parser = _build_parser(public)
        args = parser.parse_args(argv)

        events, summary, _thresholds = public.detect_stimulus_events_from_csvs(
            args.observation_csv,
            annotations_csv=args.annotations_csv,
            thresholds_csv=args.thresholds_csv,
            threshold_window=tuple(args.threshold_window),
            threshold_quantile=args.threshold_quantile,
            threshold_method=args.threshold_method,
            score_mode=args.score_mode,
            target_classes=args.target_classes,
            group_columns=args.group_columns,
            stream_columns=args.stream_columns,
            detection_window=tuple(args.detection_window) if args.detection_window is not None else None,
            min_consecutive=args.min_consecutive,
            min_duration=args.min_duration,
            merge_gap=args.merge_gap,
            refractory=args.refractory,
            conflict_resolution=args.conflict_resolution,
            match_tolerance=args.match_tolerance,
            out_events=args.out_events,
            out_summary=args.out_summary,
            out_thresholds=args.out_thresholds,
        )
        print(f"Wrote stimulus events: {args.out_events}")
        print(f"Wrote stimulus event summary: {args.out_summary}")
        if args.out_thresholds is not None:
            print(f"Wrote stimulus thresholds: {args.out_thresholds}")
        print(summary.to_string(index=False))
        if not events.empty:
            print(events.head().to_string(index=False))
        return 0

    setattr(main, _PATCH_MARKER, True)
    public.main = main


__all__ = ["install"]

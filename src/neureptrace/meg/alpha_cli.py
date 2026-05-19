"""Command-line tools for reusable MEG alpha analyses."""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path

from neureptrace.meg.alpha_metrics import (
    DEFAULT_FREQUENCY_RANGE,
    DEFAULT_TIME_WINDOW,
    AlphaMetricConfig,
    export_participant_alpha_metrics,
)
from neureptrace.meg.alpha_movement import DEFAULT_MOVEMENT_TIME_WINDOW, DEFAULT_SENSOR_PATTERN, DEFAULT_TRAJECTORY_STEP_S, AlphaMovementConfig, export_alpha_movement
from neureptrace.meg.fieldtrip_struct import DEFAULT_MAT_ROOT_PATH, load_fieldtrip_mat, parse_path
from neureptrace.meg.sensor_geometry import DEFAULT_MIN_REFERENCE_AXIS_PROJECTION, DEFAULT_OCCIPITAL_PATTERN, DEFAULT_PROJECTION_REFERENCE_PATTERN


def parse_range(value: str) -> tuple[float, float]:
    """Parse comma-separated numeric range arguments."""

    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected two comma-separated values such as -0.4,-0.05.")
    try:
        start, stop = (float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Range values must be numeric.") from exc
    return start, stop


def parse_participant_spec(value: str) -> list[int]:
    """Parse participant specifications such as ``1-4,6,8``."""

    participants: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_text, stop_text = chunk.split("-", 1)
            start, stop = int(start_text), int(stop_text)
            if start > stop:
                raise argparse.ArgumentTypeError("Participant range start must not exceed stop.")
            participants.extend(range(start, stop + 1))
        else:
            participants.append(int(chunk))
    return sorted(set(participants))


def available_participants(data_dir, *, cue: bool = False, file_pattern: str | None = None) -> list[int]:
    """Return participant IDs discoverable from ``data_dir``."""

    data_dir = Path(data_dir)
    suffix = "CueData" if cue else "Data"
    if file_pattern is not None:
        # Prefer explicit participant selection for arbitrary patterns.
        return []
    pattern = re.compile(rf"Part(\d+){suffix}\.mat$")
    participants = []
    for path in data_dir.glob(f"Part*{suffix}.mat"):
        match = pattern.match(path.name)
        if match:
            participants.append(int(match.group(1)))
    return sorted(participants)


def _add_alpha_metric_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--location-pattern", default=DEFAULT_OCCIPITAL_PATTERN, help="Regex for selecting channels by label.")
    parser.add_argument("--time-window", type=parse_range, default=DEFAULT_TIME_WINDOW, help="Time window as start,stop in seconds.")
    parser.add_argument("--frequency-range", type=parse_range, default=DEFAULT_FREQUENCY_RANGE, help="Frequency range as low,high in Hz.")
    parser.add_argument("--filter-order", type=int, default=5, help="Butterworth band-pass filter order.")
    parser.add_argument("--sensor-position-unit", default="auto", help="Unit for data.grad.chanpos: auto, m, cm, or mm.")
    parser.add_argument("--projection-reference-pattern", default=DEFAULT_PROJECTION_REFERENCE_PATTERN, help="Regex for channels used to fit the common 2D projection frame.")
    parser.add_argument("--min-reference-axis-projection", type=float, default=DEFAULT_MIN_REFERENCE_AXIS_PROJECTION, help="Minimum robust in-plane projection of a global coordinate axis.")


def _alpha_metric_config_from_args(args) -> AlphaMetricConfig:
    return AlphaMetricConfig(
        location_pattern=args.location_pattern,
        time_window=args.time_window,
        frequency_range=args.frequency_range,
        filter_order=args.filter_order,
        sensor_position_unit=args.sensor_position_unit,
        projection_reference_pattern=args.projection_reference_pattern,
        min_reference_axis_projection=args.min_reference_axis_projection,
    )


def _participants(value: str | None, data_dir, cue: bool, file_pattern: str | None) -> list[int]:
    if value:
        return parse_participant_spec(value)
    return available_participants(data_dir, cue=cue, file_pattern=file_pattern)


def _add_fieldtrip_loading_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root-path", default=",".join(str(token) for token in DEFAULT_MAT_ROOT_PATH), help="Comma-separated path to the FieldTrip struct inside the .mat file, e.g. data,0.")
    parser.add_argument("--file-pattern", default=None, help="Participant file pattern with {participant}, {suffix}, and {cue} placeholders.")


def _build_alpha_metrics_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Export exploratory prestimulus MEG alpha metrics to CSV.")
    parser.add_argument("--mat", default=None, help="Single FieldTrip-like MAT file. Bypasses --data-dir/--participant.")
    parser.add_argument("--data-dir", default=None, help="Directory containing participant MAT files.")
    parser.add_argument("--participant", type=int, default=None, help="Participant id to export when using --data-dir.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--cue", action="store_true", help="Use Part*CueData.mat instead of Part*Data.mat.")
    _add_fieldtrip_loading_arguments(parser)
    _add_alpha_metric_arguments(parser)
    return parser


def alpha_metrics(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_alpha_metrics_parser(prog=prog)
    args = parser.parse_args(argv)
    config = _alpha_metric_config_from_args(args)
    root_path = parse_path(args.root_path)

    if args.mat:
        from neureptrace.meg.alpha_metrics import compute_alpha_metrics, write_alpha_metrics_csv

        data = load_fieldtrip_mat(args.mat, root_path=root_path)
        rows = compute_alpha_metrics(data, participant_id=args.participant, dataset="cue" if args.cue else "main", config=config)
        write_alpha_metrics_csv(rows, args.output)
    else:
        if args.data_dir is None or args.participant is None:
            parser.error("Pass either --mat or both --data-dir and --participant.")
        rows = export_participant_alpha_metrics(args.data_dir, args.participant, args.output, cue=args.cue, config=config, file_pattern=args.file_pattern, root_path=root_path)

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


def _build_alpha_movement_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description="Export sensor-level MEG alpha-power centroid trajectories.")
    parser.add_argument("--data-dir", required=True, help="Directory containing participant MAT files.")
    parser.add_argument("--participants", default=None, help="Participant ids such as 1-4,6,8. Defaults to all available MAT files.")
    parser.add_argument("--trajectory-output", required=True, help="Output CSV for trial/timepoint sensor-level trajectories.")
    parser.add_argument("--summary-output", default=None, help="Optional output CSV averaged by participant, condition, and time.")
    parser.add_argument("--cue", action="store_true", help="Use Part*CueData.mat instead of Part*Data.mat.")
    parser.add_argument("--location-pattern", default=DEFAULT_SENSOR_PATTERN, help="Regex for selecting channels by label. Defaults to all MEG channels.")
    parser.add_argument("--projection-reference-pattern", default=DEFAULT_PROJECTION_REFERENCE_PATTERN, help="Regex for channels used to fit the common 2D projection frame.")
    parser.add_argument("--min-reference-axis-projection", type=float, default=DEFAULT_MIN_REFERENCE_AXIS_PROJECTION, help="Minimum robust in-plane projection of a global coordinate axis.")
    parser.add_argument("--time-window", type=parse_range, default=DEFAULT_MOVEMENT_TIME_WINDOW, help="Time window as start,stop in seconds.")
    parser.add_argument("--frequency-range", type=parse_range, default=DEFAULT_FREQUENCY_RANGE, help="Frequency range as low,high in Hz.")
    parser.add_argument("--trajectory-step-s", type=float, default=DEFAULT_TRAJECTORY_STEP_S, help="Trajectory sampling step in seconds.")
    parser.add_argument("--filter-order", type=int, default=5, help="Butterworth band-pass filter order.")
    parser.add_argument("--sensor-position-unit", default="auto", help="Unit for data.grad.chanpos: auto, m, cm, or mm.")
    parser.add_argument("--min-total-alpha-power", type=float, default=0.0, help="Minimum total alpha power required for a reliable centroid.")
    _add_fieldtrip_loading_arguments(parser)
    return parser


def alpha_movement(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    parser = _build_alpha_movement_parser(prog=prog)
    args = parser.parse_args(argv)

    participants = _participants(args.participants, args.data_dir, args.cue, args.file_pattern)
    if not participants:
        parser.error("No participants found. Pass --participants or use default Part*Data.mat naming.")

    config = AlphaMovementConfig(
        location_pattern=args.location_pattern,
        time_window=args.time_window,
        frequency_range=args.frequency_range,
        trajectory_step_s=args.trajectory_step_s,
        filter_order=args.filter_order,
        sensor_position_unit=args.sensor_position_unit,
        projection_reference_pattern=args.projection_reference_pattern,
        min_reference_axis_projection=args.min_reference_axis_projection,
        min_total_alpha_power=args.min_total_alpha_power,
    )
    rows, summary_rows = export_alpha_movement(
        args.data_dir,
        participants,
        args.trajectory_output,
        summary_output_path=args.summary_output,
        cue=args.cue,
        config=config,
        file_pattern=args.file_pattern,
        root_path=parse_path(args.root_path),
    )
    print(f"Wrote {len(rows)} trajectory rows to {args.trajectory_output}")
    if args.summary_output:
        print(f"Wrote {len(summary_rows)} summary rows to {args.summary_output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MEG alpha-analysis commands.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    metrics_parser = subparsers.add_parser("metrics", help="Export per-trial alpha metrics.")
    movement_parser = subparsers.add_parser("movement", help="Export alpha-power centroid trajectories.")

    # Reuse the full parsers to keep help and parsing centralized.
    metrics_parser.set_defaults(handler=lambda rest: alpha_metrics(rest, prog="neureptrace-alpha metrics"))
    movement_parser.set_defaults(handler=lambda rest: alpha_movement(rest, prog="neureptrace-alpha movement"))
    args, remaining = parser.parse_known_args(argv)
    return args.handler(remaining)


if __name__ == "__main__":
    raise SystemExit(main())

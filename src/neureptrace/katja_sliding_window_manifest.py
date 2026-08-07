"""Build label-agnostic Katja online-decoding window manifests.

The manifest stores the raw intersection of every decoding window with every
press interval. It deliberately does not decide which overlaps become finger
labels, which windows are null, or how ties are resolved; those conventions are
left to the externally supplied reference window function.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_EXECUTION_START_SECONDS = 0.0
DEFAULT_EXECUTION_STOP_SECONDS = 6.0
DEFAULT_WINDOW_WIDTH_SECONDS = 0.5
DEFAULT_STRIDE_SECONDS = 0.04
DEFAULT_PRESS_BEFORE_SECONDS = 0.4
DEFAULT_PRESS_AFTER_SECONDS = 0.1


@dataclass(frozen=True, slots=True)
class WindowGrid:
    """A deterministic sliding-window grid."""

    starts_seconds: np.ndarray
    stops_seconds: np.ndarray
    centers_seconds: np.ndarray
    reference: str


def make_window_grid(
    *,
    execution_start_seconds: float = DEFAULT_EXECUTION_START_SECONDS,
    execution_stop_seconds: float = DEFAULT_EXECUTION_STOP_SECONDS,
    window_width_seconds: float = DEFAULT_WINDOW_WIDTH_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    reference: str = "start",
    require_full_windows: bool = True,
) -> WindowGrid:
    """Construct windows without accumulating floating-point stride error."""

    start = float(execution_start_seconds)
    stop = float(execution_stop_seconds)
    width = float(window_width_seconds)
    stride = float(stride_seconds)
    if not all(np.isfinite(value) for value in (start, stop, width, stride)):
        raise ValueError("Execution bounds, window width, and stride must be finite.")
    if stop <= start or width <= 0.0 or stride <= 0.0:
        raise ValueError("Execution stop, window width, and stride must define positive intervals.")
    mode = str(reference).strip().lower()
    if mode not in {"start", "center"}:
        raise ValueError("reference must be either 'start' or 'center'.")

    if mode == "start":
        origin = start
        limit = stop - width if require_full_windows else stop
        if limit < origin:
            raise ValueError("No decoding window fits inside the execution interval.")
        count = int(np.floor((limit - origin) / stride + 1e-10)) + 1
        starts = origin + np.arange(count, dtype=float) * stride
        stops = starts + width
        centers = starts + width / 2.0
    else:
        origin = start + width / 2.0 if require_full_windows else start
        limit = stop - width / 2.0 if require_full_windows else stop
        if limit < origin:
            raise ValueError("No decoding-window center fits inside the execution interval.")
        count = int(np.floor((limit - origin) / stride + 1e-10)) + 1
        centers = origin + np.arange(count, dtype=float) * stride
        starts = centers - width / 2.0
        stops = centers + width / 2.0

    return WindowGrid(
        starts_seconds=starts,
        stops_seconds=stops,
        centers_seconds=centers,
        reference=mode,
    )


def interval_intersections(
    window_starts: Sequence[float] | np.ndarray,
    window_stops: Sequence[float] | np.ndarray,
    interval_starts: Sequence[float] | np.ndarray,
    interval_stops: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Return window-by-interval intersection durations in seconds."""

    starts = np.asarray(window_starts, dtype=float).reshape(-1)
    stops = np.asarray(window_stops, dtype=float).reshape(-1)
    left = np.asarray(interval_starts, dtype=float).reshape(-1)
    right = np.asarray(interval_stops, dtype=float).reshape(-1)
    if starts.shape != stops.shape or left.shape != right.shape:
        raise ValueError("Window and interval start/stop vectors must have matching shapes.")
    if np.any(stops <= starts) or np.any(right <= left):
        raise ValueError("Every window and press interval must have positive duration.")
    return np.maximum(
        0.0,
        np.minimum(stops[:, None], right[None, :])
        - np.maximum(starts[:, None], left[None, :]),
    )


def _resolve_press_time_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in frame.columns:
            raise ValueError(f"Requested press-time column {requested!r} is absent.")
        return requested
    for candidate in (
        "recommended_time_seconds",
        "trigger_time_seconds",
        "expected_trigger_time_seconds",
        "behavior_time_seconds",
    ):
        if candidate in frame.columns:
            return candidate
    raise ValueError("Timing table contains no supported press-time column.")


def _repeat_scalar(value: Any, count: int) -> np.ndarray:
    if isinstance(value, np.generic):
        value = value.item()
    return np.repeat(np.asarray([value], dtype=object), count)


def build_sliding_window_manifest(
    press_timing: pd.DataFrame,
    *,
    press_time_column: str | None = None,
    execution_start_seconds: float = DEFAULT_EXECUTION_START_SECONDS,
    execution_stop_seconds: float = DEFAULT_EXECUTION_STOP_SECONDS,
    window_width_seconds: float = DEFAULT_WINDOW_WIDTH_SECONDS,
    stride_seconds: float = DEFAULT_STRIDE_SECONDS,
    press_before_seconds: float = DEFAULT_PRESS_BEFORE_SECONDS,
    press_after_seconds: float = DEFAULT_PRESS_AFTER_SECONDS,
    grid_reference: str = "start",
    require_full_windows: bool = True,
    require_complete_trials: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Expand per-press timing rows into raw sliding-window intersections."""

    required = {"subject", "trial_id", "press_position", "finger_code"}
    missing = sorted(required.difference(press_timing.columns))
    if missing:
        raise ValueError(f"Press timing table is missing required columns: {missing}.")
    before = float(press_before_seconds)
    after = float(press_after_seconds)
    if not np.isfinite(before) or not np.isfinite(after) or before < 0.0 or after <= 0.0:
        raise ValueError("press_before_seconds must be non-negative and press_after_seconds positive.")
    time_column = _resolve_press_time_column(press_timing, press_time_column)
    grid = make_window_grid(
        execution_start_seconds=execution_start_seconds,
        execution_stop_seconds=execution_stop_seconds,
        window_width_seconds=window_width_seconds,
        stride_seconds=stride_seconds,
        reference=grid_reference,
        require_full_windows=require_full_windows,
    )
    window_count = int(grid.starts_seconds.shape[0])
    frames: list[pd.DataFrame] = []
    skipped_trials: list[str] = []

    grouping = press_timing.groupby(["subject", "trial_id"], sort=True, dropna=False)
    for (subject, trial_id), group in grouping:
        ordered = group.sort_values("press_position")
        positions = ordered["press_position"].to_numpy(dtype=int)
        expected_positions = np.arange(1, 6, dtype=int)
        complete = np.array_equal(positions, expected_positions)
        if not complete:
            identifier = f"{subject}:{trial_id}"
            if require_complete_trials:
                raise ValueError(
                    f"Trial {identifier} must contain press positions 1--5 exactly; got {positions.tolist()}."
                )
            skipped_trials.append(identifier)
            continue
        press_times = ordered[time_column].to_numpy(dtype=float)
        if not np.all(np.isfinite(press_times)):
            identifier = f"{subject}:{trial_id}"
            if require_complete_trials:
                raise ValueError(f"Trial {identifier} contains non-finite press times.")
            skipped_trials.append(identifier)
            continue
        interval_starts = press_times - before
        interval_stops = press_times + after
        intersections = interval_intersections(
            grid.starts_seconds,
            grid.stops_seconds,
            interval_starts,
            interval_stops,
        )
        window_durations = grid.stops_seconds - grid.starts_seconds
        press_duration = before + after
        overlap_window = intersections / window_durations[:, None]
        overlap_press = intersections / press_duration
        maximum = np.max(intersections, axis=1)
        argmax = np.argmax(intersections, axis=1)
        positive = intersections > 0.0
        tie_count = np.sum(
            np.isclose(intersections, maximum[:, None], atol=1e-12, rtol=0.0)
            & positive,
            axis=1,
        )
        max_positions = np.where(maximum > 0.0, argmax + 1, 0)
        finger_codes = ordered["finger_code"].to_numpy()
        max_finger_codes = np.asarray(
            [
                finger_codes[index] if position > 0 else None
                for index, position in zip(argmax, max_positions, strict=True)
            ],
            dtype=object,
        )
        trial_frame = pd.DataFrame(
            {
                "subject": _repeat_scalar(subject, window_count),
                "trial_id": _repeat_scalar(trial_id, window_count),
                "window_id": np.arange(window_count, dtype=int),
                "window_start_seconds": grid.starts_seconds,
                "window_stop_seconds": grid.stops_seconds,
                "window_center_seconds": grid.centers_seconds,
                "window_duration_seconds": window_durations,
                "n_overlapping_presses": np.sum(positive, axis=1),
                "has_any_press_overlap": np.any(positive, axis=1),
                "max_overlap_seconds": maximum,
                "max_overlap_fraction_window": np.max(overlap_window, axis=1),
                "max_overlap_fraction_press": np.max(overlap_press, axis=1),
                "max_overlap_press_position": max_positions,
                "max_overlap_finger_code": max_finger_codes,
                "max_overlap_tie": tie_count > 1,
            }
        )
        for optional in ("sequence_id", "correct_order"):
            if optional in ordered.columns:
                values = ordered[optional].drop_duplicates()
                if values.shape[0] != 1:
                    raise ValueError(
                        f"Trial {subject}:{trial_id} has inconsistent {optional} values."
                    )
                trial_frame[optional] = _repeat_scalar(values.iloc[0], window_count)
        for index in range(5):
            position = index + 1
            prefix = f"press_{position}"
            trial_frame[f"{prefix}_time_seconds"] = press_times[index]
            trial_frame[f"{prefix}_finger_code"] = _repeat_scalar(
                finger_codes[index], window_count
            )
            trial_frame[f"{prefix}_interval_start_seconds"] = interval_starts[index]
            trial_frame[f"{prefix}_interval_stop_seconds"] = interval_stops[index]
            trial_frame[f"{prefix}_intersection_seconds"] = intersections[:, index]
            trial_frame[f"{prefix}_overlap_fraction_window"] = overlap_window[:, index]
            trial_frame[f"{prefix}_overlap_fraction_press"] = overlap_press[:, index]
        frames.append(trial_frame)

    if not frames:
        raise ValueError("No complete trials were available for the sliding-window manifest.")
    manifest = pd.concat(frames, ignore_index=True)
    metadata = {
        "format": "neureptrace_katja_sliding_window_intersections_v1",
        "press_time_column": time_column,
        "execution_start_seconds": float(execution_start_seconds),
        "execution_stop_seconds": float(execution_stop_seconds),
        "window_width_seconds": float(window_width_seconds),
        "stride_seconds": float(stride_seconds),
        "grid_reference": grid.reference,
        "require_full_windows": bool(require_full_windows),
        "press_interval_relative_seconds": [
            -float(press_before_seconds),
            float(press_after_seconds),
        ],
        "n_windows_per_trial": window_count,
        "n_trials": int(manifest[["subject", "trial_id"]].drop_duplicates().shape[0]),
        "n_rows": int(manifest.shape[0]),
        "skipped_trials": skipped_trials,
        "label_assignment_status": "pending_external_reference_window_function",
        "defines_null_labels": False,
        "defines_finger_labels": False,
        "includes_first_press": True,
    }
    return manifest, metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--press-timing", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output")
    parser.add_argument("--press-time-column")
    parser.add_argument("--execution-start-seconds", type=float, default=0.0)
    parser.add_argument("--execution-stop-seconds", type=float, default=6.0)
    parser.add_argument("--window-width-ms", type=float, default=500.0)
    parser.add_argument("--stride-ms", type=float, default=40.0)
    parser.add_argument("--press-before-ms", type=float, default=400.0)
    parser.add_argument("--press-after-ms", type=float, default=100.0)
    parser.add_argument("--grid-reference", choices=("start", "center"), default="start")
    parser.add_argument("--allow-partial-windows", action="store_true")
    parser.add_argument("--allow-incomplete-trials", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    timing = pd.read_csv(args.press_timing)
    manifest, metadata = build_sliding_window_manifest(
        timing,
        press_time_column=args.press_time_column,
        execution_start_seconds=args.execution_start_seconds,
        execution_stop_seconds=args.execution_stop_seconds,
        window_width_seconds=float(args.window_width_ms) / 1000.0,
        stride_seconds=float(args.stride_ms) / 1000.0,
        press_before_seconds=float(args.press_before_ms) / 1000.0,
        press_after_seconds=float(args.press_after_ms) / 1000.0,
        grid_reference=args.grid_reference,
        require_full_windows=not args.allow_partial_windows,
        require_complete_trials=not args.allow_incomplete_trials,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    compression = "gzip" if output.suffix == ".gz" else None
    manifest.to_csv(output, index=False, compression=compression)
    metadata_path = (
        Path(args.metadata_output)
        if args.metadata_output
        else output.with_suffix(".metadata.json")
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

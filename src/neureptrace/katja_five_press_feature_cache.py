"""Build exact five-press Katja Button Press MEG feature caches.

The established four-class benchmark scores presses 2--5. This auxiliary cache
also retains press 1 so a globally consistent five-finger sequence model can use
the complete trial while still reporting accuracy only on the original scored
presses. Raw arrays remain memory mapped one subject at a time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from neureptrace._katja_finger_sequence_support import DEFAULT_PARTICIPANTS
from neureptrace.katja_spm_feature_cache import (
    DEFAULT_BASELINE_SECONDS,
    DEFAULT_BUTTON_LAG_SECONDS,
    DEFAULT_WINDOW_CENTERS_SECONDS,
    DEFAULT_WINDOW_WIDTH_SECONDS,
    SPMContinuousHeader,
    SubjectFeatureRows,
    _baseline_scale,
    _channel_indices,
    _sample_bounds,
    common_good_meggrad_labels,
    load_spm_continuous_header,
    load_subject_behavior,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

FIVE_PRESS_CACHE_FORMAT = "neureptrace_katja_button_press_five_press_exact_ica_v1"


def extract_five_press_subject_rows(
    *,
    subject: str,
    header: SPMContinuousHeader,
    behavior: dict[str, np.ndarray],
    selected_channel_labels: Sequence[str],
    window_centers_seconds: Sequence[float] = DEFAULT_WINDOW_CENTERS_SECONDS,
    window_width_seconds: float = DEFAULT_WINDOW_WIDTH_SECONDS,
    baseline_seconds: tuple[float, float] = DEFAULT_BASELINE_SECONDS,
    button_lag_seconds: float = DEFAULT_BUTTON_LAG_SECONDS,
) -> SubjectFeatureRows:
    """Extract response-locked features for all five presses in retained trials."""

    centers = tuple(float(value) for value in window_centers_seconds)
    if not centers or not np.all(np.isfinite(centers)):
        raise ValueError("window_centers_seconds must be a non-empty finite sequence.")
    width = float(window_width_seconds)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("window_width_seconds must be positive and finite.")
    if not (
        np.isfinite(baseline_seconds[0])
        and np.isfinite(baseline_seconds[1])
        and baseline_seconds[0] < baseline_seconds[1]
    ):
        raise ValueError("baseline_seconds must be a finite increasing interval.")
    lag = float(button_lag_seconds)
    if not np.isfinite(lag):
        raise ValueError("button_lag_seconds must be finite.")

    n_trials = min(
        int(header.shape[2]),
        *(int(np.asarray(value).shape[0]) for value in behavior.values()),
    )
    channel_indices = _channel_indices(header, selected_channel_labels)
    data = np.memmap(
        header.data_path,
        dtype=header.dtype,
        mode="r",
        offset=header.offset,
        shape=header.shape,
        order="F",
    )
    baseline_bounds = _sample_bounds(
        start_seconds=baseline_seconds[0],
        stop_seconds=baseline_seconds[1],
        time_onset=header.time_onset,
        sampling_frequency=header.sampling_frequency,
        n_samples=header.shape[1],
    )
    if baseline_bounds is None:
        raise ValueError(
            f"Baseline interval {baseline_seconds!r} is outside {header.header_path}."
        )

    press = np.asarray(behavior["press"])
    instructed = np.asarray(behavior["cue_finger"])
    timing_ms = np.asarray(behavior["timing_ms"], dtype=float)
    cue_duration_ms = np.asarray(behavior["cue_duration_ms"], dtype=float)
    sequence_ids = np.asarray(behavior["sequence_ids"])

    feature_rows: list[np.ndarray] = []
    trial_ids: list[int] = []
    press_positions: list[int] = []
    sequence_rows: list[Any] = []
    finger_codes: list[Any] = []
    event_times: list[float] = []
    correct_order_count = 0
    retained_trials = 0
    dropped_out_of_bounds = 0
    dropped_nonfinite = 0
    baseline_start, baseline_stop = baseline_bounds
    half_width = width / 2.0

    for trial_index in range(n_trials):
        recorded = press[trial_index, :5]
        planned = instructed[trial_index, :5]
        variable_timing_finite = bool(
            np.all(np.isfinite(timing_ms[trial_index, 1:5]))
        )
        correct_order = bool(
            variable_timing_finite and np.array_equal(recorded, planned)
        )
        if not correct_order:
            continue
        correct_order_count += 1
        if np.unique(recorded).shape[0] != 5:
            raise ValueError(
                f"Participant {subject}, trial {trial_index + 1} does not contain "
                "five unique physical finger codes."
            )
        if not np.all(np.isfinite(timing_ms[trial_index, :5])):
            dropped_nonfinite += 1
            continue
        if not np.isfinite(cue_duration_ms[trial_index]):
            dropped_nonfinite += 1
            continue

        baseline = np.asarray(
            data[channel_indices, baseline_start:baseline_stop, trial_index],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(baseline)):
            dropped_nonfinite += 1
            continue
        try:
            baseline_mean, baseline_scale = _baseline_scale(baseline)
        except ValueError:
            dropped_nonfinite += 1
            continue

        trial_features: list[np.ndarray] = []
        trial_times: list[float] = []
        trial_valid = True
        trial_nonfinite = False
        for press_index in range(5):
            event_seconds = (
                cue_duration_ms[trial_index]
                + timing_ms[trial_index, press_index]
            ) / 1000.0 + lag
            vectors: list[np.ndarray] = []
            for center in centers:
                bounds = _sample_bounds(
                    start_seconds=event_seconds + center - half_width,
                    stop_seconds=event_seconds + center + half_width,
                    time_onset=header.time_onset,
                    sampling_frequency=header.sampling_frequency,
                    n_samples=header.shape[1],
                )
                if bounds is None:
                    trial_valid = False
                    break
                start, stop = bounds
                window = np.asarray(
                    data[channel_indices, start:stop, trial_index],
                    dtype=np.float64,
                )
                window_mean = np.nanmean(window, axis=1)
                normalized = (window_mean - baseline_mean) / baseline_scale
                if not np.all(np.isfinite(normalized)):
                    trial_valid = False
                    trial_nonfinite = True
                    break
                vectors.append(normalized.astype(np.float32, copy=False))
            if not trial_valid:
                break
            trial_features.append(
                np.concatenate(vectors).astype(np.float32, copy=False)
            )
            trial_times.append(float(event_seconds))

        if not trial_valid:
            if trial_nonfinite:
                dropped_nonfinite += 1
            else:
                dropped_out_of_bounds += 1
            continue

        retained_trials += 1
        for press_index, feature_vector in enumerate(trial_features):
            feature_rows.append(feature_vector)
            trial_ids.append(trial_index + 1)
            press_positions.append(press_index + 1)
            sequence_rows.append(sequence_ids[trial_index])
            finger_codes.append(recorded[press_index])
            event_times.append(trial_times[press_index])

    if not feature_rows:
        raise ValueError(f"No valid five-press rows were extracted for {subject}.")
    features = np.stack(feature_rows).astype(np.float32, copy=False)
    return SubjectFeatureRows(
        features=features,
        subjects=np.asarray([subject] * features.shape[0]),
        trial_ids=np.asarray(trial_ids, dtype=int),
        press_positions=np.asarray(press_positions, dtype=int),
        sequence_ids=np.asarray(sequence_rows),
        finger_codes=np.asarray(finger_codes),
        event_times_seconds=np.asarray(event_times, dtype=np.float64),
        n_trials_considered=n_trials,
        n_trials_correct_order=correct_order_count,
        n_trials_retained=retained_trials,
        n_trials_dropped_out_of_bounds=dropped_out_of_bounds,
        n_trials_dropped_nonfinite=dropped_nonfinite,
    )


def _concatenate(rows: Sequence[SubjectFeatureRows], field: str) -> np.ndarray:
    return np.concatenate(
        [np.asarray(getattr(row, field)) for row in rows],
        axis=0,
    )


def build_katja_five_press_feature_cache(
    dataset_root: str | Path,
    output_path: str | Path,
    *,
    participants: Sequence[str] = DEFAULT_PARTICIPANTS,
    window_centers_seconds: Sequence[float] = DEFAULT_WINDOW_CENTERS_SECONDS,
    window_width_seconds: float = DEFAULT_WINDOW_WIDTH_SECONDS,
    baseline_seconds: tuple[float, float] = DEFAULT_BASELINE_SECONDS,
    button_lag_seconds: float = DEFAULT_BUTTON_LAG_SECONDS,
) -> dict[str, Any]:
    """Build one NPZ containing all five response events per retained trial."""

    root = Path(dataset_root)
    participant_tuple = tuple(str(value) for value in participants)
    if not participant_tuple or len(set(participant_tuple)) != len(participant_tuple):
        raise ValueError("participants must contain unique identifiers.")
    headers = [
        load_spm_continuous_header(root / participant)
        for participant in participant_tuple
    ]
    common_labels = common_good_meggrad_labels(headers)
    subject_rows: list[SubjectFeatureRows] = []
    subject_summary: dict[str, dict[str, Any]] = {}
    for participant, header in zip(participant_tuple, headers, strict=True):
        behavior = load_subject_behavior(root / "beh_data" / participant)
        rows = extract_five_press_subject_rows(
            subject=participant,
            header=header,
            behavior=behavior,
            selected_channel_labels=common_labels,
            window_centers_seconds=window_centers_seconds,
            window_width_seconds=window_width_seconds,
            baseline_seconds=baseline_seconds,
            button_lag_seconds=button_lag_seconds,
        )
        subject_rows.append(rows)
        subject_summary[participant] = {
            "spm_header": str(header.header_path),
            "spm_data": str(header.data_path),
            "n_trials_considered": rows.n_trials_considered,
            "n_trials_correct_order": rows.n_trials_correct_order,
            "n_trials_retained": rows.n_trials_retained,
            "n_trials_dropped_out_of_bounds": rows.n_trials_dropped_out_of_bounds,
            "n_trials_dropped_nonfinite": rows.n_trials_dropped_nonfinite,
            "n_event_rows": int(rows.features.shape[0]),
        }

    features = _concatenate(subject_rows, "features").astype(np.float32, copy=False)
    metadata = {
        "format": FIVE_PRESS_CACHE_FORMAT,
        "participants": list(participant_tuple),
        "selected_channel_policy": "intersection_of_good_MEGGRAD_across_participants",
        "selected_channel_count": len(common_labels),
        "selected_channel_labels": list(common_labels),
        "window_centers_seconds": [
            float(value) for value in window_centers_seconds
        ],
        "window_width_seconds": float(window_width_seconds),
        "baseline_seconds": [float(value) for value in baseline_seconds],
        "button_lag_seconds": float(button_lag_seconds),
        "event_positions": [1, 2, 3, 4, 5],
        "scored_event_positions": [2, 3, 4, 5],
        "correct_order_rule": (
            "finite variable timing and recorded five-press sequence equals cueFinger"
        ),
        "subject_summary": subject_summary,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        features=features,
        subjects=_concatenate(subject_rows, "subjects"),
        trial_ids=_concatenate(subject_rows, "trial_ids"),
        press_positions=_concatenate(subject_rows, "press_positions"),
        sequence_ids=_concatenate(subject_rows, "sequence_ids"),
        finger_codes=_concatenate(subject_rows, "finger_codes"),
        event_times_seconds=_concatenate(subject_rows, "event_times_seconds"),
        correct_order=np.ones(features.shape[0], dtype=bool),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--participants", default=",".join(DEFAULT_PARTICIPANTS))
    parser.add_argument("--button-lag-ms", type=float, default=30.0)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.force:
        raise FileExistsError(output)
    participants = tuple(
        token.strip() for token in args.participants.split(",") if token.strip()
    )
    metadata = build_katja_five_press_feature_cache(
        args.dataset_root,
        output,
        participants=participants,
        button_lag_seconds=float(args.button_lag_ms) / 1000.0,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = (
    "FIVE_PRESS_CACHE_FORMAT",
    "build_katja_five_press_feature_cache",
    "extract_five_press_subject_rows",
)

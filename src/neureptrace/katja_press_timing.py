"""Audit Katja button-press timing against the MEG ``UPPT002`` channel.

The behavioral files define press times relative to the fractal-cue onset as
``cue_duration + timing``. The MEG trigger channel records the same presses on
the acquisition clock, typically with a small positive lag. This module keeps
both representations, matches trigger edges to the five behavioral presses, and
writes a per-press table that downstream online-window code can consume without
re-reading the multi-gigabyte sensor files.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from neureptrace._katja_finger_sequence_support import DEFAULT_PARTICIPANTS
from neureptrace.katja_spm_feature_cache import (
    SPMContinuousHeader,
    load_spm_continuous_header,
    load_subject_behavior,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_TRIGGER_CHANNEL = "UPPT002"
DEFAULT_EXPECTED_LAG_SECONDS = 0.03
DEFAULT_MATCH_TOLERANCE_SECONDS = 0.15
DEFAULT_MIN_SEPARATION_SECONDS = 0.05


@dataclass(frozen=True, slots=True)
class TriggerOnsets:
    """Detected non-baseline trigger pulses for one trial."""

    sample_indices: np.ndarray
    times_seconds: np.ndarray
    values: np.ndarray
    baseline: float
    threshold: float


@dataclass(frozen=True, slots=True)
class TriggerMatches:
    """Monotone one-to-one trigger matches for behavioral press times."""

    trigger_indices: np.ndarray
    trigger_times_seconds: np.ndarray
    trigger_minus_behavior_seconds: np.ndarray
    residual_from_expected_lag_seconds: np.ndarray


def _finite_float_array(values: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional; got shape {array.shape}.")
    return array


def behavioral_press_times_seconds(behavior: dict[str, np.ndarray]) -> np.ndarray:
    """Return trial-by-five behavioral press times on the MEG epoch clock."""

    timing_ms = _finite_float_array(behavior["timing_ms"], name="timing_ms", ndim=2)
    cue_duration_ms = _finite_float_array(
        behavior["cue_duration_ms"], name="cue_duration_ms", ndim=1
    )
    if timing_ms.shape[0] != cue_duration_ms.shape[0] or timing_ms.shape[1] != 5:
        raise ValueError(
            "Katja timing data must have one cue duration and five press times per trial."
        )
    return (timing_ms + cue_duration_ms[:, None]) / 1000.0


def _trigger_channel_index(
    header: SPMContinuousHeader, channel_label: str = DEFAULT_TRIGGER_CHANNEL
) -> int:
    wanted = str(channel_label).strip().upper()
    matches = [
        index
        for index, label in enumerate(header.channel_labels)
        if str(label).strip().upper() == wanted
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {channel_label!r} channel in {header.header_path}; "
            f"found {len(matches)}."
        )
    return int(matches[0])


def _auto_trigger_threshold(signal: np.ndarray, baseline: float) -> float:
    deviations = np.abs(signal - baseline)
    finite = deviations[np.isfinite(deviations)]
    if finite.size == 0:
        raise ValueError("Trigger signal contains no finite samples.")
    maximum = float(np.max(finite))
    if maximum <= 0.0:
        return np.inf
    median_deviation = float(np.median(finite))
    robust_noise = 1.4826 * median_deviation
    numeric_floor = np.finfo(float).eps * max(1.0, abs(baseline), maximum) * 16.0
    # UPPT channels are normally quantized. One percent of the full excursion
    # rejects floating-point noise while retaining low-amplitude digital pulses.
    return max(numeric_floor, robust_noise * 8.0, maximum * 0.01)


def detect_trigger_onsets(
    signal: Sequence[float] | np.ndarray,
    *,
    sampling_frequency: float,
    time_onset: float,
    min_separation_seconds: float = DEFAULT_MIN_SEPARATION_SECONDS,
    threshold: float | None = None,
) -> TriggerOnsets:
    """Detect pulse onsets as transitions away from the dominant baseline.

    The method intentionally does not interpret trigger values as finger labels;
    it uses only pulse timing. Closely spaced transitions are collapsed to the
    first onset to suppress switch bounce and filtered pulse shoulders.
    """

    values = np.asarray(signal, dtype=float).reshape(-1)
    if values.size < 2 or not np.any(np.isfinite(values)):
        raise ValueError("Trigger signal must contain at least two finite samples.")
    sfreq = float(sampling_frequency)
    onset = float(time_onset)
    separation = float(min_separation_seconds)
    if not np.isfinite(sfreq) or sfreq <= 0.0:
        raise ValueError("sampling_frequency must be positive and finite.")
    if not np.isfinite(onset):
        raise ValueError("time_onset must be finite.")
    if not np.isfinite(separation) or separation < 0.0:
        raise ValueError("min_separation_seconds must be non-negative and finite.")

    finite_values = values[np.isfinite(values)]
    rounded = np.round(finite_values, decimals=9)
    unique, counts = np.unique(rounded, return_counts=True)
    baseline = float(unique[int(np.argmax(counts))])
    used_threshold = (
        _auto_trigger_threshold(values, baseline)
        if threshold is None
        else float(threshold)
    )
    if np.isnan(used_threshold) or used_threshold < 0.0:
        raise ValueError("threshold must be non-negative and not NaN.")
    if not np.isfinite(used_threshold):
        return TriggerOnsets(
            sample_indices=np.empty(0, dtype=int),
            times_seconds=np.empty(0, dtype=float),
            values=np.empty(0, dtype=float),
            baseline=baseline,
            threshold=used_threshold,
        )

    active = np.isfinite(values) & (np.abs(values - baseline) > used_threshold)
    previous = np.concatenate(([False], active[:-1]))
    candidates = np.flatnonzero(active & ~previous)
    minimum_samples = max(1, int(round(separation * sfreq)))
    kept: list[int] = []
    for index in candidates.tolist():
        if not kept or index - kept[-1] >= minimum_samples:
            kept.append(int(index))
    indices = np.asarray(kept, dtype=int)
    return TriggerOnsets(
        sample_indices=indices,
        times_seconds=onset + indices.astype(float) / sfreq,
        values=values[indices] if indices.size else np.empty(0, dtype=float),
        baseline=baseline,
        threshold=used_threshold,
    )


def match_trigger_onsets(
    behavioral_times_seconds: Sequence[float] | np.ndarray,
    trigger_times_seconds: Sequence[float] | np.ndarray,
    *,
    expected_lag_seconds: float = DEFAULT_EXPECTED_LAG_SECONDS,
    tolerance_seconds: float = DEFAULT_MATCH_TOLERANCE_SECONDS,
) -> TriggerMatches:
    """Match triggers to presses with monotone dynamic programming.

    Extra trigger pulses are skipped without penalty. A behavioral press is
    left unmatched when every available trigger is farther than ``tolerance``
    from ``behavioral_time + expected_lag``.
    """

    behavioral = np.asarray(behavioral_times_seconds, dtype=float).reshape(-1)
    detected = np.sort(np.asarray(trigger_times_seconds, dtype=float).reshape(-1))
    lag = float(expected_lag_seconds)
    tolerance = float(tolerance_seconds)
    if not np.isfinite(lag):
        raise ValueError("expected_lag_seconds must be finite.")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance_seconds must be positive and finite.")

    result_indices = np.full(behavioral.shape, -1, dtype=int)
    result_times = np.full(behavioral.shape, np.nan, dtype=float)
    finite_expected_positions = np.flatnonzero(np.isfinite(behavioral))
    finite_detected = detected[np.isfinite(detected)]
    if finite_expected_positions.size == 0 or finite_detected.size == 0:
        return TriggerMatches(
            trigger_indices=result_indices,
            trigger_times_seconds=result_times,
            trigger_minus_behavior_seconds=np.full(behavioral.shape, np.nan),
            residual_from_expected_lag_seconds=np.full(behavioral.shape, np.nan),
        )

    expected = behavioral[finite_expected_positions]
    m = expected.shape[0]
    n = finite_detected.shape[0]
    missing_penalty = 1.0
    dp = np.full((m + 1, n + 1), np.inf, dtype=float)
    choice = np.full((m + 1, n + 1), -1, dtype=np.int8)
    dp[0, :] = 0.0
    choice[0, 1:] = 0
    for i in range(1, m + 1):
        dp[i, 0] = i * missing_penalty
        choice[i, 0] = 1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            residual = abs((finite_detected[j - 1] - expected[i - 1]) - lag)
            match_cost = residual / tolerance
            options = (
                (dp[i - 1, j - 1] + match_cost, 2),
                (dp[i - 1, j] + missing_penalty, 1),
                (dp[i, j - 1], 0),
            )
            best_cost, best_choice = min(options, key=lambda item: (item[0], -item[1]))
            dp[i, j] = best_cost
            choice[i, j] = best_choice

    i, j = m, n
    matched_pairs: list[tuple[int, int]] = []
    while i > 0 or j > 0:
        action = int(choice[i, j])
        if action == 2:
            residual = abs((finite_detected[j - 1] - expected[i - 1]) - lag)
            if residual <= tolerance:
                matched_pairs.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif action == 1:
            i -= 1
        elif action == 0:
            j -= 1
        else:
            break

    for expected_index, detected_index in reversed(matched_pairs):
        original_position = int(finite_expected_positions[expected_index])
        result_indices[original_position] = int(detected_index)
        result_times[original_position] = float(finite_detected[detected_index])
    differences = result_times - behavioral
    return TriggerMatches(
        trigger_indices=result_indices,
        trigger_times_seconds=result_times,
        trigger_minus_behavior_seconds=differences,
        residual_from_expected_lag_seconds=differences - lag,
    )


def _trial_value(values: np.ndarray, trial_index: int, press_index: int | None = None) -> Any:
    value = values[trial_index] if press_index is None else values[trial_index, press_index]
    if isinstance(value, np.generic):
        return value.item()
    return value


def audit_subject_press_timing(
    dataset_root: str | Path,
    subject: str,
    *,
    channel_label: str = DEFAULT_TRIGGER_CHANNEL,
    expected_lag_seconds: float = DEFAULT_EXPECTED_LAG_SECONDS,
    match_tolerance_seconds: float = DEFAULT_MATCH_TOLERANCE_SECONDS,
    min_separation_seconds: float = DEFAULT_MIN_SEPARATION_SECONDS,
    trigger_threshold: float | None = None,
) -> pd.DataFrame:
    """Return one timing-audit row per trial and press for one participant."""

    root = Path(dataset_root)
    header = load_spm_continuous_header(root / subject)
    behavior = load_subject_behavior(root / "beh_data" / subject)
    behavioral_times = behavioral_press_times_seconds(behavior)
    n_trials = min(header.shape[2], behavioral_times.shape[0])
    channel_index = _trigger_channel_index(header, channel_label)
    data = np.memmap(
        header.data_path,
        dtype=header.dtype,
        mode="r",
        offset=header.offset,
        shape=header.shape,
        order="F",
    )

    press = np.asarray(behavior["press"])
    instructed = np.asarray(behavior["cue_finger"])
    sequence_ids = np.asarray(behavior["sequence_ids"])
    rows: list[dict[str, Any]] = []
    for trial_index in range(n_trials):
        trigger_signal = np.asarray(data[channel_index, :, trial_index], dtype=float)
        onsets = detect_trigger_onsets(
            trigger_signal,
            sampling_frequency=header.sampling_frequency,
            time_onset=header.time_onset,
            min_separation_seconds=min_separation_seconds,
            threshold=trigger_threshold,
        )
        matches = match_trigger_onsets(
            behavioral_times[trial_index],
            onsets.times_seconds,
            expected_lag_seconds=expected_lag_seconds,
            tolerance_seconds=match_tolerance_seconds,
        )
        recorded = press[trial_index, :5]
        cue = instructed[trial_index, :5]
        correct_order = bool(
            np.all(np.isfinite(np.asarray(recorded, dtype=float)))
            and np.array_equal(recorded, cue)
        )
        for press_index in range(5):
            behavior_time = float(behavioral_times[trial_index, press_index])
            trigger_time = float(matches.trigger_times_seconds[press_index])
            difference = float(matches.trigger_minus_behavior_seconds[press_index])
            rows.append(
                {
                    "subject": str(subject),
                    "trial_id": trial_index + 1,
                    "sequence_id": _trial_value(sequence_ids, trial_index),
                    "press_position": press_index + 1,
                    "finger_code": _trial_value(press, trial_index, press_index),
                    "instructed_finger_code": _trial_value(instructed, trial_index, press_index),
                    "correct_order": correct_order,
                    "behavior_time_seconds": behavior_time,
                    "expected_trigger_time_seconds": behavior_time + float(expected_lag_seconds),
                    "trigger_time_seconds": trigger_time,
                    "trigger_matched": bool(np.isfinite(trigger_time)),
                    "trigger_minus_behavior_ms": difference * 1000.0,
                    "residual_from_expected_lag_ms": float(matches.residual_from_expected_lag_seconds[press_index]) * 1000.0,
                    "matched_trigger_index": int(matches.trigger_indices[press_index]),
                    "n_detected_trigger_onsets_trial": int(onsets.times_seconds.shape[0]),
                    "trigger_channel": str(channel_label),
                    "trigger_baseline": float(onsets.baseline),
                    "trigger_threshold": float(onsets.threshold),
                    "sampling_frequency_hz": float(header.sampling_frequency),
                    "spm_time_onset_seconds": float(header.time_onset),
                    "spm_header": str(header.header_path),
                    "spm_data": str(header.data_path),
                }
            )
    return pd.DataFrame(rows)


def finalize_press_timing_rows(
    rows: pd.DataFrame,
    *,
    default_lag_seconds: float = DEFAULT_EXPECTED_LAG_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add recommended timestamps and return participant-level audit summaries."""

    required = {
        "subject",
        "trial_id",
        "press_position",
        "behavior_time_seconds",
        "trigger_time_seconds",
        "trigger_matched",
        "trigger_minus_behavior_ms",
        "correct_order",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise ValueError(f"Timing rows are missing required columns: {missing}.")
    frame = rows.copy()
    matched = frame["trigger_matched"].astype(bool) & np.isfinite(frame["trigger_minus_behavior_ms"].to_numpy(dtype=float))
    lag_by_subject = frame.loc[matched].groupby("subject", sort=True)["trigger_minus_behavior_ms"].median().to_dict()
    default_lag_ms = float(default_lag_seconds) * 1000.0
    frame["subject_median_trigger_lag_ms"] = frame["subject"].map(lag_by_subject)
    frame["fallback_lag_ms"] = frame["subject_median_trigger_lag_ms"].fillna(default_lag_ms)
    behavior = frame["behavior_time_seconds"].to_numpy(dtype=float)
    trigger = frame["trigger_time_seconds"].to_numpy(dtype=float)
    fallback = behavior + frame["fallback_lag_ms"].to_numpy(dtype=float) / 1000.0
    use_trigger = np.isfinite(trigger)
    frame["recommended_time_seconds"] = np.where(use_trigger, trigger, fallback)
    frame["recommended_time_source"] = np.where(
        use_trigger,
        "UPPT002",
        np.where(
            frame["subject_median_trigger_lag_ms"].notna(),
            "behavior_plus_subject_median_trigger_lag",
            "behavior_plus_configured_lag",
        ),
    )

    summary_rows: list[dict[str, Any]] = []
    for subject, group in frame.groupby("subject", sort=True):
        group_matched = group[group["trigger_matched"].astype(bool) & np.isfinite(group["trigger_minus_behavior_ms"].to_numpy(dtype=float))]
        lag_values = group_matched["trigger_minus_behavior_ms"].to_numpy(dtype=float)
        complete_by_trial = group.groupby("trial_id")["trigger_matched"].all()
        median = float(np.median(lag_values)) if lag_values.size else np.nan
        summary_rows.append(
            {
                "subject": subject,
                "n_trials": int(group["trial_id"].nunique()),
                "n_press_rows": int(group.shape[0]),
                "n_trigger_matched": int(group_matched.shape[0]),
                "trigger_match_fraction": float(group_matched.shape[0] / group.shape[0]),
                "n_complete_five_trigger_trials": int(complete_by_trial.sum()),
                "n_correct_order_trials": int(group.loc[group["correct_order"].astype(bool), "trial_id"].nunique()),
                "median_trigger_lag_ms": median,
                "median_absolute_deviation_ms": float(np.median(np.abs(lag_values - median))) if lag_values.size else np.nan,
                "mean_trigger_lag_ms": float(np.mean(lag_values)) if lag_values.size else np.nan,
                "std_trigger_lag_ms": float(np.std(lag_values, ddof=1)) if lag_values.size > 1 else np.nan,
                "p05_trigger_lag_ms": float(np.quantile(lag_values, 0.05)) if lag_values.size else np.nan,
                "p95_trigger_lag_ms": float(np.quantile(lag_values, 0.95)) if lag_values.size else np.nan,
            }
        )
    return frame, pd.DataFrame(summary_rows)


def audit_dataset_press_timing(
    dataset_root: str | Path,
    *,
    participants: Sequence[str] = DEFAULT_PARTICIPANTS,
    channel_label: str = DEFAULT_TRIGGER_CHANNEL,
    expected_lag_seconds: float = DEFAULT_EXPECTED_LAG_SECONDS,
    match_tolerance_seconds: float = DEFAULT_MATCH_TOLERANCE_SECONDS,
    min_separation_seconds: float = DEFAULT_MIN_SEPARATION_SECONDS,
    trigger_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Audit all requested participants and return rows, summaries, and metadata."""

    participant_tuple = tuple(str(value) for value in participants)
    if not participant_tuple or len(set(participant_tuple)) != len(participant_tuple):
        raise ValueError("participants must contain unique identifiers.")
    frames = [
        audit_subject_press_timing(
            dataset_root,
            subject,
            channel_label=channel_label,
            expected_lag_seconds=expected_lag_seconds,
            match_tolerance_seconds=match_tolerance_seconds,
            min_separation_seconds=min_separation_seconds,
            trigger_threshold=trigger_threshold,
        )
        for subject in participant_tuple
    ]
    raw = pd.concat(frames, ignore_index=True)
    per_press, per_subject = finalize_press_timing_rows(raw, default_lag_seconds=expected_lag_seconds)
    metadata = {
        "format": "neureptrace_katja_press_timing_audit_v1",
        "dataset_root": str(Path(dataset_root)),
        "participants": list(participant_tuple),
        "trigger_channel": str(channel_label),
        "expected_lag_seconds": float(expected_lag_seconds),
        "match_tolerance_seconds": float(match_tolerance_seconds),
        "min_separation_seconds": float(min_separation_seconds),
        "trigger_threshold": None if trigger_threshold is None else float(trigger_threshold),
        "behavioral_time_definition": "(cueDur + timing) / 1000",
        "recommended_time_preference": "matched UPPT002 onset; otherwise behavioral time plus the participant median matched lag; otherwise configured lag",
        "includes_first_press": True,
        "n_press_rows": int(per_press.shape[0]),
        "n_subjects": int(per_subject.shape[0]),
    }
    return per_press, per_subject, metadata


def _parse_csv(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not result:
        raise ValueError("Comma-separated participant list must not be empty.")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--participants", default=",".join(DEFAULT_PARTICIPANTS))
    parser.add_argument("--channel-label", default=DEFAULT_TRIGGER_CHANNEL)
    parser.add_argument("--expected-lag-ms", type=float, default=30.0)
    parser.add_argument("--match-tolerance-ms", type=float, default=150.0)
    parser.add_argument("--min-separation-ms", type=float, default=50.0)
    parser.add_argument("--trigger-threshold", type=float)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    per_press, per_subject, metadata = audit_dataset_press_timing(
        args.dataset_root,
        participants=_parse_csv(args.participants),
        channel_label=args.channel_label,
        expected_lag_seconds=float(args.expected_lag_ms) / 1000.0,
        match_tolerance_seconds=float(args.match_tolerance_ms) / 1000.0,
        min_separation_seconds=float(args.min_separation_ms) / 1000.0,
        trigger_threshold=args.trigger_threshold,
    )
    per_press.to_csv(output / "katja_press_timing_per_press.csv.gz", index=False)
    per_subject.to_csv(output / "katja_press_timing_per_subject.csv", index=False)
    (output / "katja_press_timing_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(per_subject.to_string(index=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

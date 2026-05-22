"""Stage selected OpenNeuro MEG datasets for NeuRepTrace LOSO decoding."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from neureptrace.dataset_config import parse_participant_ids


@dataclass(frozen=True)
class OpenNeuroMegSpec:
    dataset_id: str
    name: str
    default_subjects: tuple[int, ...]
    subject_width: int
    task: str
    runs: tuple[str | None, ...]
    session: str | None = None
    run_width: int | None = None
    default_label_column: str = "condition"
    default_tmin: float = -0.2
    default_tmax: float = 0.8
    default_resample_sfreq: float | None = 250.0


DATASET_SPECS: dict[str, OpenNeuroMegSpec] = {
    "ds004276": OpenNeuroMegSpec(
        dataset_id="ds004276",
        name="auditory_words",
        default_subjects=tuple(range(1, 19)),
        subject_width=3,
        task="words",
        runs=(None,),
        default_label_column="word_length_binary",
        default_tmin=-0.2,
        default_tmax=0.8,
    ),
    "ds006629": OpenNeuroMegSpec(
        dataset_id="ds006629",
        name="singsing_mmnhcs",
        default_subjects=(1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21),
        subject_width=2,
        task="MMNHCS",
        runs=("0",),
        default_label_column="trial_type",
        default_tmin=-0.2,
        default_tmax=0.8,
    ),
    "ds004330": OpenNeuroMegSpec(
        dataset_id="ds004330",
        name="object_drawing",
        default_subjects=(1, 2, *range(4, 32)),
        subject_width=2,
        task="main",
        runs=("01", "02", "03", "04", "05", "06", "07", "08", "09"),
        session="01",
        run_width=2,
        default_label_column="stimulus_form",
        default_tmin=-0.2,
        default_tmax=0.8,
    ),
}


@dataclass(frozen=True)
class RunFiles:
    subject: str
    run: str | None
    raw_path: Path
    events_path: Path
    behavior_path: Path | None = None


@dataclass(frozen=True)
class StageResult:
    dataset_id: str
    subject: str
    epochs_path: Path
    n_trials: int
    labels: tuple[str, ...]
    runs: tuple[str, ...]


def normalize_dataset_id(dataset_id: str) -> str:
    normalized = str(dataset_id).strip().lower()
    if normalized not in DATASET_SPECS:
        raise ValueError(f"Unknown OpenNeuro MEG dataset '{dataset_id}'. Available datasets: {', '.join(sorted(DATASET_SPECS))}.")
    return normalized


def _split_csv_or_space(value: str) -> list[str]:
    return [part.strip() for chunk in str(value).split(",") for part in chunk.split() if part.strip()]


def parse_subjects(spec: OpenNeuroMegSpec, subjects: str | Iterable[str | int] | None) -> tuple[int | str, ...]:
    if subjects is None or str(subjects).strip().lower() == "all":
        return spec.default_subjects
    if isinstance(subjects, str):
        return tuple(parse_participant_ids(subjects))
    return tuple(parse_participant_ids(list(subjects)))


def parse_runs(spec: OpenNeuroMegSpec, runs: str | Iterable[str] | None) -> tuple[str | None, ...]:
    if runs is None or str(runs).strip().lower() == "all":
        return spec.runs
    if isinstance(runs, str):
        return tuple(_normalize_optional_run(part, spec=spec) for part in _split_csv_or_space(runs))
    return tuple(_normalize_optional_run(part, spec=spec) for part in runs)


def _normalize_optional_run(run: object, *, spec: OpenNeuroMegSpec) -> str | None:
    text = str(run).strip()
    if text.lower() in {"", "none", "null", "single"}:
        return None
    text = text.removeprefix("run-")
    if spec.run_width is not None and text.isdigit():
        return f"{int(text):0{spec.run_width}d}"
    return text


def subject_label(spec: OpenNeuroMegSpec, subject: int | str) -> str:
    text = str(subject).strip()
    if text.startswith("sub-"):
        return text
    try:
        return f"sub-{int(text):0{spec.subject_width}d}"
    except ValueError:
        return f"sub-{text}"


def _stem(spec: OpenNeuroMegSpec, subject: str, run: str | None) -> str:
    parts = [subject]
    if spec.session is not None:
        parts.append(f"ses-{spec.session}")
    parts.append(f"task-{spec.task}")
    if run is not None:
        parts.append(f"run-{run}")
    return "_".join(parts)


def run_files(spec: OpenNeuroMegSpec, bids_root: Path, subject: int | str, run: str | None) -> RunFiles:
    subject_name = subject_label(spec, subject)
    meg_dir = bids_root / subject_name
    if spec.session is not None:
        meg_dir = meg_dir / f"ses-{spec.session}"
    meg_dir = meg_dir / "meg"
    stem = _stem(spec, subject_name, run)
    behavior_path = None
    if spec.dataset_id == "ds004276":
        behavior_path = bids_root / subject_name / "beh" / f"{subject_name}_task-{spec.task}_beh.tsv"
    return RunFiles(
        subject=subject_name,
        run=run,
        raw_path=meg_dir / f"{stem}_meg.fif",
        events_path=meg_dir / f"{stem}_events.tsv",
        behavior_path=behavior_path,
    )


def expected_relative_files(dataset_id: str, *, subjects: str | Iterable[str | int] | None = None, runs: str | Iterable[str] | None = None) -> list[str]:
    spec = DATASET_SPECS[normalize_dataset_id(dataset_id)]
    paths: list[str] = []
    for subject in parse_subjects(spec, subjects):
        for run in parse_runs(spec, runs):
            files = run_files(spec, Path("."), subject, run)
            paths.extend(
                str(path).replace("\\", "/").lstrip("./")
                for path in (files.raw_path, files.events_path, files.behavior_path)
                if path is not None
            )
    return list(dict.fromkeys(paths))


def check_raw_files(dataset_id: str, *, bids_root: Path, subjects: str | Iterable[str | int] | None = None, runs: str | Iterable[str] | None = None) -> list[Path]:
    missing = []
    for relative_path in expected_relative_files(dataset_id, subjects=subjects, runs=runs):
        path = bids_root / relative_path
        if not path.is_file():
            missing.append(path)
    return missing


def _baseline(value: str) -> tuple[float | None, float | None] | None:
    text = str(value).strip().lower()
    if text in {"none", "off", "false"}:
        return None
    parts = _split_csv_or_space(value)
    if len(parts) != 2:
        raise ValueError("--baseline must be 'none' or two values such as 'None,0'.")
    start, stop = parts
    return (None if start.lower() == "none" else float(start), None if stop.lower() == "none" else float(stop))


def _events_from_metadata(raw: mne.io.BaseRaw, metadata: pd.DataFrame, *, label_column: str) -> np.ndarray:
    onsets = pd.to_numeric(metadata["onset"], errors="raise").to_numpy(dtype=float)
    samples = raw.time_as_index(onsets, use_rounding=True) + raw.first_samp
    labels = sorted(str(label) for label in metadata[label_column].dropna().unique())
    code_by_label = {label: index + 1 for index, label in enumerate(labels)}
    event_codes = metadata[label_column].map(lambda value: code_by_label[str(value)]).to_numpy(dtype=int)
    return np.column_stack([samples, np.zeros(len(samples), dtype=int), event_codes])


def _event_id(metadata: pd.DataFrame, *, label_column: str) -> dict[str, int]:
    labels = sorted(str(label) for label in metadata[label_column].dropna().unique())
    return {label: index + 1 for index, label in enumerate(labels)}


def _ds004276_sound_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return ds004276 rows corresponding to auditory word events."""
    if "trial_type" not in events.columns:
        return events
    trial_type = events["trial_type"].astype(str)
    sound_events = events[trial_type.isin({"item", "item_post_probe"})].copy()
    return sound_events.reset_index(drop=True) if not sound_events.empty else events


def _derive_metadata(spec: OpenNeuroMegSpec, files: RunFiles, events: pd.DataFrame) -> pd.DataFrame:
    metadata = events.copy().reset_index(drop=True)
    if "trial_type" in metadata.columns:
        metadata["trial_type"] = metadata["trial_type"].astype(str)

    if spec.dataset_id == "ds004276":
        if files.behavior_path is None or not files.behavior_path.is_file():
            raise FileNotFoundError(f"Missing ds004276 behavior file: {files.behavior_path}")
        behavior = pd.read_csv(files.behavior_path, sep="\t")
        sound_rows = behavior[behavior["Event_Type"].astype(str) == "Sound"].reset_index(drop=True)
        metadata = _ds004276_sound_events(metadata)
        if len(sound_rows) != len(metadata):
            event_counts = events["trial_type"].astype(str).value_counts().to_dict() if "trial_type" in events.columns else {}
            raise ValueError(
                f"{files.events_path} has {len(metadata)} word events but {files.behavior_path} has "
                f"{len(sound_rows)} sound rows. Event trial_type counts: {event_counts}."
            )
        words = sound_rows["Code"].astype(str)
        word_lengths = words.str.len()
        metadata["word"] = words
        metadata["word_length"] = word_lengths
        metadata["word_length_class"] = word_lengths.map(lambda value: "short" if value <= 4 else ("medium" if value <= 7 else "long"))
        metadata["word_length_binary"] = word_lengths.map(lambda value: "short" if value <= 4 else ("long" if value >= 8 else pd.NA))
        metadata["behavior_trial"] = sound_rows["Trial"].to_numpy()
        metadata["behavior_stim_type"] = sound_rows["Stim_Type"].to_numpy()

    if spec.dataset_id == "ds004330":
        parsed = metadata["trial_type"].str.extract(r"^(?P<stimulus_form>[^_]+)_(?P<stimulus_id>\d+)$")
        metadata["stimulus_form"] = parsed["stimulus_form"]
        metadata["stimulus_id"] = parsed["stimulus_id"]
        metadata["stimulus_modality"] = metadata["stimulus_form"].map(lambda value: "photo" if str(value) == "Photo" else "drawing")

    return metadata


def _filter_metadata(
    metadata: pd.DataFrame,
    *,
    label_column: str,
    include_labels: Sequence[str] | None,
    max_events_per_label: int | None,
    selection: str,
    seed: int,
) -> pd.DataFrame:
    if label_column not in metadata.columns:
        raise ValueError(f"Requested label column '{label_column}' is not available. Columns: {', '.join(metadata.columns)}.")
    filtered = metadata[metadata[label_column].notna()].copy()
    filtered[label_column] = filtered[label_column].astype(str)
    if include_labels:
        wanted = {str(label) for label in include_labels}
        filtered = filtered[filtered[label_column].isin(wanted)].copy()
    filtered = _limit_metadata_per_label(
        filtered,
        label_column=label_column,
        max_events_per_label=max_events_per_label,
        selection=selection,
        seed=seed,
    )
    if filtered.empty:
        raise ValueError(f"No events remain after filtering label column '{label_column}'.")
    filtered["condition"] = filtered[label_column].astype(str)
    return filtered.reset_index(drop=True)


def _limit_metadata_per_label(
    metadata: pd.DataFrame,
    *,
    label_column: str,
    max_events_per_label: int | None,
    selection: str,
    seed: int,
) -> pd.DataFrame:
    filtered = metadata.copy()
    if max_events_per_label is not None:
        if selection not in {"first", "random"}:
            raise ValueError("--selection must be first or random.")
        pieces = []
        for label, group in filtered.groupby(label_column, sort=True):
            if selection == "random" and len(group) > max_events_per_label:
                group = group.sample(n=max_events_per_label, random_state=seed + stable_label_seed(label))
            else:
                group = group.head(max_events_per_label)
            pieces.append(group.sort_index())
        filtered = pd.concat(pieces).sort_index().reset_index(drop=True) if pieces else filtered.iloc[0:0].copy()
    return filtered.reset_index(drop=True)


def _drop_non_epochable_metadata(
    raw: mne.io.BaseRaw,
    metadata: pd.DataFrame,
    *,
    label_column: str,
    tmin: float,
    tmax: float,
) -> pd.DataFrame:
    """Remove events whose requested epoch window falls outside raw data."""
    event_samples = _events_from_metadata(raw, metadata, label_column=label_column)[:, 0]
    sfreq = float(raw.info["sfreq"])
    starts = event_samples + int(np.floor(tmin * sfreq))
    stops = event_samples + int(np.ceil(tmax * sfreq))
    keep = (starts >= raw.first_samp) & (stops <= raw.last_samp)
    if keep.all():
        return metadata.reset_index(drop=True)
    kept = metadata.loc[keep].reset_index(drop=True)
    if kept.empty:
        raise ValueError(
            f"No events remain after dropping epochs outside raw bounds "
            f"({raw.first_samp}..{raw.last_samp} samples)."
        )
    return kept


def stable_label_seed(label: object) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(str(label))) % 100_000


def stage_run(
    spec: OpenNeuroMegSpec,
    files: RunFiles,
    *,
    label_column: str,
    include_labels: Sequence[str] | None,
    max_events_per_label: int | None,
    selection: str,
    seed: int,
    tmin: float,
    tmax: float,
    baseline: tuple[float | None, float | None] | None,
    picks: str,
    resample_sfreq: float | None,
) -> mne.Epochs:
    if not files.raw_path.is_file():
        raise FileNotFoundError(f"Missing raw FIF file: {files.raw_path}")
    if not files.events_path.is_file():
        raise FileNotFoundError(f"Missing events TSV file: {files.events_path}")

    raw = mne.io.read_raw_fif(files.raw_path, preload=False, verbose="error")
    events = pd.read_csv(files.events_path, sep="\t")
    metadata = _derive_metadata(spec, files, events)
    metadata = _filter_metadata(
        metadata,
        label_column=label_column,
        include_labels=include_labels,
        max_events_per_label=None,
        selection=selection,
        seed=seed,
    )
    metadata = _drop_non_epochable_metadata(raw, metadata, label_column=label_column, tmin=tmin, tmax=tmax)
    metadata = _limit_metadata_per_label(
        metadata,
        label_column=label_column,
        max_events_per_label=max_events_per_label,
        selection=selection,
        seed=seed,
    )
    metadata["dataset"] = spec.dataset_id
    metadata["subject"] = files.subject
    metadata["run"] = "" if files.run is None else files.run
    epochs = mne.Epochs(
        raw,
        _events_from_metadata(raw, metadata, label_column=label_column),
        event_id=_event_id(metadata, label_column=label_column),
        tmin=tmin,
        tmax=tmax,
        baseline=baseline,
        metadata=metadata,
        picks=picks,
        preload=True,
        reject_by_annotation=True,
        verbose="error",
    )
    if resample_sfreq is not None and epochs.info["sfreq"] > float(resample_sfreq):
        epochs.resample(float(resample_sfreq), verbose="error")
    return epochs


def stage_subject(
    dataset_id: str,
    *,
    bids_root: Path,
    staged_dir: Path,
    subject: int | str,
    runs: str | Iterable[str] | None = None,
    label_column: str | None = None,
    include_labels: Sequence[str] | None = None,
    max_events_per_label: int | None = None,
    selection: str = "random",
    seed: int = 13,
    tmin: float | None = None,
    tmax: float | None = None,
    baseline: tuple[float | None, float | None] | None = (None, 0.0),
    picks: str = "meg",
    resample_sfreq: float | None = None,
    on_mismatch: str = "warn",
    overwrite: bool = False,
) -> StageResult:
    spec = DATASET_SPECS[normalize_dataset_id(dataset_id)]
    subject_name = subject_label(spec, subject)
    staged_dataset_dir = staged_dir / spec.dataset_id
    staged_dataset_dir.mkdir(parents=True, exist_ok=True)
    epochs_path = staged_dataset_dir / f"{subject_name}_{spec.dataset_id}_{spec.name}_epo.fif"
    if epochs_path.is_file() and not overwrite:
        epochs = mne.read_epochs(epochs_path, preload=False, verbose="error")
        metadata = epochs.metadata if epochs.metadata is not None else pd.DataFrame()
        labels = tuple(sorted(str(label) for label in metadata.get("condition", pd.Series(dtype=str)).dropna().unique()))
        runs_seen = tuple(sorted(str(run) for run in metadata.get("run", pd.Series(dtype=str)).dropna().unique()))
        return StageResult(spec.dataset_id, subject_name, epochs_path, len(metadata), labels, runs_seen)

    label_column = label_column or spec.default_label_column
    run_epochs = [
        stage_run(
            spec,
            run_files(spec, bids_root, subject, run),
            label_column=label_column,
            include_labels=include_labels,
            max_events_per_label=max_events_per_label,
            selection=selection,
            seed=seed,
            tmin=spec.default_tmin if tmin is None else float(tmin),
            tmax=spec.default_tmax if tmax is None else float(tmax),
            baseline=baseline,
            picks=picks,
            resample_sfreq=spec.default_resample_sfreq if resample_sfreq is None else (None if resample_sfreq <= 0 else resample_sfreq),
        )
        for run in parse_runs(spec, runs)
    ]
    epochs = run_epochs[0] if len(run_epochs) == 1 else mne.concatenate_epochs(run_epochs, add_offset=True, on_mismatch=on_mismatch, verbose="error")
    if epochs.metadata is None:
        raise ValueError(f"Staged epochs for {subject_name} do not contain metadata.")
    metadata = epochs.metadata.reset_index(drop=True)
    labels = tuple(sorted(str(label) for label in metadata["condition"].dropna().unique()))
    runs_seen = tuple(sorted(str(run) for run in metadata["run"].dropna().unique()))
    epochs.save(epochs_path, overwrite=True)
    metadata.to_csv(epochs_path.with_suffix(".csv"), index=False)
    return StageResult(spec.dataset_id, subject_name, epochs_path, len(metadata), labels, runs_seen)


def stage_dataset(
    dataset_id: str,
    *,
    bids_root: Path,
    staged_dir: Path,
    subjects: str | Iterable[str | int] | None = None,
    runs: str | Iterable[str] | None = None,
    label_column: str | None = None,
    include_labels: Sequence[str] | None = None,
    max_events_per_label: int | None = None,
    selection: str = "random",
    seed: int = 13,
    tmin: float | None = None,
    tmax: float | None = None,
    baseline: tuple[float | None, float | None] | None = (None, 0.0),
    picks: str = "meg",
    resample_sfreq: float | None = None,
    on_mismatch: str = "warn",
    overwrite: bool = False,
) -> list[StageResult]:
    spec = DATASET_SPECS[normalize_dataset_id(dataset_id)]
    return [
        stage_subject(
            spec.dataset_id,
            bids_root=bids_root,
            staged_dir=staged_dir,
            subject=subject,
            runs=runs,
            label_column=label_column,
            include_labels=include_labels,
            max_events_per_label=max_events_per_label,
            selection=selection,
            seed=seed,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            picks=picks,
            resample_sfreq=resample_sfreq,
            on_mismatch=on_mismatch,
            overwrite=overwrite,
        )
        for subject in parse_subjects(spec, subjects)
    ]


def write_stage_summary(results: Sequence[StageResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "subject", "epochs_path", "n_trials", "labels", "runs"])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "dataset_id": result.dataset_id,
                    "subject": result.subject,
                    "epochs_path": result.epochs_path.as_posix(),
                    "n_trials": result.n_trials,
                    "labels": "|".join(result.labels),
                    "runs": "|".join(result.runs),
                }
            )


def _subject_arg(value: str) -> str:
    return value


def _add_dataset_subject_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_SPECS))
    parser.add_argument("--subjects", default="all", type=_subject_arg, help="Subject ids/ranges, or all.")
    parser.add_argument("--runs", default="all", help="Run ids, comma/space separated, or all.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-datasets", help="List supported OpenNeuro MEG recipes.")
    list_parser.set_defaults(func=_main_list_datasets)

    includes_parser = subparsers.add_parser("print-download-includes", help="Print OpenNeuro include paths for selected raw files.")
    _add_dataset_subject_run_args(includes_parser)
    includes_parser.set_defaults(func=_main_print_download_includes)

    check_parser = subparsers.add_parser("check-raw", help="Check whether selected raw BIDS files exist locally.")
    _add_dataset_subject_run_args(check_parser)
    check_parser.add_argument("--bids-root", type=Path, required=True)
    check_parser.set_defaults(func=_main_check_raw)

    stage_parser = subparsers.add_parser("stage", help="Stage selected raw BIDS MEG files into per-subject MNE Epochs FIF files.")
    _add_dataset_subject_run_args(stage_parser)
    stage_parser.add_argument("--bids-root", type=Path, required=True)
    stage_parser.add_argument("--staged-dir", type=Path, required=True)
    stage_parser.add_argument("--summary-out", type=Path)
    stage_parser.add_argument("--label-column")
    stage_parser.add_argument("--include-label", action="append", dest="include_labels")
    stage_parser.add_argument("--max-events-per-label", type=int)
    stage_parser.add_argument("--selection", choices=["first", "random"], default="random")
    stage_parser.add_argument("--seed", type=int, default=13)
    stage_parser.add_argument("--tmin", type=float)
    stage_parser.add_argument("--tmax", type=float)
    stage_parser.add_argument("--baseline", default="None,0")
    stage_parser.add_argument("--picks", default="meg")
    stage_parser.add_argument("--resample-sfreq", type=float)
    stage_parser.add_argument("--no-resample", action="store_true")
    stage_parser.add_argument("--on-mismatch", choices=["raise", "warn", "ignore"], default="warn")
    stage_parser.add_argument("--overwrite", action="store_true")
    stage_parser.set_defaults(func=_main_stage)

    args = parser.parse_args(argv)
    return args.func(args)


def _main_list_datasets(_args) -> int:
    for spec in DATASET_SPECS.values():
        print(f"{spec.dataset_id}\t{spec.name}\tsubjects={len(spec.default_subjects)}\tdefault_label={spec.default_label_column}")
    return 0


def _main_print_download_includes(args) -> int:
    for relative_path in expected_relative_files(args.dataset, subjects=args.subjects, runs=args.runs):
        print(relative_path)
    return 0


def _main_check_raw(args) -> int:
    missing = check_raw_files(args.dataset, bids_root=args.bids_root, subjects=args.subjects, runs=args.runs)
    if not missing:
        print(f"All selected {args.dataset} raw files are present under {args.bids_root}.")
        return 0
    print(f"Missing {len(missing)} selected {args.dataset} raw file(s):")
    for path in missing:
        print(path)
    return 1


def _main_stage(args) -> int:
    include_labels = None
    if args.include_labels:
        include_labels = [label for value in args.include_labels for label in _split_csv_or_space(value)]
    resample_sfreq = 0.0 if args.no_resample else args.resample_sfreq
    results = stage_dataset(
        args.dataset,
        bids_root=args.bids_root,
        staged_dir=args.staged_dir,
        subjects=args.subjects,
        runs=args.runs,
        label_column=args.label_column,
        include_labels=include_labels,
        max_events_per_label=args.max_events_per_label,
        selection=args.selection,
        seed=args.seed,
        tmin=args.tmin,
        tmax=args.tmax,
        baseline=_baseline(args.baseline),
        picks=args.picks,
        resample_sfreq=resample_sfreq,
        on_mismatch=args.on_mismatch,
        overwrite=args.overwrite,
    )
    for result in results:
        print(f"Staged {result.subject}: {result.n_trials} trials, labels={','.join(result.labels)}, runs={','.join(result.runs)}")
    if args.summary_out is not None:
        write_stage_summary(results, args.summary_out)
        print(f"Wrote OpenNeuro MEG stage summary: {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

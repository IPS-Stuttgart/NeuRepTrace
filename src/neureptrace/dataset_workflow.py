from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PathToken = str | int

SUPPORTED_INPUT_FORMATS = {"mne-epochs", "fieldtrip-mat"}
SUPPORTED_TASKS = {"time_decode", "time-decode", "mne_time_decode", "mne-time-decode"}
DEFAULT_SCHEMA_VERSION = "neureptrace.dataset.v1"
DEFAULT_LABEL_COLUMN = "condition"
DEFAULT_SUBJECT_TEMPLATE = "Part{participant}"
DEFAULT_OUTPUT_TEMPLATE = "outputs/{analysis}_{subject}.csv"
DEFAULT_OBSERVATIONS_TEMPLATE = "outputs/{analysis}_{subject}_observations.csv"


@dataclass(frozen=True)
class LoaderSpec:
    input_format: str
    root_path: tuple[PathToken, ...]
    trim_overlong_labels: bool
    label_base: int
    ch_type: str
    label_column: str


@dataclass(frozen=True)
class RecordingSpec:
    name: str
    path: str | None
    pattern: str | None
    input_format: str | None
    metadata_csv: str | None


@dataclass(frozen=True)
class AnalysisSpec:
    name: str
    task: str
    recording: str
    label_column: str | None
    output_csv: str
    observations_out: str | None
    options: dict[str, Any]


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    schema_version: str
    dataset_id: str
    root: Path
    subject_template: str
    participants: tuple[int | str, ...]
    loader: LoaderSpec
    recordings: dict[str, RecordingSpec]
    analyses: tuple[AnalysisSpec, ...]


@dataclass(frozen=True)
class DatasetRunPlan:
    analysis: str
    task: str
    participant: int | str | None
    subject: str | None
    recording: str
    input_format: str
    recording_path: Path
    metadata_csv: Path | None
    output_csv: Path
    observations_out: Path | None
    label_column: str
    options: dict[str, Any]


@dataclass(frozen=True)
class DatasetRunResult:
    plan: DatasetRunPlan
    executed: bool


def load_dataset_config(config_path: Path | str) -> DatasetConfig:
    """Load and validate a NeuRepTrace dataset YAML/JSON configuration."""

    path = Path(config_path).expanduser().resolve()
    data = _load_config_mapping(path)

    schema_version = str(data.get("schema_version", DEFAULT_SCHEMA_VERSION))
    if schema_version != DEFAULT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset config schema_version '{schema_version}'. Expected '{DEFAULT_SCHEMA_VERSION}'.")

    dataset_block = _mapping(data.get("dataset", {}), "dataset")
    dataset_id = str(dataset_block.get("id", path.stem))
    root_value = dataset_block.get("root", data.get("root", path.parent))
    root = _resolve_path(str(root_value), base_dir=path.parent)
    subject_template = str(dataset_block.get("subject_template", data.get("subject_template", DEFAULT_SUBJECT_TEMPLATE)))

    participants = tuple(_expand_participants(data.get("participants", dataset_block.get("participants"))))
    loader = _loader_spec(_mapping(data.get("loader", {}), "loader"))
    recordings = _recording_specs(_mapping(data.get("recordings", {}), "recordings"))
    analyses = tuple(_analysis_specs(data.get("analyses", []), loader=loader))

    if not recordings:
        raise ValueError("Dataset config must define at least one recording under 'recordings'.")
    if not analyses:
        raise ValueError("Dataset config must define at least one analysis under 'analyses'.")

    return DatasetConfig(
        path=path,
        schema_version=schema_version,
        dataset_id=dataset_id,
        root=root,
        subject_template=subject_template,
        participants=participants,
        loader=loader,
        recordings=recordings,
        analyses=analyses,
    )


def plan_dataset_runs(
    config: DatasetConfig | Path | str,
    *,
    participants: Sequence[int | str] | None = None,
    analyses: Sequence[str] | None = None,
) -> list[DatasetRunPlan]:
    """Expand a dataset config into concrete per-participant/per-analysis run plans."""

    if not isinstance(config, DatasetConfig):
        config = load_dataset_config(config)

    requested_participants = None if participants is None else {str(participant) for participant in _expand_participants(participants)}
    requested_analyses = None if analyses is None else {str(analysis) for analysis in analyses}
    config_participants = config.participants or (None,)
    plans: list[DatasetRunPlan] = []
    seen_outputs: dict[Path, DatasetRunPlan] = {}

    for participant in config_participants:
        if requested_participants is not None and str(participant) not in requested_participants:
            continue
        subject = _format_template(config.subject_template, participant=participant, analysis="", recording="") if participant is not None else None
        for analysis in config.analyses:
            if requested_analyses is not None and analysis.name not in requested_analyses:
                continue
            if analysis.task not in SUPPORTED_TASKS:
                raise ValueError(f"Analysis '{analysis.name}' uses unsupported task '{analysis.task}'. Supported tasks: {sorted(SUPPORTED_TASKS)}.")
            try:
                recording = config.recordings[analysis.recording]
            except KeyError as exc:
                raise ValueError(f"Analysis '{analysis.name}' references unknown recording '{analysis.recording}'.") from exc

            context = {
                "participant": "" if participant is None else participant,
                "subject": "" if subject is None else subject,
                "analysis": analysis.name,
                "recording": recording.name,
                "dataset": config.dataset_id,
            }
            recording_path = _recording_path(config.root, recording, context)
            input_format = _normalize_input_format(recording.input_format or config.loader.input_format)
            metadata_csv = _optional_template_path(recording.metadata_csv, base_dir=config.root, context=context)
            output_csv = _template_path(analysis.output_csv, base_dir=config.path.parent, context=context)
            observations_out = _optional_template_path(analysis.observations_out, base_dir=config.path.parent, context=context)
            label_column = analysis.label_column or config.loader.label_column
            plan = DatasetRunPlan(
                analysis=analysis.name,
                task=analysis.task,
                participant=participant,
                subject=subject,
                recording=recording.name,
                input_format=input_format,
                recording_path=recording_path,
                metadata_csv=metadata_csv,
                output_csv=output_csv,
                observations_out=observations_out,
                label_column=label_column,
                options=dict(analysis.options),
            )
            if output_csv in seen_outputs:
                previous = seen_outputs[output_csv]
                raise ValueError(
                    f"Dataset config maps multiple runs to the same output CSV '{output_csv}': "
                    f"{previous.analysis}/{previous.subject} and {analysis.name}/{subject}. "
                    "Use {participant}, {subject}, or {analysis} in output_csv."
                )
            seen_outputs[output_csv] = plan
            plans.append(plan)
    return plans


def run_dataset_config(
    config_path: Path | str,
    *,
    participants: Sequence[int | str] | None = None,
    analyses: Sequence[str] | None = None,
    dry_run: bool = False,
    staging_dir: Path | str | None = None,
    overwrite_staged: bool = True,
) -> list[DatasetRunResult]:
    """Run all selected analyses from a dataset YAML/JSON config."""

    config = load_dataset_config(config_path)
    plans = plan_dataset_runs(config, participants=participants, analyses=analyses)
    staging_base = _resolve_path(str(staging_dir), base_dir=config.path.parent) if staging_dir is not None else config.path.parent / ".neureptrace_staged"

    results: list[DatasetRunResult] = []
    for plan in plans:
        if dry_run:
            results.append(DatasetRunResult(plan=plan, executed=False))
            continue
        epochs_path, metadata_csv = _prepare_decoder_input(
            config=config,
            plan=plan,
            staging_dir=staging_base,
            overwrite_staged=overwrite_staged,
        )
        _run_time_resolved_decode(
            epochs_path=epochs_path,
            metadata_csv=metadata_csv,
            label_column=plan.label_column,
            out_path=plan.output_csv,
            observation_out_path=plan.observations_out,
            subject=plan.subject,
            **plan.options,
        )
        results.append(DatasetRunResult(plan=plan, executed=True))
    return results


def _load_config_mapping(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise RuntimeError("YAML dataset configs require PyYAML. Install the package dependency 'pyyaml'.") from exc
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise ValueError(f"Unsupported dataset config extension '{path.suffix}'. Use .yml, .yaml, or .json.")
    if not isinstance(loaded, Mapping):
        raise ValueError("Dataset config must contain a mapping at the top level.")
    return dict(loaded)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"'{name}' must be a mapping.")
    return dict(value)


def _resolve_path(value: str, *, base_dir: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _normalize_input_format(value: str | None) -> str:
    normalized = "mne-epochs" if value is None else str(value).strip().lower().replace("_", "-")
    aliases = {
        "mne": "mne-epochs",
        "mne-epochs-fif": "mne-epochs",
        "epochs": "mne-epochs",
        "fif": "mne-epochs",
        "fieldtrip": "fieldtrip-mat",
        "fieldtrip-raw": "fieldtrip-mat",
        "fieldtrip-raw-mat": "fieldtrip-mat",
        "matlab-fieldtrip-raw": "fieldtrip-mat",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_INPUT_FORMATS:
        raise ValueError(f"Unsupported input format '{value}'. Supported formats: {sorted(SUPPORTED_INPUT_FORMATS)}.")
    return normalized


def _path_tokens(value: Any, *, default: Sequence[PathToken]) -> tuple[PathToken, ...]:
    if value is None:
        return tuple(default)
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise ValueError("root_path must be a comma-separated string or a sequence.")
    tokens: list[PathToken] = []
    for item in items:
        if isinstance(item, int):
            tokens.append(item)
        elif isinstance(item, str) and re.fullmatch(r"[+-]?\d+", item.strip()):
            tokens.append(int(item))
        else:
            tokens.append(str(item))
    return tuple(tokens)


def _loader_spec(loader: Mapping[str, Any]) -> LoaderSpec:
    input_format = loader.get("input_format")
    if input_format is None and str(loader.get("format", "")).strip().lower() == "mat":
        structure = str(loader.get("structure", loader.get("adapter", ""))).strip().lower().replace("_", "-")
        if structure in {"fieldtrip", "fieldtrip-raw", "fieldtrip-raw-mat", "matlab-fieldtrip-raw"}:
            input_format = "fieldtrip-mat"
    return LoaderSpec(
        input_format=_normalize_input_format(input_format),
        root_path=_path_tokens(loader.get("root_path"), default=("data", 0)),
        trim_overlong_labels=bool(loader.get("trim_overlong_labels", True)),
        label_base=int(loader.get("label_base", 1)),
        ch_type=str(loader.get("ch_type", "grad")),
        label_column=str(loader.get("label_column", DEFAULT_LABEL_COLUMN)),
    )


def _recording_specs(recordings: Mapping[str, Any]) -> dict[str, RecordingSpec]:
    specs: dict[str, RecordingSpec] = {}
    for name, value in recordings.items():
        if isinstance(value, str):
            block = {"pattern": value}
        else:
            block = _mapping(value, f"recordings.{name}")
        path = block.get("path")
        pattern = block.get("pattern")
        if path is None and pattern is None:
            raise ValueError(f"Recording '{name}' must define either 'path' or 'pattern'.")
        specs[str(name)] = RecordingSpec(
            name=str(name),
            path=None if path is None else str(path),
            pattern=None if pattern is None else str(pattern),
            input_format=None if block.get("input_format") is None else str(block["input_format"]),
            metadata_csv=None if block.get("metadata_csv") is None else str(block["metadata_csv"]),
        )
    return specs


def _analysis_specs(analyses: Any, *, loader: LoaderSpec) -> list[AnalysisSpec]:
    if not isinstance(analyses, Sequence) or isinstance(analyses, (str, bytes)):
        raise ValueError("'analyses' must be a list of analysis mappings.")
    specs: list[AnalysisSpec] = []
    for index, value in enumerate(analyses):
        block = _mapping(value, f"analyses[{index}]")
        name = str(block.get("name", f"analysis_{index}"))
        recording = block.get("recording")
        if recording is None:
            recording = block.get("recording_name", block.get("input", block.get("train_recording")))
        if recording is None:
            raise ValueError(f"Analysis '{name}' must define a recording.")
        options = dict(block.get("options", {}))
        for key in (
            "group_column",
            "picks",
            "tmin",
            "tmax",
            "window_ms",
            "step_ms",
            "n_splits",
            "max_iter",
            "decoder",
            "emission_mode",
            "feature_preprocessor",
            "pca_components",
            "normalization",
            "baseline_window",
            "tune_hyperparameters",
            "tuning_cv_splits",
            "tuning_scoring",
            "tuning_c_grid",
            "calibration_out_path",
            "calibration_bins",
            "temporal_train_window",
        ):
            if key in block:
                options[key] = block[key]
        if "calibration_out" in block:
            options["calibration_out_path"] = block["calibration_out"]
        output_csv = str(block.get("output_csv", block.get("out", DEFAULT_OUTPUT_TEMPLATE)))
        observations_out = block.get("observations_out", block.get("observation_out", None))
        if observations_out is True:
            observations_out = DEFAULT_OBSERVATIONS_TEMPLATE
        elif observations_out is False:
            observations_out = None
        specs.append(
            AnalysisSpec(
                name=name,
                task=str(block.get("task", "time_decode")),
                recording=str(recording),
                label_column=None if block.get("label_column") is None else str(block["label_column"]),
                output_csv=output_csv,
                observations_out=None if observations_out is None else str(observations_out),
                options=options,
            )
        )
    return specs


def _expand_participants(value: Any) -> list[int | str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        value = value.get("include", [])
    if isinstance(value, (int, str)):
        items = [value]
    elif isinstance(value, Sequence):
        items = list(value)
    else:
        raise ValueError("participants must be a list, scalar, or mapping with an 'include' list.")

    expanded: list[int | str] = []
    for item in items:
        if isinstance(item, int):
            expanded.append(item)
            continue
        text = str(item).strip()
        range_match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", text)
        if range_match is not None:
            start, stop = map(int, range_match.groups())
            step = 1 if stop >= start else -1
            expanded.extend(range(start, stop + step, step))
        elif re.fullmatch(r"[+-]?\d+", text):
            expanded.append(int(text))
        else:
            expanded.append(text)
    return expanded


def _format_template(template: str, *, participant: int | str | None, subject: str | None = None, analysis: str, recording: str, dataset: str = "") -> str:
    return template.format(
        participant="" if participant is None else participant,
        subject="" if subject is None else subject,
        analysis=analysis,
        recording=recording,
        dataset=dataset,
    )


def _template_path(template: str, *, base_dir: Path, context: Mapping[str, Any]) -> Path:
    formatted = template.format(**context)
    return _resolve_path(formatted, base_dir=base_dir)


def _optional_template_path(template: str | None, *, base_dir: Path, context: Mapping[str, Any]) -> Path | None:
    return None if template is None else _template_path(template, base_dir=base_dir, context=context)


def _recording_path(root: Path, recording: RecordingSpec, context: Mapping[str, Any]) -> Path:
    if recording.path is not None:
        return _template_path(recording.path, base_dir=root, context=context)
    if recording.pattern is None:
        raise ValueError(f"Recording '{recording.name}' has neither path nor pattern.")
    return _template_path(recording.pattern, base_dir=root, context=context)


def _prepare_decoder_input(
    *,
    config: DatasetConfig,
    plan: DatasetRunPlan,
    staging_dir: Path,
    overwrite_staged: bool,
) -> tuple[Path, Path | None]:
    if plan.input_format == "mne-epochs":
        return plan.recording_path, plan.metadata_csv
    if plan.input_format == "fieldtrip-mat":
        return _stage_fieldtrip_mat(config=config, plan=plan, staging_dir=staging_dir, overwrite_staged=overwrite_staged)
    raise ValueError(f"Unsupported input format: {plan.input_format}")


def _stage_fieldtrip_mat(
    *,
    config: DatasetConfig,
    plan: DatasetRunPlan,
    staging_dir: Path,
    overwrite_staged: bool,
) -> tuple[Path, Path]:
    try:
        from neureptrace.fieldtrip_mat import load_fieldtrip_raw_mat_epochs
    except ImportError as exc:  # pragma: no cover - depends on applying the FieldTrip loader patch first
        raise RuntimeError(
            "Dataset config requested input_format: fieldtrip-mat, but neureptrace.fieldtrip_mat is not available. "
            "Apply the FieldTrip MAT loader patch first or use input_format: mne-epochs."
        ) from exc

    staging_dir.mkdir(parents=True, exist_ok=True)
    safe_subject = _safe_name(plan.subject or "dataset")
    safe_analysis = _safe_name(plan.analysis)
    safe_recording = _safe_name(plan.recording)
    epochs_path = staging_dir / f"{safe_analysis}_{safe_subject}_{safe_recording}-epo.fif"
    metadata_path = staging_dir / f"{safe_analysis}_{safe_subject}_{safe_recording}_metadata.csv"
    epochs, metadata = load_fieldtrip_raw_mat_epochs(
        plan.recording_path,
        root_path=config.loader.root_path,
        label_base=config.loader.label_base,
        trim_overlong_labels=config.loader.trim_overlong_labels,
        ch_type=config.loader.ch_type,
    )
    epochs.save(epochs_path, overwrite=overwrite_staged, verbose="error")
    metadata.to_csv(metadata_path, index=False)
    return epochs_path, metadata_path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _run_time_resolved_decode(**kwargs: Any) -> Any:
    from neureptrace.mne_time_decode import run_time_resolved_decode

    return run_time_resolved_decode(**kwargs)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NeuRepTrace analyses from a dataset YAML/JSON config.")
    parser.add_argument("config", type=Path, help="Dataset config YAML/JSON file.")
    parser.add_argument("--participant", action="append", dest="participants", help="Participant id or range to run, e.g. 10 or 13-27. May be passed multiple times.")
    parser.add_argument("--analysis", action="append", dest="analyses", help="Analysis name to run. May be passed multiple times.")
    parser.add_argument("--dry-run", action="store_true", help="Print expanded runs without executing decoders.")
    parser.add_argument("--staging-dir", type=Path, help="Directory for staged MNE FIF files created from FieldTrip MAT inputs.")
    parser.add_argument("--no-overwrite-staged", action="store_true", help="Do not overwrite staged FieldTrip-to-MNE files.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    results = run_dataset_config(
        args.config,
        participants=args.participants,
        analyses=args.analyses,
        dry_run=args.dry_run,
        staging_dir=args.staging_dir,
        overwrite_staged=not args.no_overwrite_staged,
    )
    verb = "Would run" if args.dry_run else "Ran"
    for result in results:
        plan = result.plan
        print(
            f"{verb} {plan.analysis} for {plan.subject or 'dataset'} "
            f"from {plan.recording_path} -> {plan.output_csv}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

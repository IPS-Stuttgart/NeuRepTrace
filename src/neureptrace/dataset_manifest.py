"""Generate NeuRepTrace benchmark manifests from compact dataset configs."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

_PARTICIPANT_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")

PathToken = str | int


def load_dataset_config(path: Path | str) -> dict[str, Any]:
    """Load a dataset config from JSON or YAML."""

    config_path = Path(path)
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        loaded = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is declared by the package
            raise RuntimeError("YAML dataset configs require the optional dependency 'PyYAML'.") from exc
        loaded = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported dataset config suffix '{config_path.suffix}'. Use .json, .yml, or .yaml.")
    if not isinstance(loaded, dict):
        raise ValueError("Dataset config must contain a mapping at the top level.")
    return loaded


def write_manifest_from_dataset_config(
    config_path: Path | str,
    out_path: Path | str,
    *,
    run_names: Sequence[str] | None = None,
    absolute_paths: bool = False,
) -> pd.DataFrame:
    """Write a CSV benchmark manifest from a dataset config and return it as a frame."""

    config_file = Path(config_path)
    frame = manifest_from_dataset_config(
        load_dataset_config(config_file),
        config_dir=config_file.parent,
        run_names=run_names,
        absolute_paths=absolute_paths,
    )
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_file, index=False)
    return frame


def manifest_from_dataset_config(
    config: Mapping[str, Any],
    *,
    config_dir: Path | None = None,
    run_names: Sequence[str] | None = None,
    absolute_paths: bool = False,
) -> pd.DataFrame:
    """Expand a compact dataset config into a NeuRepTrace benchmark manifest."""

    dataset = _mapping(config.get("dataset", {}), name="dataset")
    files = _mapping(config.get("files", {}), name="files")
    if not files:
        raise ValueError("Dataset config must define at least one file pattern under 'files'.")

    participants = _participants(config.get("participants", {}))
    if not participants:
        raise ValueError("Dataset config must define at least one participant.")

    all_runs = _runs(config, files=files, run_names=None)
    selected_runs = _runs(config, files=files, run_names=run_names)
    root = _string(dataset.get("root", ""))
    input_format = _string(dataset.get("input_format", dataset.get("format", "mne-epochs")))
    subject_template = _string(dataset.get("subject_template", "Part{participant}"))
    path_base = config_dir if absolute_paths and config_dir is not None else None

    common_columns = _common_manifest_columns(config)
    rows: list[dict[str, Any]] = []
    for participant in participants:
        for run in selected_runs:
            file_key = _string(run.get("file", run.get("file_key", run["name"])))
            if file_key not in files:
                raise ValueError(f"Run '{run['name']}' references unknown file key '{file_key}'.")
            file_path = _materialize_file_path(
                files[file_key],
                participant=participant,
                root=root,
                config_dir=path_base,
            )
            subject = subject_template.format(participant=participant, file=file_key, run=run["name"])
            row: dict[str, Any] = {
                "subject": subject,
                "epochs": file_path,
                "input": file_path,
                "input_format": _string(run.get("input_format", input_format)),
            }
            variant = run.get("variant", run["name"] if len(all_runs) > 1 else None)
            if variant is not None:
                row["variant"] = _string(variant)
            row.update(common_columns)
            row.update(_manifest_columns_from_mapping(run.get("columns", {}), prefix=None))
            rows.append(row)

    return pd.DataFrame(rows)


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"'{name}' must be a mapping.")
    return dict(value)


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _participants(value: Any) -> list[int]:
    participants_cfg = value
    if isinstance(value, Mapping):
        participants_cfg = value.get("include", value.get("participants", []))
        exclude_cfg = value.get("exclude", [])
    else:
        exclude_cfg = []
    participants = _participant_values(participants_cfg)
    excluded = set(_participant_values(exclude_cfg))
    return [participant for participant in participants if participant not in excluded]


def _participant_values(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
        return _participant_values(tokens)
    if isinstance(value, Iterable):
        participants: list[int] = []
        for item in value:
            if isinstance(item, int):
                participants.append(item)
                continue
            token = str(item).strip()
            match = _PARTICIPANT_RANGE_RE.match(token)
            if match:
                start, stop = int(match.group(1)), int(match.group(2))
                step = 1 if stop >= start else -1
                participants.extend(range(start, stop + step, step))
            elif token:
                participants.append(int(token))
        return participants
    raise ValueError("participants.include must be an integer, string, or sequence.")


def _runs(config: Mapping[str, Any], *, files: Mapping[str, Any], run_names: Sequence[str] | None) -> list[dict[str, Any]]:
    raw_runs = config.get("runs")
    if raw_runs is None:
        first_key = next(iter(files))
        runs = [{"name": first_key, "file": first_key}]
    else:
        if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
            raise ValueError("'runs' must be a sequence of mappings.")
        runs = []
        for index, item in enumerate(raw_runs):
            if not isinstance(item, Mapping):
                raise ValueError(f"Run entry {index} must be a mapping.")
            run = dict(item)
            if "name" not in run:
                raise ValueError(f"Run entry {index} is missing required key 'name'.")
            runs.append(run)
    if run_names is None:
        return runs
    requested = set(run_names)
    selected = [run for run in runs if str(run["name"]) in requested]
    missing = requested.difference(str(run["name"]) for run in selected)
    if missing:
        raise ValueError(f"Requested run(s) not found in dataset config: {', '.join(sorted(missing))}.")
    return selected


def _materialize_file_path(value: Any, *, participant: int, root: str, config_dir: Path | None) -> str:
    if isinstance(value, Mapping):
        pattern = _string(value.get("pattern", value.get("path")))
    else:
        pattern = _string(value)
    if not pattern:
        raise ValueError("File pattern must be a non-empty string.")
    rendered = pattern.format(participant=participant)
    path = rendered if _is_absolute_path(rendered) or not root else _join_path(root, rendered)
    if config_dir is not None and not _is_absolute_path(path):
        path = str(config_dir / path)
    return path


def _is_absolute_path(path: str) -> bool:
    return path.startswith(("/", "~")) or bool(_DRIVE_RE.match(path))


def _join_path(root: str, child: str) -> str:
    if root.endswith(("/", "\\")):
        return f"{root}{child}"
    return f"{root}/{child}"


def _common_manifest_columns(config: Mapping[str, Any]) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for section_name in ("metadata", "preprocessing", "decoding", "fieldtrip", "benchmark"):
        section = config.get(section_name, {})
        if section is None:
            continue
        if not isinstance(section, Mapping):
            raise ValueError(f"'{section_name}' must be a mapping if present.")
        columns.update(_manifest_columns_from_mapping(section, prefix=section_name))
    return columns


def _manifest_columns_from_mapping(section: Mapping[str, Any], *, prefix: str | None) -> dict[str, Any]:
    columns: dict[str, Any] = {}
    for key, value in section.items():
        if value is None:
            continue
        column = _manifest_column_name(key, prefix=prefix)
        columns[column] = _manifest_value(value)
    return columns


def _manifest_column_name(key: str, *, prefix: str | None) -> str:
    if prefix == "metadata":
        aliases = {
            "label": "label_column",
            "labels": "label_column",
            "label_column": "label_column",
            "group": "group_column",
            "group_column": "group_column",
            "events_csv": "events_csv",
            "source_column": "source_column",
            "positive_pattern": "positive_pattern",
            "negative_pattern": "negative_pattern",
            "positive_label": "positive_label",
            "negative_label": "negative_label",
            "case_sensitive": "case_sensitive",
        }
        return aliases.get(key, key)
    if prefix == "preprocessing":
        aliases = {
            "picks": "picks",
            "tmin": "tmin",
            "tmax": "tmax",
            "window_ms": "window_ms",
            "step_ms": "step_ms",
            "normalization": "normalization",
            "baseline_window": "baseline_window",
            "baseline_window_start": "baseline_window_start",
            "baseline_window_stop": "baseline_window_stop",
        }
        return aliases.get(key, key)
    if prefix == "decoding":
        aliases = {
            "decoder": "decoder",
            "emission_mode": "emission_mode",
            "feature_preprocessor": "feature_preprocessor",
            "pca_components": "pca_components",
            "tune_hyperparameters": "tune_hyperparameters",
            "tuning_cv_splits": "tuning_cv_splits",
            "tuning_scoring": "tuning_scoring",
            "tuning_c_grid": "tuning_c_grid",
            "temporal_train_window": "temporal_train_window",
            "temporal_train_window_start": "temporal_train_window_start",
            "temporal_train_window_stop": "temporal_train_window_stop",
            "n_splits": "n_splits",
            "max_iter": "max_iter",
        }
        return aliases.get(key, key)
    if prefix == "fieldtrip":
        return f"fieldtrip_{key}"
    return key


def _manifest_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a NeuRepTrace benchmark CSV manifest from a dataset YAML/JSON config.")
    parser.add_argument("config", type=Path, help="Dataset config path (.yml, .yaml, or .json).")
    parser.add_argument("--out", type=Path, required=True, help="Output benchmark manifest CSV.")
    parser.add_argument("--run", action="append", dest="runs", help="Only emit the named run. Repeat to select multiple runs.")
    parser.add_argument("--absolute-paths", action="store_true", help="Resolve relative generated paths against the config file directory.")
    args = parser.parse_args()

    frame = write_manifest_from_dataset_config(
        args.config,
        args.out,
        run_names=tuple(args.runs) if args.runs else None,
        absolute_paths=args.absolute_paths,
    )
    print(f"Wrote {len(frame)} row(s) to {args.out}")


if __name__ == "__main__":
    main()

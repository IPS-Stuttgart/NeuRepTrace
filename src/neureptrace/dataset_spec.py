"""Declarative dataset specifications for NeuRepTrace loaders.

Dataset specs move file naming, participant selection, split definitions, and
metadata mapping into versioned YAML or JSON while keeping scientific behavior
in ordinary Python code.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SUPPORTED_SCHEMA_VERSION = "neureptrace.dataset.v1"
SUPPORTED_LOADERS = {"mne_epochs", "matlab_fieldtrip", "csv_feature_matrix"}
_SUBJECT_RANGE_RE = re.compile(r"^(-?\d+)\s*-\s*(-?\d+)$")


@dataclass(frozen=True)
class DatasetRoot:
    """Root-resolution rules for a dataset."""

    path: str | None = None
    env: str | None = None
    fallback_file: str | None = None


@dataclass(frozen=True)
class LabelSpec:
    """Dataset-wide label semantics."""

    column: str | None = None
    chance_classes: int | None = None
    index_base: int = 0
    subtract_one_when_no_null_class: bool = False


@dataclass(frozen=True)
class PreprocessingSpec:
    """Preprocessing defaults that workflows may opt into."""

    frequency_range_hz: tuple[float, float] | None = None
    window_size_s: float | None = None
    train_window_center_s: float | None = None
    null_window_center_s: float | None = None
    resample_hz: float | None = None
    pca_components: int | float | None = None
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitSpec:
    """One named file convention, such as ``main`` or ``cue``."""

    name: str
    loader: str
    path_template: str
    metadata_template: str | None = None
    mat_key: str = "data"
    trial_key: str = "trial"
    time_key: str = "time"
    channel_key: str | None = "label"
    label_key: str | None = "trialinfo"
    label_column: str | None = None
    group_column: str | None = None
    label_index_base: int | None = None
    trial_layout: str = "channels_by_time"
    manifest: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkflowSpec:
    """Workflow defaults that can be merged into generated manifests."""

    name: str
    split: str | None = None
    manifest: Mapping[str, Any] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetSpec:
    """Validated declarative dataset description."""

    dataset_id: str
    schema_version: str = SUPPORTED_SCHEMA_VERSION
    description: str = ""
    root: DatasetRoot = field(default_factory=DatasetRoot)
    subjects: tuple[str, ...] = ()
    splits: Mapping[str, SplitSpec] = field(default_factory=dict)
    labels: LabelSpec = field(default_factory=LabelSpec)
    preprocessing_defaults: PreprocessingSpec = field(default_factory=PreprocessingSpec)
    workflows: Mapping[str, WorkflowSpec] = field(default_factory=dict)
    outputs: Mapping[str, Any] = field(default_factory=dict)
    source_path: Path | None = None


@dataclass(frozen=True)
class ResolvedSplit:
    """Concrete files resolved for one subject and split."""

    dataset_id: str
    subject: str
    split: str
    loader: str
    data_path: Path
    metadata_path: Path | None = None
    label_column: str | None = None
    group_column: str | None = None
    manifest: Mapping[str, Any] = field(default_factory=dict)

    @property
    def data_exists(self) -> bool:
        """Return whether the resolved data file exists."""

        return self.data_path.is_file()

    @property
    def metadata_exists(self) -> bool:
        """Return whether metadata is absent by design or exists."""

        return self.metadata_path is None or self.metadata_path.is_file()

    def to_inventory_row(self) -> dict[str, Any]:
        """Return a CSV-friendly validation row."""

        return {
            "dataset_id": self.dataset_id,
            "subject": self.subject,
            "split": self.split,
            "loader": self.loader,
            "data_path": str(self.data_path),
            "data_exists": self.data_exists,
            "metadata_path": "" if self.metadata_path is None else str(self.metadata_path),
            "metadata_exists": self.metadata_exists,
            "label_column": "" if self.label_column is None else self.label_column,
            "group_column": "" if self.group_column is None else self.group_column,
        }

    def to_manifest_row(self) -> dict[str, Any]:
        """Return a benchmark-style manifest row."""

        row: dict[str, Any] = {"subject": self.subject, "loader": self.loader, "split": self.split}
        if self.loader == "mne_epochs":
            row["epochs"] = str(self.data_path)
        else:
            row["data_path"] = str(self.data_path)
        if self.metadata_path is not None:
            row["metadata_csv"] = str(self.metadata_path)
        if self.label_column is not None:
            row["label_column"] = self.label_column
        if self.group_column is not None:
            row["group_column"] = self.group_column
        row.update(dict(self.manifest))
        return row


@dataclass(frozen=True)
class TrialDataset:
    """Canonical trial array emitted by non-MNE loaders."""

    data: np.ndarray
    times: np.ndarray
    labels: np.ndarray | None
    metadata: pd.DataFrame | None
    channels: tuple[str, ...]
    subject: str
    split: str
    source_path: Path


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    """Load a YAML or JSON dataset spec from disk."""

    spec_path = Path(path)
    payload = _load_mapping_file(spec_path)
    return dataset_spec_from_mapping(payload, source_path=spec_path)


def dataset_spec_from_mapping(payload: Mapping[str, Any], *, source_path: str | Path | None = None) -> DatasetSpec:
    """Validate a mapping and return a :class:`DatasetSpec`."""

    mapping = _as_mapping(payload, "dataset spec")
    schema_version = str(mapping.get("schema_version", SUPPORTED_SCHEMA_VERSION))
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise ValueError(f"Unsupported dataset schema_version {schema_version!r}; expected {SUPPORTED_SCHEMA_VERSION!r}.")
    return DatasetSpec(
        dataset_id=_required_str(mapping, "dataset_id"),
        schema_version=schema_version,
        description=str(mapping.get("description", "")),
        root=_parse_root(_optional_mapping(mapping, "root")),
        subjects=parse_subjects(mapping.get("subjects", ())),
        splits=_parse_splits(_required_mapping(mapping, "splits")),
        labels=_parse_labels(_optional_mapping(mapping, "labels")),
        preprocessing_defaults=_parse_preprocessing(_optional_mapping(mapping, "preprocessing_defaults")),
        workflows=_parse_workflows(_optional_mapping(mapping, "workflows")),
        outputs=dict(_optional_mapping(mapping, "outputs")),
        source_path=None if source_path is None else Path(source_path),
    )


def parse_subjects(value: Any) -> tuple[str, ...]:
    """Parse subject specs such as ``"1-4,6,8"`` or ``["sub-01"]``."""

    if value is None or value == "":
        return ()
    if isinstance(value, Mapping):
        included = parse_subjects(value.get("include", ()))
        excluded = set(parse_subjects(value.get("exclude", ())))
        return tuple(subject for subject in included if subject not in excluded)
    if isinstance(value, int):
        return (str(value),)
    if isinstance(value, str):
        subjects: list[str] = []
        for token in value.replace(";", ",").split(","):
            token = token.strip()
            if token:
                subjects.extend(_expand_subject_token(token))
        return tuple(_deduplicate(subjects))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        subjects = []
        for item in value:
            subjects.extend(parse_subjects(item))
        return tuple(_deduplicate(subjects))
    raise TypeError(f"subjects must be a string, integer, sequence, or mapping; got {type(value).__name__}.")


def resolve_dataset_root(spec: DatasetSpec) -> Path:
    """Resolve the root directory from path, environment variable, or fallback file."""

    spec_dir = Path.cwd() if spec.source_path is None else spec.source_path.parent
    if spec.root.path:
        return _resolve_relative_path(spec.root.path, spec_dir)
    if spec.root.env and os.environ.get(spec.root.env):
        return Path(os.environ[spec.root.env]).expanduser().resolve()
    if spec.root.fallback_file:
        fallback = _resolve_relative_path(spec.root.fallback_file, spec_dir)
        if fallback.is_file():
            target = fallback.read_text(encoding="utf-8").strip()
            if target:
                return Path(target).expanduser().resolve()
    return spec_dir.resolve()


def resolve_split(spec: DatasetSpec, split_name: str, subject: str | int, *, root: str | Path | None = None) -> ResolvedSplit:
    """Resolve one split for one subject."""

    if split_name not in spec.splits:
        raise KeyError(f"Unknown split {split_name!r}; available splits: {', '.join(sorted(spec.splits))}.")
    split = spec.splits[split_name]
    root_path = Path(root).expanduser().resolve() if root is not None else resolve_dataset_root(spec)
    format_values = {"dataset_id": spec.dataset_id, "split": split.name, **_subject_format_values(subject)}
    metadata_path = None
    if split.metadata_template:
        metadata_path = _resolve_relative_path(_format_template(split.metadata_template, format_values), root_path)
    return ResolvedSplit(
        dataset_id=spec.dataset_id,
        subject=str(subject),
        split=split.name,
        loader=split.loader,
        data_path=_resolve_relative_path(_format_template(split.path_template, format_values), root_path),
        metadata_path=metadata_path,
        label_column=split.label_column or spec.labels.column,
        group_column=split.group_column,
        manifest=split.manifest,
    )


def iter_resolved_splits(
    spec: DatasetSpec,
    *,
    subjects: Iterable[str | int] | None = None,
    splits: Iterable[str] | None = None,
    root: str | Path | None = None,
) -> list[ResolvedSplit]:
    """Resolve all requested subject and split combinations."""

    subject_values = tuple(str(subject) for subject in (subjects if subjects is not None else spec.subjects))
    split_values = tuple(splits if splits is not None else spec.splits.keys())
    return [resolve_split(spec, split_name, subject, root=root) for subject in subject_values for split_name in split_values]


def validate_dataset_spec(
    spec: DatasetSpec,
    *,
    subjects: Iterable[str | int] | None = None,
    splits: Iterable[str] | None = None,
    require_files: bool = False,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Return an inventory table and optionally fail when files are missing."""

    inventory = pd.DataFrame([item.to_inventory_row() for item in iter_resolved_splits(spec, subjects=subjects, splits=splits, root=root)])
    if require_files and not inventory.empty:
        missing = inventory.loc[~inventory["data_exists"] | ~inventory["metadata_exists"]]
        if not missing.empty:
            missing_items = ", ".join(f"{row.subject}:{row.split}" for row in missing.itertuples())
            raise FileNotFoundError(f"Dataset spec resolves missing files: {missing_items}.")
    return inventory


def expand_manifest(
    spec: DatasetSpec,
    *,
    workflow: str | None = None,
    subjects: Iterable[str | int] | None = None,
    split: str | None = None,
    root: str | Path | None = None,
) -> pd.DataFrame:
    """Expand a dataset spec into a benchmark-style manifest table."""

    workflow_spec = spec.workflows.get(workflow) if workflow else None
    selected_split = split or (workflow_spec.split if workflow_spec is not None else None)
    split_names = (selected_split,) if selected_split is not None else tuple(spec.splits)
    rows: list[dict[str, Any]] = []
    for item in iter_resolved_splits(spec, subjects=subjects, splits=split_names, root=root):
        row = item.to_manifest_row()
        if workflow_spec is not None:
            row.update(dict(workflow_spec.manifest))
        rows.append(row)
    return pd.DataFrame(rows)


def load_split_dataset(spec: DatasetSpec, split_name: str, subject: str | int, *, root: str | Path | None = None) -> TrialDataset | ResolvedSplit:
    """Load a configured split.

    MNE splits return a :class:`ResolvedSplit` because existing MNE workflows
    read the epochs file directly. MATLAB and CSV splits are loaded into a
    canonical :class:`TrialDataset`.
    """

    resolved = resolve_split(spec, split_name, subject, root=root)
    split = spec.splits[split_name]
    if split.loader == "mne_epochs":
        return resolved
    if split.loader == "matlab_fieldtrip":
        return _load_matlab_fieldtrip(resolved, split, spec.labels)
    if split.loader == "csv_feature_matrix":
        return _load_csv_feature_matrix(resolved)
    raise ValueError(f"Unsupported loader {split.loader!r}.")


def _load_mapping_file(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset spec not found: {path}")
    if path.suffix.lower() == ".json":
        return _as_mapping(json.loads(path.read_text(encoding="utf-8")), f"JSON file {path}")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("YAML dataset specs require PyYAML. Install PyYAML or use JSON.") from exc
        return _as_mapping(yaml.safe_load(path.read_text(encoding="utf-8")), f"YAML file {path}")
    raise ValueError(f"Unsupported dataset spec suffix {path.suffix!r}; use .json, .yaml, or .yml.")


def _parse_root(mapping: Mapping[str, Any]) -> DatasetRoot:
    return DatasetRoot(path=_optional_str(mapping, "path"), env=_optional_str(mapping, "env"), fallback_file=_optional_str(mapping, "fallback_file"))


def _parse_labels(mapping: Mapping[str, Any]) -> LabelSpec:
    return LabelSpec(
        column=_optional_str(mapping, "column"),
        chance_classes=_optional_int(mapping, "chance_classes"),
        index_base=int(mapping.get("index_base", 0)),
        subtract_one_when_no_null_class=bool(mapping.get("subtract_one_when_no_null_class", False)),
    )


def _parse_preprocessing(mapping: Mapping[str, Any]) -> PreprocessingSpec:
    known = {"frequency_range_hz", "window_size_s", "train_window_center_s", "null_window_center_s", "resample_hz", "pca_components"}
    frequency_range = mapping.get("frequency_range_hz")
    if frequency_range is not None:
        frequency_range = _two_float_tuple(frequency_range, "preprocessing_defaults.frequency_range_hz")
    return PreprocessingSpec(
        frequency_range_hz=frequency_range,
        window_size_s=_optional_float(mapping, "window_size_s"),
        train_window_center_s=_optional_float(mapping, "train_window_center_s"),
        null_window_center_s=_optional_float(mapping, "null_window_center_s"),
        resample_hz=_optional_float(mapping, "resample_hz"),
        pca_components=mapping.get("pca_components"),
        extras={key: value for key, value in mapping.items() if key not in known},
    )


def _parse_splits(mapping: Mapping[str, Any]) -> Mapping[str, SplitSpec]:
    splits: dict[str, SplitSpec] = {}
    for name, value in mapping.items():
        split_mapping = _as_mapping(value, f"split {name}")
        loader = _required_str(split_mapping, "loader")
        if loader not in SUPPORTED_LOADERS:
            raise ValueError(f"Split {name!r} uses unsupported loader {loader!r}; supported loaders: {', '.join(sorted(SUPPORTED_LOADERS))}.")
        known = {
            "loader",
            "path_template",
            "metadata_template",
            "meta_template",
            "mat_key",
            "trial_key",
            "time_key",
            "channel_key",
            "label_key",
            "label_column",
            "group_column",
            "label_index_base",
            "trial_layout",
            "manifest",
        }
        splits[str(name)] = SplitSpec(
            name=str(name),
            loader=loader,
            path_template=_required_str(split_mapping, "path_template"),
            metadata_template=_optional_str(split_mapping, "metadata_template") or _optional_str(split_mapping, "meta_template"),
            mat_key=str(split_mapping.get("mat_key", "data")),
            trial_key=str(split_mapping.get("trial_key", "trial")),
            time_key=str(split_mapping.get("time_key", "time")),
            channel_key=_optional_str(split_mapping, "channel_key") if "channel_key" in split_mapping else "label",
            label_key=_optional_str(split_mapping, "label_key") if "label_key" in split_mapping else "trialinfo",
            label_column=_optional_str(split_mapping, "label_column"),
            group_column=_optional_str(split_mapping, "group_column"),
            label_index_base=_optional_int(split_mapping, "label_index_base"),
            trial_layout=str(split_mapping.get("trial_layout", "channels_by_time")),
            manifest=dict(_optional_mapping(split_mapping, "manifest")),
            extras={key: item for key, item in split_mapping.items() if key not in known},
        )
    if not splits:
        raise ValueError("Dataset spec must define at least one split.")
    return splits


def _parse_workflows(mapping: Mapping[str, Any]) -> Mapping[str, WorkflowSpec]:
    workflows: dict[str, WorkflowSpec] = {}
    for name, value in mapping.items():
        workflow_mapping = _as_mapping(value, f"workflow {name}")
        known = {"split", "manifest"}
        workflows[str(name)] = WorkflowSpec(
            name=str(name),
            split=_optional_str(workflow_mapping, "split"),
            manifest=dict(_optional_mapping(workflow_mapping, "manifest")),
            extras={key: item for key, item in workflow_mapping.items() if key not in known},
        )
    return workflows


def _load_matlab_fieldtrip(resolved: ResolvedSplit, split: SplitSpec, labels: LabelSpec) -> TrialDataset:
    try:
        from scipy.io import loadmat
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("The matlab_fieldtrip loader requires scipy.") from exc
    if not resolved.data_path.is_file():
        raise FileNotFoundError(f"MATLAB data file not found: {resolved.data_path}")
    payload = loadmat(resolved.data_path, squeeze_me=True, struct_as_record=False)
    if split.mat_key not in payload:
        raise KeyError(f"MATLAB file {resolved.data_path} does not contain key {split.mat_key!r}.")
    container = _unwrap_mat_object(payload[split.mat_key])
    data = _stack_trials(_mat_field(container, split.trial_key), trial_layout=split.trial_layout)
    times = _first_time_axis(_mat_field(container, split.time_key))
    labels_array = None if split.label_key is None else _normalize_labels(_optional_mat_field(container, split.label_key), split, labels)
    channels = () if split.channel_key is None else _string_tuple(_optional_mat_field(container, split.channel_key))
    metadata = pd.read_csv(resolved.metadata_path) if resolved.metadata_path is not None and resolved.metadata_path.is_file() else None
    return TrialDataset(
        data=data,
        times=times,
        labels=labels_array,
        metadata=metadata,
        channels=channels,
        subject=resolved.subject,
        split=resolved.split,
        source_path=resolved.data_path,
    )


def _load_csv_feature_matrix(resolved: ResolvedSplit) -> TrialDataset:
    if not resolved.data_path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {resolved.data_path}")
    frame = pd.read_csv(resolved.data_path)
    metadata = pd.read_csv(resolved.metadata_path) if resolved.metadata_path is not None and resolved.metadata_path.is_file() else None
    labels = None
    feature_frame = frame
    if resolved.label_column is not None and resolved.label_column in frame.columns:
        labels = frame[resolved.label_column].to_numpy()
        feature_frame = frame.drop(columns=[resolved.label_column])
    numeric = feature_frame.select_dtypes(include=[np.number])
    data = numeric.to_numpy(dtype=float)[:, :, np.newaxis]
    return TrialDataset(
        data=data,
        times=np.array([0.0]),
        labels=labels,
        metadata=metadata,
        channels=tuple(numeric.columns),
        subject=resolved.subject,
        split=resolved.split,
        source_path=resolved.data_path,
    )


def _expand_subject_token(token: str) -> tuple[str, ...]:
    match = _SUBJECT_RANGE_RE.match(token)
    if match is None:
        return (token,)
    start, stop = int(match.group(1)), int(match.group(2))
    step = 1 if stop >= start else -1
    return tuple(str(value) for value in range(start, stop + step, step))


def _deduplicate(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _subject_format_values(subject: str | int) -> dict[str, Any]:
    text = str(subject)
    values: dict[str, Any] = {"subject": text, "subject_id": text, "participant": text, "participant_id": text}
    try:
        numeric = int(text)
    except ValueError:
        return values
    values.update({"subject_int": numeric, "participant_int": numeric, "subject02d": f"{numeric:02d}", "participant02d": f"{numeric:02d}"})
    return values


def _format_template(template: str, values: Mapping[str, Any]) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        available = ", ".join(sorted(values))
        raise KeyError(f"Unknown path-template placeholder {exc.args[0]!r}; available placeholders: {available}.") from exc


def _resolve_relative_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _unwrap_mat_object(value: Any) -> Any:
    current = value
    while isinstance(current, np.ndarray) and current.shape == ():
        current = current.item()
    if isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
        return _unwrap_mat_object(current.reshape(-1)[0])
    return current


def _mat_field(container: Any, key: str) -> Any:
    value = _optional_mat_field(container, key)
    if value is None:
        raise KeyError(f"MATLAB data object does not contain field {key!r}.")
    return value


def _optional_mat_field(container: Any, key: str) -> Any:
    if isinstance(container, Mapping):
        return container.get(key)
    if hasattr(container, key):
        return getattr(container, key)
    if isinstance(container, np.ndarray) and container.dtype.names and key in container.dtype.names:
        return container[key]
    return None


def _sequence_from_mat_value(value: Any) -> list[Any]:
    value = _unwrap_mat_object(value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            try:
                numeric = value.astype(float)
            except (TypeError, ValueError):
                return [_unwrap_mat_object(item) for item in value.reshape(-1)]
            if numeric.ndim >= 2:
                return [numeric]
            return [item for item in numeric.reshape(-1)]
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _stack_trials(trials: Any, *, trial_layout: str) -> np.ndarray:
    unwrapped = _unwrap_mat_object(trials)
    if isinstance(unwrapped, np.ndarray) and unwrapped.ndim == 3:
        data = np.asarray(unwrapped, dtype=float)
    else:
        items = _sequence_from_mat_value(unwrapped)
        if len(items) == 1 and np.asarray(items[0]).ndim == 3:
            data = np.asarray(items[0], dtype=float)
        else:
            arrays = [np.asarray(_unwrap_mat_object(item), dtype=float) for item in items]
            if trial_layout not in {"channels_by_time", "time_by_channels"}:
                raise ValueError("trial_layout must be 'channels_by_time' or 'time_by_channels'.")
            if trial_layout == "time_by_channels":
                arrays = [array.T for array in arrays]
            data = np.stack(arrays, axis=0)
    if data.ndim != 3:
        raise ValueError(f"Expected trial data with shape trials x channels x time, got {data.shape}.")
    return data


def _first_time_axis(times: Any) -> np.ndarray:
    items = _sequence_from_mat_value(times)
    time_axis = np.asarray(_unwrap_mat_object(items[0]), dtype=float)
    if time_axis.ndim > 1:
        time_axis = time_axis.reshape(time_axis.shape[0], -1)[0]
    else:
        time_axis = time_axis.reshape(-1)
    if time_axis.size == 0:
        raise ValueError("MATLAB time axis is empty.")
    return time_axis


def _normalize_labels(label_values: Any, split: SplitSpec, labels: LabelSpec) -> np.ndarray | None:
    if label_values is None:
        return None
    label_array = np.asarray(label_values).reshape(-1)
    index_base = split.label_index_base if split.label_index_base is not None else labels.index_base
    if index_base and np.issubdtype(label_array.dtype, np.number):
        return label_array.astype(int) - int(index_base)
    return label_array


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    items = _sequence_from_mat_value(value)
    if len(items) == 1 and np.asarray(items[0]).ndim > 0:
        items = list(np.asarray(items[0]).reshape(-1))
    return tuple(str(_unwrap_mat_object(item)) for item in items)


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping; got {type(value).__name__}.")
    return value


def _required_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    if key not in mapping:
        raise KeyError(f"Dataset spec is missing required key {key!r}.")
    return _as_mapping(mapping[key], key)


def _optional_mapping(mapping: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = mapping.get(key, {})
    return {} if value is None else _as_mapping(value, key)


def _required_str(mapping: Mapping[str, Any], key: str) -> str:
    if key not in mapping or mapping[key] is None or str(mapping[key]).strip() == "":
        raise KeyError(f"Dataset spec is missing required string key {key!r}.")
    return str(mapping[key])


def _optional_str(mapping: Mapping[str, Any], key: str) -> str | None:
    value = mapping.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(mapping: Mapping[str, Any], key: str) -> float | None:
    return None if mapping.get(key) is None else float(mapping[key])


def _optional_int(mapping: Mapping[str, Any], key: str) -> int | None:
    return None if mapping.get(key) is None else int(mapping[key])


def _two_float_tuple(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{name} must contain exactly two numeric values.")
    return float(value[0]), float(value[1])

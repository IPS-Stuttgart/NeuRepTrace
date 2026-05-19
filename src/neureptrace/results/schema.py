"""Canonical result-table schemas for reusable decoding workflows."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

RESULT_TABLE_SCHEMA_VERSION = "1"
SCHEMA_VERSION = RESULT_TABLE_SCHEMA_VERSION
PROBABILITY_COLUMN_PREFIX = "prob_class_"
ColumnKind = Literal["string", "integer", "number", "boolean", "object"]


@dataclass(frozen=True)
class ResultTableSchema:
    """Column contract for a workflow result table.

    The schema is intentionally permissive: project packages may keep additional
    columns, but the canonical columns listed here give downstream reporting code
    stable names for common decoding outputs.
    """

    name: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    at_least_one_of: tuple[tuple[str, ...], ...] = ()
    aliases: Mapping[str, str] = field(default_factory=dict)
    description: str = ""

    @property
    def canonical_columns(self) -> tuple[str, ...]:
        """Return required and optional canonical columns without duplicates."""
        return tuple(dict.fromkeys((*self.required_columns, *self.optional_columns)))


@dataclass(frozen=True)
class ColumnSpec:
    """Column contract for a typed result table schema."""

    name: str
    kind: ColumnKind = "object"
    required: bool = True
    nullable: bool = False
    description: str = ""


@dataclass(frozen=True)
class TableSchema:
    """Reusable typed schema for workflow result/provenance tables."""

    name: str
    required_columns: tuple[ColumnSpec, ...]
    optional_columns: tuple[ColumnSpec, ...] = ()
    primary_key: tuple[str, ...] = ()
    description: str = ""

    @property
    def columns(self) -> tuple[ColumnSpec, ...]:
        return (*self.required_columns, *self.optional_columns)

    @property
    def required_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.required_columns)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns)


_COMMON_ALIASES = {
    "participant": "subject",
    "held_out_participant": "subject",
    "heldout_participant": "subject",
    "held_out_subject": "subject",
    "test_subject": "subject",
    "test_participant": "subject",
    "outer_test_participant": "subject",
    "stimulus": "label",
    "stimulus_id": "label",
    "true_stimulus": "true_label",
    "true_stimulus_id": "true_label",
    "predicted_stimulus": "predicted_label",
    "predicted_stimulus_id": "predicted_label",
    "candidate_index": "candidate_id",
    "selected_candidate_index": "candidate_id",
    "fold": "outer_fold",
    "count": "n",
    "window_center_s": "window_center",
    "window_size_s": "window_size",
    "components_pca": "pca_components",
    "n_test_trials": "n_test",
    "participants_total": "n_subjects",
    "n_test_participants": "n_subjects",
}

RESULT_TABLE_SCHEMAS: dict[str, ResultTableSchema] = {
    "outer_fold_scores": ResultTableSchema(
        name="outer_fold_scores",
        required_columns=("outer_fold", "subject"),
        at_least_one_of=(("score", "accuracy", "balanced_accuracy"),),
        optional_columns=(
            "workflow",
            "candidate_id",
            "metric",
            "score",
            "accuracy",
            "accuracy_mean",
            "balanced_accuracy",
            "balanced_accuracy_mean",
            "chance_accuracy",
            "score_minus_chance",
            "n_test_trials",
            "n_test",
            "p_value",
            "window_center",
            "window_size",
            "feature_mode",
            "normalization",
            "alignment",
            "classifier",
            "pca_components",
        ),
        aliases=_COMMON_ALIASES,
        description="Untouched outer-fold scores, typically one row per held-out subject.",
    ),
    "inner_candidate_scores": ResultTableSchema(
        name="inner_candidate_scores",
        required_columns=("outer_fold", "inner_fold", "candidate_id"),
        at_least_one_of=(("score", "accuracy", "balanced_accuracy", "mean_score"),),
        optional_columns=(
            "workflow",
            "subject",
            "metric",
            "score",
            "accuracy",
            "balanced_accuracy",
            "mean_score",
            "n_validation",
            "window_center",
            "window_size",
            "feature_mode",
            "normalization",
            "alignment",
            "classifier",
            "pca_components",
        ),
        aliases={**_COMMON_ALIASES, "inner_subject": "subject", "validation_subject": "subject"},
        description="Inner validation-fold scores for every candidate considered by nested selection.",
    ),
    "selected_candidates": ResultTableSchema(
        name="selected_candidates",
        required_columns=("outer_fold", "candidate_id"),
        optional_columns=(
            "workflow",
            "subject",
            "selection_metric",
            "selected_score",
            "runner_up_score",
            "winner_margin",
            "window_center",
            "window_size",
            "feature_mode",
            "normalization",
            "alignment",
            "classifier",
            "classifier_params",
            "pca_components",
        ),
        aliases=_COMMON_ALIASES,
        description="Selected hyperparameters/candidates for each outer fold.",
    ),
    "predictions": ResultTableSchema(
        name="predictions",
        required_columns=("subject", "true_label", "predicted_label"),
        optional_columns=(
            "workflow",
            "outer_fold",
            "trial",
            "time",
            "candidate_id",
            "score",
            "probability",
            "correct",
            "window_center",
            "window_size",
        ),
        aliases=_COMMON_ALIASES,
        description="Trial-level predictions from held-out data.",
    ),
    "confusion": ResultTableSchema(
        name="confusion",
        required_columns=("true_label", "predicted_label", "n"),
        optional_columns=("workflow", "subject", "outer_fold", "fraction"),
        aliases={key: value for key, value in _COMMON_ALIASES.items() if key not in {"count"}} | {"count": "n", "n_predictions": "n"},
        description="Confusion counts grouped by true and predicted labels.",
    ),
    "per_class": ResultTableSchema(
        name="per_class",
        required_columns=("label",),
        at_least_one_of=(("recall", "accuracy"),),
        optional_columns=("workflow", "subject", "outer_fold", "recall", "accuracy", "support", "chance_accuracy"),
        aliases={**_COMMON_ALIASES, "class": "label", "class_label": "label", "stimulus_recall": "recall"},
        description="Per-label recall or accuracy summaries.",
    ),
    "group_summary": ResultTableSchema(
        name="group_summary",
        required_columns=(),
        at_least_one_of=(("metric", "value", "mean", "score", "accuracy", "accuracy_mean", "balanced_accuracy", "balanced_accuracy_mean"),),
        optional_columns=(
            "workflow",
            "metric",
            "value",
            "mean",
            "score",
            "accuracy",
            "accuracy_mean",
            "balanced_accuracy",
            "balanced_accuracy_mean",
            "std",
            "sem",
            "median",
            "n_subjects",
            "chance_accuracy",
            "p_value",
            "sign_test_p_value",
        ),
        aliases={"p": "p_value", "n_participants": "n_subjects", "participants_total": "n_subjects", "n_test_participants": "n_subjects"},
        description="Workflow-level summaries intended for reports and CI artifacts.",
    ),
    "provenance": ResultTableSchema(
        name="provenance",
        required_columns=("workflow",),
        optional_columns=(
            "schema_version",
            "created_at",
            "source_files",
            "n_outer_folds",
            "n_subjects",
            "n_predictions",
            "n_candidates",
            "n_selected_candidates",
        ),
        aliases={"workflow_name": "workflow"},
        description="One-row workflow provenance and compact configuration metadata.",
    ),
    "manifest": ResultTableSchema(
        name="manifest",
        required_columns=("table", "path", "n_rows", "n_columns", "schema_version"),
        optional_columns=("columns",),
        description="Manifest produced when a bundle of result tables is written.",
    ),
}


CONDITION_COLUMNS = (
    "decoder",
    "emission_mode",
    "feature_preprocessor",
    "pca_components",
    "tuned_hyperparameters",
    "tuning_cv_splits",
    "tuning_scoring",
    "tuning_c_grid",
    "temporal_mode",
    "temporal_train_window_start",
    "temporal_train_window_stop",
    "temporal_smoothing_method",
    "temporal_smoothing_fit_window_start",
    "temporal_smoothing_fit_window_stop",
)


def _column(
    name: str,
    kind: ColumnKind = "object",
    *,
    required: bool = True,
    nullable: bool = False,
    description: str = "",
) -> ColumnSpec:
    return ColumnSpec(name=name, kind=kind, required=required, nullable=nullable, description=description)


def _optional(
    name: str,
    kind: ColumnKind = "object",
    *,
    nullable: bool = True,
    description: str = "",
) -> ColumnSpec:
    return _column(name, kind, required=False, nullable=nullable, description=description)


_COMMON_CONDITION_SPECS = tuple(
    _optional(column, "boolean" if column == "tuned_hyperparameters" else "string", nullable=True)
    for column in CONDITION_COLUMNS
)
_METRIC_SPECS = (
    _column("accuracy", "number"),
    _column("log_loss", "number"),
    _column("brier", "number"),
    _column("ece", "number"),
)
_METRIC_SUMMARY_SPECS = (
    _column("accuracy_mean", "number"),
    _column("log_loss_mean", "number"),
    _column("brier_mean", "number"),
    _column("ece_mean", "number"),
)
_METRIC_SEM_SPECS = tuple(_optional(f"{metric}_sem", "number") for metric in ("accuracy", "log_loss", "brier", "ece"))

OUTER_FOLD_SCORE_SCHEMA = TableSchema(
    name="outer_fold_scores",
    description="Fold-level time-resolved decoding scores before subject/group aggregation.",
    required_columns=(
        _column("subject", "string"),
        _column("time", "number"),
        *_METRIC_SPECS,
    ),
    optional_columns=(
        _optional("fold", "object"),
        _optional("n_test", "integer"),
        _optional("best_params", "string"),
        _optional("best_score", "number"),
        _optional("source_file", "string"),
        *_COMMON_CONDITION_SPECS,
    ),
)
TIME_DECODE_SUMMARY_SCHEMA = TableSchema(
    name="time_decode_summary",
    description="Subject-aggregated time-resolved decoding summary.",
    required_columns=(
        _column("time", "number"),
        _column("n_subjects", "integer"),
        *_METRIC_SUMMARY_SPECS,
    ),
    optional_columns=(*_METRIC_SEM_SPECS, *_COMMON_CONDITION_SPECS),
)
PROBABILITY_OBSERVATION_SCHEMA = TableSchema(
    name="probability_observations",
    description="Held-out probability rows used for exact probability metrics such as ECE.",
    required_columns=(
        _column("time", "number"),
        _column("true_label", "integer"),
    ),
    optional_columns=(
        _optional("subject", "string"),
        _optional("fold", "object"),
        _optional("trial", "object"),
        _optional("predicted_label", "integer"),
        _optional("source_file", "string"),
        *_COMMON_CONDITION_SPECS,
    ),
)
PREDICTION_SCHEMA = TableSchema(
    name="predictions",
    description="Held-out prediction rows independent of probability-column layout.",
    required_columns=(
        _column("subject", "string"),
        _column("true_label", "object"),
        _column("predicted_label", "object"),
    ),
    optional_columns=(
        _optional("fold", "object"),
        _optional("trial", "object"),
        _optional("time", "number"),
        _optional("score", "number"),
        _optional("source_file", "string"),
        *_COMMON_CONDITION_SPECS,
    ),
)
CONFUSION_SCHEMA = TableSchema(
    name="confusion",
    description="Pairwise true/predicted-label confusion counts.",
    required_columns=(
        _column("true_label", "object"),
        _column("predicted_label", "object"),
        _column("n", "integer"),
    ),
    optional_columns=(
        _optional("subject", "string"),
        _optional("fold", "object"),
        *_COMMON_CONDITION_SPECS,
    ),
)
PER_CLASS_SCHEMA = TableSchema(
    name="per_class",
    description="Per-class or per-stimulus decoding summary.",
    required_columns=(
        _column("label", "object"),
        _column("n_test", "integer"),
        _column("accuracy", "number"),
    ),
    optional_columns=(
        _optional("subject", "string"),
        _optional("fold", "object"),
        _optional("chance_accuracy", "number"),
        *_COMMON_CONDITION_SPECS,
    ),
)
PROVENANCE_SCHEMA = TableSchema(
    name="provenance",
    description="One row per run/condition with selected model parameters, selected time, window metrics, and source files.",
    required_columns=(
        _column("selection_metric", "string"),
        _column("selected_time", "number"),
        _column("selected_accuracy", "number", nullable=True),
        _column("selected_log_loss", "number", nullable=True),
        _column("selected_brier", "number", nullable=True),
        _column("selected_ece", "number", nullable=True),
    ),
    optional_columns=(
        _optional("decoder", "string"),
        _optional("emission_mode", "string"),
        _optional("pca_mode", "string"),
        _optional("pca_components", "string"),
        _optional("tuned_hyperparameters", "boolean"),
        _optional("tuning_cv_splits", "string"),
        _optional("tuning_scoring", "string"),
        _optional("tuning_c_grid", "string"),
        _optional("selected_params", "string"),
        _optional("selected_params_unique", "integer"),
        _optional("best_score_mean", "number"),
        _optional("best_score_min", "number"),
        _optional("best_score_max", "number"),
        _optional("temporal_mode", "string"),
        _optional("temporal_train_window_start", "string"),
        _optional("temporal_train_window_stop", "string"),
        _optional("temporal_smoothing_method", "string"),
        _optional("temporal_smoothing_fit_window_start", "string"),
        _optional("temporal_smoothing_fit_window_stop", "string"),
        _optional("n_subjects", "integer"),
        _optional("baseline_accuracy_mean", "number"),
        _optional("baseline_log_loss_mean", "number"),
        _optional("baseline_brier_mean", "number"),
        _optional("baseline_ece_mean", "number"),
        _optional("effect_accuracy_mean", "number"),
        _optional("effect_log_loss_mean", "number"),
        _optional("effect_brier_mean", "number"),
        _optional("effect_ece_mean", "number"),
        _optional("accuracy_effect_minus_baseline", "number"),
        _optional("log_loss_effect_improvement", "number"),
        _optional("brier_effect_improvement", "number"),
        _optional("ece_effect_improvement", "number"),
        _optional("source_files", "string"),
    ),
)
TYPED_RESULT_TABLE_SCHEMAS = {
    schema.name: schema
    for schema in (
        OUTER_FOLD_SCORE_SCHEMA,
        TIME_DECODE_SUMMARY_SCHEMA,
        PROBABILITY_OBSERVATION_SCHEMA,
        PREDICTION_SCHEMA,
        CONFUSION_SCHEMA,
        PER_CLASS_SCHEMA,
        PROVENANCE_SCHEMA,
    )
}


def available_result_table_schemas() -> tuple[str, ...]:
    """Return registered result-table schema names."""
    return tuple(sorted(RESULT_TABLE_SCHEMAS))


def get_result_table_schema(schema: str | ResultTableSchema) -> ResultTableSchema:
    """Resolve a schema name or pass through an explicit schema object."""
    if isinstance(schema, ResultTableSchema):
        return schema
    try:
        return RESULT_TABLE_SCHEMAS[schema]
    except KeyError as exc:
        known = ", ".join(available_result_table_schemas())
        raise ValueError(f"Unknown result-table schema '{schema}'. Known schemas: {known}") from exc


def normalize_result_table(
    frame: pd.DataFrame,
    schema: str | ResultTableSchema,
    *,
    alias_columns: Mapping[str, str] | None = None,
    allow_extra: bool = True,
) -> pd.DataFrame:
    """Rename known aliases, validate required columns, and order canonical columns first."""
    resolved = get_result_table_schema(schema)
    normalized = frame.copy()
    aliases = {**resolved.aliases, **dict(alias_columns or {})}
    _rename_alias_columns(normalized, aliases)
    validate_result_table(normalized, resolved, allow_extra=allow_extra)

    canonical = [column for column in resolved.canonical_columns if column in normalized.columns]
    extras = [column for column in normalized.columns if column not in canonical]
    return normalized[[*canonical, *extras]]


def validate_result_table(
    frame: pd.DataFrame,
    schema: str | ResultTableSchema | TableSchema,
    *,
    allow_extra: bool = True,
) -> pd.DataFrame:
    """Validate that a frame satisfies a result-table schema."""
    if isinstance(schema, TableSchema):
        return _validate_typed_result_table(frame, schema, allow_extra=allow_extra)

    resolved = get_result_table_schema(schema)
    missing = [column for column in resolved.required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{resolved.name} table is missing required columns: {missing}")

    for choices in resolved.at_least_one_of:
        if not any(column in frame.columns for column in choices):
            joined = ", ".join(choices)
            raise ValueError(f"{resolved.name} table must contain at least one of: {joined}")

    if not allow_extra:
        allowed = set(resolved.canonical_columns)
        extra = [column for column in frame.columns if column not in allowed]
        if extra:
            raise ValueError(f"{resolved.name} table contains non-schema columns: {extra}")
    return frame


def write_result_table(
    frame: pd.DataFrame,
    path: Path,
    schema: str | ResultTableSchema | TableSchema,
    *,
    alias_columns: Mapping[str, str] | None = None,
    allow_extra: bool = True,
    include_optional: bool = False,
) -> pd.DataFrame:
    """Normalize, validate, and write a result table as CSV."""
    if isinstance(schema, TableSchema):
        normalized = canonicalize_result_table(
            frame,
            schema,
            allow_extra=allow_extra,
            include_optional=include_optional,
        )
    else:
        normalized = normalize_result_table(frame, schema, alias_columns=alias_columns, allow_extra=allow_extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(path, index=False)
    return normalized


def schema_as_frame(schema: TableSchema) -> pd.DataFrame:
    """Return a schema documentation table."""
    return pd.DataFrame(
        {
            "table": schema.name,
            "column": [column.name for column in schema.columns],
            "kind": [column.kind for column in schema.columns],
            "required": [column.required for column in schema.columns],
            "nullable": [column.nullable for column in schema.columns],
            "description": [column.description for column in schema.columns],
        }
    )


def canonicalize_result_table(
    frame: pd.DataFrame,
    schema: TableSchema,
    *,
    allow_extra: bool = True,
    include_optional: bool = False,
) -> pd.DataFrame:
    """Validate and order known columns before extra project-specific columns."""
    validate_result_table(frame, schema, allow_extra=allow_extra)

    canonical = frame.copy()
    if include_optional:
        for spec in schema.optional_columns:
            if spec.name not in canonical.columns:
                canonical[spec.name] = pd.NA

    known_order = [column for column in schema.column_names if column in canonical.columns]
    extra_order = [column for column in canonical.columns if column not in known_order]
    return canonical[[*known_order, *extra_order]]


def _null_mask(series: pd.Series, *, treat_blank_as_null: bool) -> pd.Series:
    mask = series.isna()
    if treat_blank_as_null and series.dtype == object:
        mask = mask | series.astype(str).str.strip().eq("")
    return mask


def _bad_examples(series: pd.Series, mask: pd.Series, *, limit: int = 5) -> list[object]:
    return series.loc[mask].head(limit).tolist()


def _validate_number_like(series: pd.Series, spec: ColumnSpec, *, integer: bool) -> None:
    null_mask = _null_mask(series, treat_blank_as_null=True)
    if not spec.nullable and bool(null_mask.any()):
        examples = _bad_examples(series, null_mask)
        raise ValueError(f"Column '{spec.name}' must not contain missing values. Examples: {examples}")

    values = series.loc[~null_mask]
    if values.empty:
        return

    numeric = pd.to_numeric(values, errors="coerce")
    bad = numeric.isna()
    if bool(bad.any()):
        examples = _bad_examples(values, bad)
        raise ValueError(f"Column '{spec.name}' must be numeric. Non-numeric examples: {examples}")

    finite = np.isfinite(numeric.astype(float).to_numpy())
    if not bool(finite.all()):
        examples = values.loc[~finite].head(5).tolist()
        raise ValueError(f"Column '{spec.name}' must contain finite numeric values. Examples: {examples}")

    if integer:
        numeric_float = numeric.astype(float)
        fractional = ~np.isclose(numeric_float, np.round(numeric_float))
        if bool(fractional.any()):
            examples = values.loc[fractional].head(5).tolist()
            raise ValueError(f"Column '{spec.name}' must contain integer values. Examples: {examples}")


def _validate_boolean_like(series: pd.Series, spec: ColumnSpec) -> None:
    null_mask = _null_mask(series, treat_blank_as_null=True)
    if not spec.nullable and bool(null_mask.any()):
        examples = _bad_examples(series, null_mask)
        raise ValueError(f"Column '{spec.name}' must not contain missing values. Examples: {examples}")

    values = series.loc[~null_mask]
    if values.empty:
        return

    if pd.api.types.is_bool_dtype(values):
        return
    if pd.api.types.is_numeric_dtype(values):
        numeric = pd.to_numeric(values, errors="coerce")
        bad = ~numeric.isin([0, 1])
    else:
        bad = ~values.astype(str).str.strip().str.lower().isin({"0", "1", "false", "true", "n", "no", "y", "yes"})
    if bool(bad.any()):
        examples = _bad_examples(values, bad)
        raise ValueError(f"Column '{spec.name}' must contain boolean-like values. Examples: {examples}")


def _validate_column_values(frame: pd.DataFrame, spec: ColumnSpec) -> None:
    series = frame[spec.name]
    if spec.kind == "number":
        _validate_number_like(series, spec, integer=False)
    elif spec.kind == "integer":
        _validate_number_like(series, spec, integer=True)
    elif spec.kind == "boolean":
        _validate_boolean_like(series, spec)
    elif not spec.nullable and bool(series.isna().any()):
        examples = _bad_examples(series, series.isna())
        raise ValueError(f"Column '{spec.name}' must not contain missing values. Examples: {examples}")


def _validate_probability_columns(frame: pd.DataFrame) -> None:
    probability_columns = [column for column in frame.columns if column.startswith(PROBABILITY_COLUMN_PREFIX)]
    if not probability_columns:
        raise ValueError(
            f"Table '{PROBABILITY_OBSERVATION_SCHEMA.name}' must contain at least one "
            f"'{PROBABILITY_COLUMN_PREFIX}*' probability column."
        )
    for column in probability_columns:
        _validate_number_like(frame[column], _column(column, "number"), integer=False)


def _validate_typed_result_table(
    frame: pd.DataFrame,
    schema: TableSchema,
    *,
    allow_extra: bool = True,
    require_primary_key_unique: bool = False,
) -> pd.DataFrame:
    """Validate a result DataFrame against a typed NeuRepTrace table schema."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")

    missing = [column for column in schema.required_names if column not in frame.columns]
    if missing:
        raise ValueError(f"Table '{schema.name}' is missing required columns: {missing}")

    known_columns = set(schema.column_names)
    if not allow_extra:
        unexpected = [column for column in frame.columns if column not in known_columns]
        if unexpected:
            raise ValueError(f"Table '{schema.name}' contains unexpected columns: {unexpected}")

    for spec in schema.columns:
        if spec.name in frame.columns:
            _validate_column_values(frame, spec)

    if schema.name == PROBABILITY_OBSERVATION_SCHEMA.name:
        _validate_probability_columns(frame)

    if require_primary_key_unique and schema.primary_key:
        missing_key_columns = [column for column in schema.primary_key if column not in frame.columns]
        if missing_key_columns:
            raise ValueError(f"Table '{schema.name}' is missing primary-key columns: {missing_key_columns}")
        duplicated = frame.duplicated(list(schema.primary_key), keep=False)
        if bool(duplicated.any()):
            examples = frame.loc[duplicated, list(schema.primary_key)].head(5).to_dict("records")
            raise ValueError(f"Table '{schema.name}' contains duplicate primary-key rows. Examples: {examples}")

    return frame


def write_result_bundle(
    tables: Mapping[str, pd.DataFrame],
    output_dir: Path,
    *,
    prefix: str = "",
    schema_aliases: Mapping[str, Mapping[str, str]] | None = None,
    include_manifest: bool = True,
) -> pd.DataFrame:
    """Write a named collection of canonical result tables and return a manifest."""
    if not tables:
        raise ValueError("At least one result table is required.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    schema_aliases = schema_aliases or {}

    manifest_rows: list[dict[str, object]] = []
    for table_name, frame in tables.items():
        if table_name == "manifest":
            raise ValueError("'manifest' is reserved for the generated bundle manifest.")
        filename = _prefixed_filename(prefix, table_name)
        normalized = write_result_table(
            frame,
            output_dir / filename,
            table_name,
            alias_columns=schema_aliases.get(table_name),
        )
        manifest_rows.append(
            {
                "table": table_name,
                "path": filename,
                "n_rows": int(len(normalized)),
                "n_columns": int(len(normalized.columns)),
                "schema_version": RESULT_TABLE_SCHEMA_VERSION,
                "columns": "|".join(map(str, normalized.columns)),
            }
        )

    manifest = normalize_result_table(pd.DataFrame(manifest_rows), "manifest")
    if include_manifest:
        manifest_path = output_dir / _prefixed_filename(prefix, "manifest")
        manifest.to_csv(manifest_path, index=False)
    return manifest


def build_result_bundle_provenance(
    tables: Mapping[str, pd.DataFrame],
    *,
    workflow: str,
    created_at: datetime | str | None = None,
    source_files: Sequence[str | Path] = (),
    parameters: Mapping[str, object] | None = None,
    software: Mapping[str, object] | None = None,
) -> pd.DataFrame:
    """Build a compact one-row provenance table for a result-table bundle."""
    if not str(workflow).strip():
        raise ValueError("workflow must be a non-empty string")
    created = _format_created_at(created_at)
    row: dict[str, object] = {
        "workflow": workflow,
        "schema_version": RESULT_TABLE_SCHEMA_VERSION,
        "created_at": created,
        "source_files": "|".join(str(Path(path)) for path in source_files),
    }

    for table_name, frame in tables.items():
        row[f"n_{table_name}_rows"] = int(len(frame))

    if "outer_fold_scores" in tables:
        outer = normalize_result_table(tables["outer_fold_scores"], "outer_fold_scores")
        row["n_outer_folds"] = int(outer["outer_fold"].nunique(dropna=True))
        row["n_subjects"] = int(outer["subject"].nunique(dropna=True))
    elif "predictions" in tables:
        predictions = normalize_result_table(tables["predictions"], "predictions")
        row["n_subjects"] = int(predictions["subject"].nunique(dropna=True))

    if "predictions" in tables:
        row["n_predictions"] = int(len(tables["predictions"]))
    if "inner_candidate_scores" in tables:
        inner = normalize_result_table(tables["inner_candidate_scores"], "inner_candidate_scores")
        row["n_candidates"] = int(inner["candidate_id"].nunique(dropna=True))
    if "selected_candidates" in tables:
        selected = normalize_result_table(tables["selected_candidates"], "selected_candidates")
        row["n_selected_candidates"] = int(selected["candidate_id"].nunique(dropna=True))

    for prefix, values in (("param", parameters or {}), ("software", software or {})):
        for key, value in sorted(values.items()):
            row[f"{prefix}_{key}"] = _compact_value(value)

    return normalize_result_table(pd.DataFrame([row]), "provenance")


def _rename_alias_columns(frame: pd.DataFrame, aliases: Mapping[str, str]) -> None:
    rename: dict[str, str] = {}
    for source, target in aliases.items():
        if source == target or source not in frame.columns:
            continue
        if target in frame.columns:
            # Project-specific tables may carry both a canonical zero-based label
            # and a human-facing alias such as a one-based stimulus id. Preserve
            # both rather than overwriting the canonical column.
            continue
        rename[source] = target
    if rename:
        frame.rename(columns=rename, inplace=True)


def _prefixed_filename(prefix: str, table_name: str) -> str:
    clean_prefix = prefix.strip("_-")
    stem = f"{clean_prefix}_{table_name}" if clean_prefix else table_name
    return f"{stem}.csv"


def _format_created_at(value: datetime | str | None) -> str:
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def _compact_value(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)

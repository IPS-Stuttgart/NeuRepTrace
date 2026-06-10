"""Compare matched OpenNeuro real-label and label-shuffle LOSO artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from neureptrace.loso_observation_diagnostics import confusion_matrix, write_loso_observation_diagnostics

PERCENT_METRICS = {
    "accuracy",
    "balanced_accuracy",
    "top2_accuracy",
    "top3_accuracy",
    "chance_accuracy",
    "top2_chance",
    "top3_chance",
}
MATCHED_PROVENANCE_COLUMNS = (
    "decoder",
    "backend",
    "emission_mode",
    "feature_preprocessor",
    "pca_components",
    "normalization",
    "temporal_mode",
    "class_prior_correction",
    "source_calibration",
    "source_time_selection",
    "source_time_selection_candidate_times",
    "source_time_selection_selected_time",
    "alignment_method",
    "alignment_anchor_mode",
    "alignment_anchor_column",
    "alignment_repetition_cap",
    "alignment_components",
    "alignment_times",
    "alignment_window_mode",
    "alignment_same_decode_window",
    "alignment_target_projection",
    "alignment_strict_source_only",
    "alignment_uses_unlabeled_target_data",
    "alignment_uses_class_labels",
    "alignment_target_calibrated",
    "alignment_target_calibration_per_anchor",
    "alignment_target_calibration_seed",
    "alignment_oracle_target_calibrated",
    "alignment_debug_upper_bound",
    "alignment_valid_for_benchmark",
    "alignment_protocol",
    "response_window_combine",
    "response_window_mode",
    "response_window_times",
    "response_window_requested_times",
    "response_window_actual_times",
    "temporal_smoothing_method",
    "temporal_smoothing_stay_probability",
    "temporal_smoothing_fit_window_start",
    "temporal_smoothing_fit_window_stop",
    "temporal_smoothing_apply_window_start",
    "temporal_smoothing_apply_window_stop",
)


def _artifact_root(path: str | Path) -> Path:
    """Return the OpenNeuro aggregate root inside a downloaded artifact directory."""

    candidate = Path(path)
    if (candidate / "decode" / "diagnostics" / "quality_summary.csv").is_file():
        return candidate
    matches = sorted(candidate.rglob("decode/diagnostics/quality_summary.csv"))
    if matches:
        return matches[0].parents[2]
    manifests = sorted(candidate.rglob("run_manifest.json"))
    for manifest in manifests:
        root = manifest.parent
        if (root / "decode" / "observations.csv").is_file():
            return root
    raise FileNotFoundError(f"Could not find an OpenNeuro aggregate artifact root below {candidate}.")


def _read_csv(root: Path, relative_path: str) -> pd.DataFrame:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required report input is missing: {path}")
    return pd.read_csv(path)


def _ensure_diagnostics(root: Path, fixed_time: float) -> None:
    diagnostics_dir = root / "decode" / "diagnostics"
    required = [
        diagnostics_dir / "quality_summary.csv",
        diagnostics_dir / "per_subject.csv",
        diagnostics_dir / "time_course_summary.csv",
        diagnostics_dir / "confusion_matrix.csv",
    ]
    if all(path.is_file() for path in required):
        return
    observations = root / "decode" / "observations.csv"
    summary = root / "decode" / "time_decode_summary.csv"
    stage_summary = root / "stage_summary.csv"
    write_loso_observation_diagnostics(
        observations,
        out_dir=diagnostics_dir,
        summary_csv=summary if summary.is_file() else None,
        stage_summary_csv=stage_summary if stage_summary.is_file() else None,
        best_time=fixed_time,
    )


def _nearest_row(frame: pd.DataFrame, time: float) -> pd.Series:
    if frame.empty or "time" not in frame.columns:
        raise ValueError("Time-course table must contain at least one time row.")
    times = frame["time"].astype(float).to_numpy()
    index = int(np.argmin(np.abs(times - float(time))))
    return frame.iloc[index]


def _best_time_row(frame: pd.DataFrame, metric: str = "balanced_accuracy") -> pd.Series:
    if frame.empty or metric not in frame.columns:
        raise ValueError(f"Time-course table must contain '{metric}'.")
    return frame.loc[frame[metric].astype(float).idxmax()]


def _pre_stimulus_summary(frame: pd.DataFrame) -> dict[str, float | int]:
    pre = frame.loc[frame["time"].astype(float) < 0].copy()
    if pre.empty:
        return {
            "n_pre_stimulus_times": 0,
            "pre_stimulus_balanced_accuracy_mean": float("nan"),
            "pre_stimulus_balanced_accuracy_max": float("nan"),
            "pre_stimulus_balanced_accuracy_max_time": float("nan"),
            "pre_stimulus_top2_accuracy_mean": float("nan"),
            "pre_stimulus_top2_accuracy_max": float("nan"),
        }
    max_index = pre["balanced_accuracy"].astype(float).idxmax()
    return {
        "n_pre_stimulus_times": int(len(pre)),
        "pre_stimulus_balanced_accuracy_mean": float(pre["balanced_accuracy"].astype(float).mean()),
        "pre_stimulus_balanced_accuracy_max": float(pre.loc[max_index, "balanced_accuracy"]),
        "pre_stimulus_balanced_accuracy_max_time": float(pre.loc[max_index, "time"]),
        "pre_stimulus_top2_accuracy_mean": float(pre["top2_accuracy"].astype(float).mean()),
        "pre_stimulus_top2_accuracy_max": float(pre["top2_accuracy"].astype(float).max()),
    }


def _first_row(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        raise ValueError("Expected a non-empty CSV table.")
    return frame.iloc[0]


def _float(row: pd.Series, column: str, default: float = float("nan")) -> float:
    if column not in row:
        return default
    value = pd.to_numeric(row[column], errors="coerce")
    return float(value) if pd.notna(value) else default


def _string(row: pd.Series, column: str, default: str = "") -> str:
    if column not in row or pd.isna(row[column]):
        return default
    return str(row[column])


def _as_bool_token(value: object) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean provenance value {value!r}.")


def _provenance_value_token(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if pd.isna(value):
        return ""
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"true", "false", "yes", "no", "on", "off"}:
        return lowered
    return text


def _provenance_values(run: dict[str, object], column: str) -> list[str]:
    values: list[str] = []
    for table_name in ("observations", "quality"):
        table = run.get(table_name)
        if not isinstance(table, pd.DataFrame) or column not in table.columns:
            continue
        values.extend(
            token
            for token in (_provenance_value_token(value) for value in table[column].drop_duplicates().tolist())
            if token
        )
    return sorted(set(values))


def _single_provenance_value(run: dict[str, object], column: str, *, label: str) -> str:
    values = _provenance_values(run, column)
    if not values:
        raise ValueError(f"{label} artifact is missing required {column!r} provenance.")
    if len(values) != 1:
        raise ValueError(f"{label} artifact has inconsistent {column!r} provenance values: {values}")
    return values[0]


def _validate_shuffle_provenance(real: dict[str, object], shuffle: dict[str, object]) -> dict[str, object]:
    real_control = _as_bool_token(_single_provenance_value(real, "label_shuffle_control", label="real"))
    shuffle_control = _as_bool_token(_single_provenance_value(shuffle, "label_shuffle_control", label="shuffle"))
    if real_control:
        raise ValueError("real artifact is marked label_shuffle_control=true; pass the non-shuffled artifact as --real-dir.")
    if not shuffle_control:
        raise ValueError("shuffle artifact is not marked label_shuffle_control=true; pass a train-label-shuffle artifact as --shuffle-dir.")
    return {
        "real_label_shuffle_control": real_control,
        "shuffle_label_shuffle_control": shuffle_control,
        "real_label_shuffle_seed": ",".join(_provenance_values(real, "label_shuffle_seed")),
        "shuffle_label_shuffle_seed": ",".join(_provenance_values(shuffle, "label_shuffle_seed")),
    }


def _validate_matched_provenance(real: dict[str, object], shuffle: dict[str, object]) -> None:
    mismatches: list[str] = []
    for column in MATCHED_PROVENANCE_COLUMNS:
        real_values = _provenance_values(real, column)
        shuffle_values = _provenance_values(shuffle, column)
        if not real_values or not shuffle_values:
            continue
        if real_values != shuffle_values:
            mismatches.append(f"{column}: real={real_values}, shuffle={shuffle_values}")
    if mismatches:
        raise ValueError(
            "Real and shuffle artifacts are not matched on decoder/protocol provenance: "
            + "; ".join(mismatches[:8])
        )


def _fixed_metric_row(quality: pd.DataFrame, time_course: pd.DataFrame, fixed_time: float) -> pd.Series:
    row = _first_row(quality)
    if "fixed_time" in row and np.isclose(_float(row, "fixed_time"), fixed_time):
        return row
    fixed = _nearest_row(time_course, fixed_time)
    mapped = fixed.rename(
        {
            "accuracy": "fixed_accuracy",
            "balanced_accuracy": "fixed_balanced_accuracy",
            "top2_accuracy": "fixed_top2_accuracy",
            "top3_accuracy": "fixed_top3_accuracy",
            "log_loss": "fixed_log_loss",
            "brier": "fixed_brier",
            "ece": "fixed_ece",
            "time": "fixed_time",
        }
    )
    combined = row.copy()
    for column, value in mapped.items():
        combined[column] = value
    return combined


def _load_run(root: Path, fixed_time: float) -> dict[str, pd.DataFrame | pd.Series | Path]:
    _ensure_diagnostics(root, fixed_time)
    quality = _read_csv(root, "decode/diagnostics/quality_summary.csv")
    per_subject = _read_csv(root, "decode/diagnostics/per_subject.csv")
    time_course = _read_csv(root, "decode/diagnostics/time_course_summary.csv")
    observations = _read_csv(root, "decode/observations.csv")
    fixed_quality = _fixed_metric_row(quality, time_course, fixed_time)
    fixed_confusion = confusion_matrix(observations, time=fixed_time)
    return {
        "root": root,
        "quality": quality,
        "fixed_quality": fixed_quality,
        "per_subject": per_subject,
        "time_course": time_course,
        "observations": observations,
        "confusion": fixed_confusion,
    }


def _summary_row(real: dict[str, object], shuffle: dict[str, object], fixed_time: float) -> pd.DataFrame:
    real_quality = real["fixed_quality"]
    shuffle_quality = shuffle["fixed_quality"]
    assert isinstance(real_quality, pd.Series)
    assert isinstance(shuffle_quality, pd.Series)
    real_time = real["time_course"]
    shuffle_time = shuffle["time_course"]
    assert isinstance(real_time, pd.DataFrame)
    assert isinstance(shuffle_time, pd.DataFrame)
    real_best = _best_time_row(real_time)
    shuffle_best = _best_time_row(shuffle_time)
    real_pre = _pre_stimulus_summary(real_time)
    shuffle_pre = _pre_stimulus_summary(shuffle_time)
    real_per_subject = real["per_subject"]
    shuffle_per_subject = shuffle["per_subject"]
    assert isinstance(real_per_subject, pd.DataFrame)
    assert isinstance(shuffle_per_subject, pd.DataFrame)
    merged_subjects = _per_subject_delta(real_per_subject, shuffle_per_subject)
    if merged_subjects.empty:
        raise ValueError("Real and shuffle artifacts have no overlapping subjects to compare.")
    real_subjects = set(real_per_subject["subject"].dropna().astype(str))
    shuffle_subjects = set(shuffle_per_subject["subject"].dropna().astype(str))
    if real_subjects != shuffle_subjects:
        raise ValueError(
            "Real and shuffle artifacts have different subject sets: "
            f"real_only={sorted(real_subjects - shuffle_subjects)}, "
            f"shuffle_only={sorted(shuffle_subjects - real_subjects)}."
        )
    for column in ("n_trials", "class_counts"):
        real_column = f"{column}_real"
        shuffle_column = f"{column}_shuffle"
        if real_column not in merged_subjects.columns or shuffle_column not in merged_subjects.columns:
            continue
        real_values = merged_subjects[real_column].astype(str)
        shuffle_values = merged_subjects[shuffle_column].astype(str)
        mismatched = merged_subjects.loc[real_values.ne(shuffle_values), ["subject", real_column, shuffle_column]]
        if not mismatched.empty:
            raise ValueError(
                f"Real and shuffle artifacts have different per-subject {column}. "
                f"Mismatch examples: {mismatched.head(5).to_dict('records')}"
            )
    real_fixed_time = _float(real_quality, "fixed_time")
    shuffle_fixed_time = _float(shuffle_quality, "fixed_time")
    if not np.isclose(real_fixed_time, shuffle_fixed_time, atol=1e-9):
        raise ValueError(
            "Real and shuffle artifacts resolved to different fixed diagnostic times: "
            f"real={real_fixed_time}, shuffle={shuffle_fixed_time}."
        )
    real_n_classes = int(_float(real_quality, "n_classes"))
    shuffle_n_classes = int(_float(shuffle_quality, "n_classes"))
    if real_n_classes != shuffle_n_classes:
        raise ValueError(f"Real and shuffle artifacts have different class counts: real={real_n_classes}, shuffle={shuffle_n_classes}.")
    real_fixed = _float(real_quality, "fixed_balanced_accuracy")
    shuffle_fixed = _float(shuffle_quality, "fixed_balanced_accuracy")
    real_top2 = _float(real_quality, "fixed_top2_accuracy")
    shuffle_top2 = _float(shuffle_quality, "fixed_top2_accuracy")
    real_top3 = _float(real_quality, "fixed_top3_accuracy")
    shuffle_top3 = _float(shuffle_quality, "fixed_top3_accuracy")
    return pd.DataFrame(
        [
            {
                "fixed_time_requested": float(fixed_time),
                "fixed_time_real": real_fixed_time,
                "fixed_time_shuffle": shuffle_fixed_time,
                "n_subjects_real": int(_float(real_quality, "n_subjects")),
                "n_subjects_shuffle": int(_float(shuffle_quality, "n_subjects")),
                "n_classes": real_n_classes,
                "chance_accuracy": _float(real_quality, "chance_accuracy"),
                "top2_chance": _float(real_quality, "top2_chance"),
                "top3_chance": _float(real_quality, "top3_chance"),
                "fixed_accuracy_real": _float(real_quality, "fixed_accuracy"),
                "fixed_accuracy_shuffle": _float(shuffle_quality, "fixed_accuracy"),
                "fixed_accuracy_delta": _float(real_quality, "fixed_accuracy") - _float(shuffle_quality, "fixed_accuracy"),
                "fixed_balanced_accuracy_real": real_fixed,
                "fixed_balanced_accuracy_shuffle": shuffle_fixed,
                "fixed_balanced_accuracy_delta": real_fixed - shuffle_fixed,
                "fixed_balanced_accuracy_delta_pp": 100.0 * (real_fixed - shuffle_fixed),
                "fixed_top2_accuracy_real": real_top2,
                "fixed_top2_accuracy_shuffle": shuffle_top2,
                "fixed_top2_accuracy_delta": real_top2 - shuffle_top2,
                "fixed_top2_accuracy_delta_pp": 100.0 * (real_top2 - shuffle_top2),
                "fixed_top3_accuracy_real": real_top3,
                "fixed_top3_accuracy_shuffle": shuffle_top3,
                "fixed_top3_accuracy_delta": real_top3 - shuffle_top3,
                "top2_interpretation": _string(real_quality, "top2_interpretation"),
                "top3_interpretation": _string(real_quality, "top3_interpretation"),
                "subjects_real_above_shuffle_fixed": int(merged_subjects["fixed_balanced_accuracy_delta"].gt(0).sum()),
                "subjects_compared": int(len(merged_subjects)),
                "exploratory_best_time_real": float(real_best["time"]),
                "exploratory_best_balanced_accuracy_real": float(real_best["balanced_accuracy"]),
                "exploratory_best_time_shuffle": float(shuffle_best["time"]),
                "exploratory_best_balanced_accuracy_shuffle": float(shuffle_best["balanced_accuracy"]),
                "exploratory_best_balanced_accuracy_delta": float(real_best["balanced_accuracy"]) - float(shuffle_best["balanced_accuracy"]),
                **{f"real_{key}": value for key, value in real_pre.items()},
                **{f"shuffle_{key}": value for key, value in shuffle_pre.items()},
            }
        ]
    )


def _per_subject_delta(real_per_subject: pd.DataFrame, shuffle_per_subject: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "subject",
        "best_time",
        "balanced_accuracy",
        "top2_accuracy",
        "top3_accuracy",
        "fixed_time",
        "fixed_balanced_accuracy",
        "fixed_top2_accuracy",
        "fixed_top3_accuracy",
        "n_trials",
        "class_counts",
    ]
    real = real_per_subject.loc[:, [column for column in columns if column in real_per_subject.columns]].copy()
    shuffle = shuffle_per_subject.loc[:, [column for column in columns if column in shuffle_per_subject.columns]].copy()
    merged = real.merge(shuffle, on="subject", how="inner", suffixes=("_real", "_shuffle"))
    for metric in ("balanced_accuracy", "top2_accuracy", "top3_accuracy", "fixed_balanced_accuracy", "fixed_top2_accuracy", "fixed_top3_accuracy"):
        real_column = f"{metric}_real"
        shuffle_column = f"{metric}_shuffle"
        if real_column in merged.columns and shuffle_column in merged.columns:
            merged[f"{metric}_delta"] = pd.to_numeric(merged[real_column], errors="coerce") - pd.to_numeric(merged[shuffle_column], errors="coerce")
    if "fixed_balanced_accuracy_delta" in merged.columns:
        merged["real_above_shuffle_fixed_balanced"] = merged["fixed_balanced_accuracy_delta"] > 0
    return merged.sort_values("subject").reset_index(drop=True)


def _classwise_recalls(confusion: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for true_class, frame in confusion.groupby("true_class", sort=True):
        total = int(frame["count"].sum())
        correct = int(frame.loc[frame["predicted_class"].astype(str) == str(true_class), "count"].sum())
        rows.append({"class": str(true_class), "n": total, "recall": correct / total if total else float("nan")})
    return pd.DataFrame(rows)


def _confusion_wide(confusion: pd.DataFrame) -> pd.DataFrame:
    return (
        confusion.pivot_table(index="true_class", columns="predicted_class", values="count", aggfunc="sum", fill_value=0)
        .sort_index()
        .sort_index(axis=1)
        .reset_index()
    )


def _format_value(value: object, *, percent: bool = False) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{100.0 * float(value):.2f}%" if percent else f"{float(value):.3f}"
    return str(value)


def _humanize_token(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def _markdown_table(rows: Iterable[Sequence[object]], headers: Sequence[str]) -> str:
    rendered_rows = [[str(cell) for cell in row] for row in rows]
    rendered_headers = [str(header) for header in headers]
    widths = [len(header) for header in rendered_headers]
    for row in rendered_rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    header = "| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(rendered_headers)) + " |"
    divider = "| " + " | ".join("-" * widths[index] for index in range(len(widths))) + " |"
    body = ["| " + " | ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)) + " |" for row in rendered_rows]
    return "\n".join([header, divider, *body])


def _metric_table(real_row: pd.Series, shuffle_row: pd.Series, metrics: Sequence[tuple[str, str]]) -> str:
    rows = []
    for label, column in metrics:
        percent = any(name in column for name in PERCENT_METRICS)
        real_value = _float(real_row, column)
        shuffle_value = _float(shuffle_row, column)
        rows.append(
            [
                label,
                _format_value(real_value, percent=percent),
                _format_value(shuffle_value, percent=percent),
                _format_value(real_value - shuffle_value, percent=percent),
            ]
        )
    return _markdown_table(rows, ["metric", "real", "shuffle", "real - shuffle"])


def _write_markdown(
    *,
    summary: pd.DataFrame,
    per_subject: pd.DataFrame,
    real: dict[str, object],
    shuffle: dict[str, object],
    out_path: Path,
) -> None:
    summary_row = summary.iloc[0]
    real_quality = real["fixed_quality"]
    shuffle_quality = shuffle["fixed_quality"]
    real_time = real["time_course"]
    shuffle_time = shuffle["time_course"]
    real_confusion = real["confusion"]
    shuffle_confusion = shuffle["confusion"]
    assert isinstance(real_quality, pd.Series)
    assert isinstance(shuffle_quality, pd.Series)
    assert isinstance(real_time, pd.DataFrame)
    assert isinstance(shuffle_time, pd.DataFrame)
    assert isinstance(real_confusion, pd.DataFrame)
    assert isinstance(shuffle_confusion, pd.DataFrame)
    real_best = _best_time_row(real_time)
    shuffle_best = _best_time_row(shuffle_time)
    real_pre = _pre_stimulus_summary(real_time)
    shuffle_pre = _pre_stimulus_summary(shuffle_time)
    real_recalls = _classwise_recalls(real_confusion).merge(_classwise_recalls(shuffle_confusion), on="class", suffixes=("_real", "_shuffle"))
    real_recalls["recall_delta"] = real_recalls["recall_real"] - real_recalls["recall_shuffle"]
    report_label = out_path.stem.replace("_", " ").strip() or "OpenNeuro"

    lines = [
        f"# {report_label} Real vs Label-Shuffle LOSO Report",
        "",
        f"Fixed diagnostic time: {_format_value(summary_row['fixed_time_real'])} s real, {_format_value(summary_row['fixed_time_shuffle'])} s shuffle.",
        f"Subjects compared: {int(summary_row['subjects_compared'])}; classes: {int(summary_row['n_classes'])}.",
        "",
        "## Fixed-Time Real vs Shuffle",
        "",
        _metric_table(
            real_quality,
            shuffle_quality,
            [
                ("accuracy", "fixed_accuracy"),
                ("balanced accuracy", "fixed_balanced_accuracy"),
                ("top-2 accuracy", "fixed_top2_accuracy"),
                ("top-3 accuracy", "fixed_top3_accuracy"),
                ("log loss", "fixed_log_loss"),
                ("Brier score", "fixed_brier"),
                ("ECE", "fixed_ece"),
            ],
        ),
        "",
        f"Real exceeds shuffle for fixed-time balanced accuracy in {int(summary_row['subjects_real_above_shuffle_fixed'])}/{int(summary_row['subjects_compared'])} subjects.",
        "",
        "## Best-Time Real vs Shuffle (Exploratory)",
        "",
        "These rows select each condition's own best time and are exploratory, not the primary fixed-time result.",
        "",
        _markdown_table(
            [
                [
                    "real",
                    _format_value(float(real_best["time"])),
                    _format_value(float(real_best["balanced_accuracy"]), percent=True),
                    _format_value(float(real_best["top2_accuracy"]), percent=True),
                    _format_value(float(real_best["top3_accuracy"]), percent=True),
                ],
                [
                    "shuffle",
                    _format_value(float(shuffle_best["time"])),
                    _format_value(float(shuffle_best["balanced_accuracy"]), percent=True),
                    _format_value(float(shuffle_best["top2_accuracy"]), percent=True),
                    _format_value(float(shuffle_best["top3_accuracy"]), percent=True),
                ],
            ],
            ["condition", "best time (s)", "balanced accuracy", "top-2", "top-3"],
        ),
        "",
        "## Pre-Stimulus Sanity Check",
        "",
        _markdown_table(
            [
                [
                    "real",
                    int(real_pre["n_pre_stimulus_times"]),
                    _format_value(real_pre["pre_stimulus_balanced_accuracy_mean"], percent=True),
                    _format_value(real_pre["pre_stimulus_balanced_accuracy_max"], percent=True),
                    _format_value(real_pre["pre_stimulus_balanced_accuracy_max_time"]),
                    _format_value(real_pre["pre_stimulus_top2_accuracy_mean"], percent=True),
                ],
                [
                    "shuffle",
                    int(shuffle_pre["n_pre_stimulus_times"]),
                    _format_value(shuffle_pre["pre_stimulus_balanced_accuracy_mean"], percent=True),
                    _format_value(shuffle_pre["pre_stimulus_balanced_accuracy_max"], percent=True),
                    _format_value(shuffle_pre["pre_stimulus_balanced_accuracy_max_time"]),
                    _format_value(shuffle_pre["pre_stimulus_top2_accuracy_mean"], percent=True),
                ],
            ],
            ["condition", "n times", "mean balanced", "max balanced", "max time (s)", "mean top-2"],
        ),
        "",
        "## Per-Subject Deltas",
        "",
        _markdown_table(
            [
                [
                    row.subject,
                    _format_value(row.fixed_balanced_accuracy_real, percent=True),
                    _format_value(row.fixed_balanced_accuracy_shuffle, percent=True),
                    _format_value(row.fixed_balanced_accuracy_delta, percent=True),
                    _format_value(row.fixed_top2_accuracy_real, percent=True),
                    _format_value(row.fixed_top2_accuracy_shuffle, percent=True),
                ]
                for row in per_subject.itertuples(index=False)
            ],
            ["subject", "real balanced", "shuffle balanced", "delta", "real top-2", "shuffle top-2"],
        ),
        "",
        "## Confusion Matrix at Fixed Time",
        "",
        "Real-label confusion matrix:",
        "",
        _markdown_table(_confusion_wide(real_confusion).itertuples(index=False, name=None), list(_confusion_wide(real_confusion).columns)),
        "",
        "Label-shuffle confusion matrix:",
        "",
        _markdown_table(_confusion_wide(shuffle_confusion).itertuples(index=False, name=None), list(_confusion_wide(shuffle_confusion).columns)),
        "",
        "## Classwise Balanced Recalls",
        "",
        _markdown_table(
            [
                [
                    row["class"],
                    int(row["n_real"]),
                    _format_value(row["recall_real"], percent=True),
                    _format_value(row["recall_shuffle"], percent=True),
                    _format_value(row["recall_delta"], percent=True),
                ]
                for _, row in real_recalls.iterrows()
            ],
            ["class", "n fixed-time trials", "real recall", "shuffle recall", "delta"],
        ),
        "",
        "## Top-k Interpretation",
        "",
        f"Top-2 is {_humanize_token(_string(real_quality, 'top2_interpretation', 'unknown'))}: with {int(summary_row['n_classes'])} classes, chance top-2 is {_format_value(summary_row['top2_chance'], percent=True)}, so top-2 can support the fixed-time result.",
        f"Top-3 is {_humanize_token(_string(real_quality, 'top3_interpretation', 'unknown'))}: with {int(summary_row['n_classes'])} classes, top-3 is {_format_value(summary_row['top3_chance'], percent=True)} by construction and is not evidence for decoding.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_real_shuffle_report(
    *,
    real_dir: str | Path,
    shuffle_dir: str | Path,
    out_dir: str | Path,
    fixed_time: float,
    output_prefix: str = "ds006629_real_vs_shuffle",
) -> dict[str, Path]:
    """Write CSV and Markdown reports comparing a real-label run to its matched shuffle null."""

    real_root = _artifact_root(real_dir)
    shuffle_root = _artifact_root(shuffle_dir)
    real = _load_run(real_root, fixed_time)
    shuffle = _load_run(shuffle_root, fixed_time)
    shuffle_provenance = _validate_shuffle_provenance(real, shuffle)
    _validate_matched_provenance(real, shuffle)
    summary = _summary_row(real, shuffle, fixed_time)
    for column, value in shuffle_provenance.items():
        summary[column] = value
    real_per_subject = real["per_subject"]
    shuffle_per_subject = shuffle["per_subject"]
    assert isinstance(real_per_subject, pd.DataFrame)
    assert isinstance(shuffle_per_subject, pd.DataFrame)
    per_subject = _per_subject_delta(real_per_subject, shuffle_per_subject)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": output_dir / f"{output_prefix}_summary.csv",
        "per_subject": output_dir / f"{output_prefix}_per_subject.csv",
        "markdown": output_dir / f"{output_prefix}.md",
    }
    summary.to_csv(paths["summary"], index=False)
    per_subject.to_csv(paths["per_subject"], index=False)
    _write_markdown(summary=summary, per_subject=per_subject, real=real, shuffle=shuffle, out_path=paths["markdown"])
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", type=Path, required=True, help="Downloaded real-label aggregate artifact directory.")
    parser.add_argument("--shuffle-dir", type=Path, required=True, help="Downloaded label-shuffle aggregate artifact directory.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for report files.")
    parser.add_argument("--fixed-time", type=float, required=True, help="Primary fixed diagnostic time in seconds.")
    parser.add_argument("--output-prefix", default="ds006629_real_vs_shuffle", help="Filename prefix for report outputs.")
    args = parser.parse_args(argv)

    paths = write_real_shuffle_report(
        real_dir=args.real_dir,
        shuffle_dir=args.shuffle_dir,
        out_dir=args.out_dir,
        fixed_time=args.fixed_time,
        output_prefix=args.output_prefix,
    )
    for label, path in paths.items():
        print(f"Wrote {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

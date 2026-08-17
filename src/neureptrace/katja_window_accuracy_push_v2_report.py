"""Aggregate and report the Katja sliding-window accuracy-push v2 analyses."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = (
    "Source-only",
    "Direct Transformer ensemble",
    "Direct + auxiliary blend",
    "Direct + auxiliary blend + prior match",
    "Geometric structured Transformer",
    "Explicit-duration Transformer",
    "Auxiliary heads + duration only",
    "Task timing/template prior only",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _read_validated_result(path: Path) -> pd.DataFrame:
    validation_path = path / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if not validation.get("all_required_checks_pass", False):
        raise ValueError(f"Validation failed for {path}")
    return pd.read_csv(path / "fold_results.csv")


def _seed_averaged_summary(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "accuracy_raw_labels",
        "balanced_accuracy_raw_labels",
        "press_only_finger_accuracy",
        "rest_recall",
        "press_detection_accuracy",
    ]
    subject = rows.groupby(
        ["method_label", "protocol_scope", "k_trials_per_sequence", "target"],
        as_index=False,
    )[metrics].mean()
    summaries: list[dict[str, Any]] = []
    for keys, frame in subject.groupby(
        ["method_label", "protocol_scope", "k_trials_per_sequence"], sort=True
    ):
        method, scope, k = keys
        row: dict[str, Any] = {
            "method_label": method,
            "protocol_scope": scope,
            "k_trials_per_sequence": int(k),
            "n_subjects": int(frame["target"].nunique()),
        }
        for metric in metrics:
            values = frame[metric].to_numpy(dtype=float)
            row[f"mean_{metric}"] = float(np.mean(values))
            row[f"sem_{metric}"] = (
                float(np.std(values, ddof=1) / math.sqrt(values.size))
                if values.size > 1
                else float("nan")
            )
        summaries.append(row)
    return subject, pd.DataFrame(summaries)


def _method_rows(
    rows: pd.DataFrame,
    *,
    method: str | None,
    family: str | None,
    label: str,
    scope: str,
) -> pd.DataFrame:
    selected = rows.copy()
    if method is not None:
        selected = selected[selected["method"].eq(method)]
    if family is not None:
        selected = selected[selected["family"].eq(family)]
    selected = selected.copy()
    selected["method_label"] = label
    selected["protocol_scope"] = scope
    return selected


def _plot(summary: pd.DataFrame, output_dir: Path) -> None:
    styles = {
        "Source-only": dict(color="#111111", marker="D", linestyle="--"),
        "Direct Transformer ensemble": dict(color="#377eb8", marker="o", linestyle="-"),
        "Direct + auxiliary blend": dict(color="#00a6a6", marker="v", linestyle="--"),
        "Direct + auxiliary blend + prior match": dict(
            color="#4daf4a", marker="P", linestyle="-"
        ),
        "Geometric structured Transformer": dict(
            color="#984ea3", marker="s", linestyle="-"
        ),
        "Explicit-duration Transformer": dict(color="#e41a1c", marker="o", linestyle="-"),
        "Auxiliary heads + duration only": dict(
            color="#ff7f00", marker="^", linestyle="none"
        ),
        "Task timing/template prior only": dict(
            color="#777777", marker="x", linestyle="none"
        ),
    }
    fig, axis = plt.subplots(figsize=(11.8, 5.9), constrained_layout=True)
    axis.fill_between(
        [17.0, 23.0],
        [0.625, 0.625],
        [0.645, 0.645],
        color="#b8d8c0",
        alpha=0.45,
        label="Julia reported endpoint range",
    )
    for method in METHOD_ORDER:
        frame = summary[summary["method_label"].eq(method)].sort_values(
            "k_trials_per_sequence"
        )
        if frame.empty:
            continue
        style = styles[method]
        axis.errorbar(
            frame["k_trials_per_sequence"],
            frame["mean_accuracy_raw_labels"],
            yerr=frame["sem_accuracy_raw_labels"],
            linewidth=2.0,
            markersize=6.5,
            capsize=3,
            label=method,
            **style,
        )
    ticks = sorted(summary["k_trials_per_sequence"].unique())
    axis.set_xscale("log", base=2)
    axis.set_xticks(ticks, [str(int(value)) for value in ticks])
    axis.set_xlim(0.85, 24.0)
    axis.set_xlabel("Labeled calibration trials per sequence ID (k)")
    axis.set_ylabel("Six-class sliding-window accuracy")
    axis.set_ylim(0.25, min(0.82, max(0.75, summary["mean_accuracy_raw_labels"].max() + 0.06)))
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=False,
        fontsize=8.5,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"katja_window_accuracy_push_v2.{suffix}", dpi=220)
    plt.close(fig)


def _format_percent(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def build_report(args: argparse.Namespace) -> Path:
    baseline_root = Path(args.baseline_root).expanduser().resolve()
    output_dir = Path(args.out_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = pd.read_csv(baseline_root / "combined" / "fold_results.csv")
    duration_shards = [Path(value).expanduser().resolve() for value in args.duration_shards.split(",")]
    duration_rows = pd.concat(
        [_read_validated_result(path) for path in duration_shards], ignore_index=True
    ).drop_duplicates(
        ["target", "family", "split_seed", "k_trials_per_sequence", "candidate_id"]
    )
    duration_rows.to_csv(output_dir / "duration_curve_fold_results.csv", index=False)
    prior_rows = _read_validated_result(Path(args.prior_match_root).expanduser().resolve())
    strict_composition_rows = _read_validated_result(
        Path(args.strict_composition_root).expanduser().resolve()
    )
    control_rows = _read_validated_result(Path(args.control_root).expanduser().resolve())

    methods = pd.concat(
        [
            _method_rows(
                baseline,
                method="hierarchical_source_only",
                family=None,
                label="Source-only",
                scope="Protocol 1 source-only reference",
            ),
            _method_rows(
                baseline,
                method="trial_transformer_ensemble",
                family=None,
                label="Direct Transformer ensemble",
                scope="Protocol 3 offline trial-context, independent finger head",
            ),
            _method_rows(
                strict_composition_rows,
                method=None,
                family="trial_transformer_offline",
                label="Direct + auxiliary blend",
                scope="Exploratory Protocol 3 calibration-only composition",
            ),
            _method_rows(
                prior_rows,
                method=None,
                family="trial_transformer_offline",
                label="Direct + auxiliary blend + prior match",
                scope="Exploratory transductive Protocol 2+3",
            ),
            _method_rows(
                baseline,
                method="trial_transformer_ensemble_structured",
                family=None,
                label="Geometric structured Transformer",
                scope="Protocol 3 offline structured",
            ),
            _method_rows(
                duration_rows,
                method=None,
                family="trial_transformer_offline",
                label="Explicit-duration Transformer",
                scope="Protocol 3 offline structured",
            ),
            _method_rows(
                control_rows,
                method=None,
                family="auxiliary_duration_prior",
                label="Auxiliary heads + duration only",
                scope="Protocol 3 offline structured ablation",
            ),
            _method_rows(
                control_rows,
                method=None,
                family="duration_prior_only",
                label="Task timing/template prior only",
                scope="Non-neural task-prior control",
            ),
        ],
        ignore_index=True,
    )
    common_targets = sorted(
        set.intersection(
            *(
                set(frame["target"])
                for _, frame in duration_rows.groupby("k_trials_per_sequence")
            )
        )
    )
    common = methods[methods["target"].isin(common_targets)].copy()
    subject, summary = _seed_averaged_summary(common)
    subject.to_csv(output_dir / "subject_seed_averages_common9.csv", index=False)
    summary.to_csv(output_dir / "summary_common9.csv", index=False)
    methods.to_csv(output_dir / "combined_fold_results.csv", index=False)

    k20_subject = subject[subject["k_trials_per_sequence"].eq(20)]
    pivot = k20_subject.pivot(index="target", columns="method_label", values="accuracy_raw_labels")
    paired_rows: list[dict[str, Any]] = []
    reference = "Direct Transformer ensemble"
    for method in pivot.columns:
        if method == reference:
            continue
        paired = pivot[[reference, method]].dropna()
        delta = paired[method] - paired[reference]
        paired_rows.append(
            {
                "method_label": method,
                "reference_method": reference,
                "n_subjects": int(len(paired)),
                "mean_paired_delta_accuracy": float(delta.mean()),
                "sem_paired_delta_accuracy": (
                    float(delta.std(ddof=1) / math.sqrt(len(delta)))
                    if len(delta) > 1
                    else float("nan")
                ),
            }
        )
    pd.DataFrame(paired_rows).to_csv(output_dir / "paired_k20_vs_direct.csv", index=False)
    _plot(summary, output_dir)

    k20 = summary[summary["k_trials_per_sequence"].eq(20)].set_index("method_label")
    report_lines = [
        "# Katja sliding-window accuracy push v2",
        "",
        "All means average five split seeds within participant first, then average the fixed common-nine participant cohort. Error bars are SEM across participants.",
        "",
        "## k=20 results",
        "",
        "| Method | Accuracy | SEM | Scope |",
        "|---|---:|---:|---|",
    ]
    for method in METHOD_ORDER:
        if method not in k20.index:
            continue
        row = k20.loc[method]
        report_lines.append(
            f"| {method} | {_format_percent(row['mean_accuracy_raw_labels'])} | "
            f"{_format_percent(row['sem_accuracy_raw_labels'])} | {row['protocol_scope']} |"
        )
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The explicit-duration result is an offline structured decoder. It uses trial "
                "boundaries, five ordered presses, two target templates learned from calibration "
                "trials, and source-plus-calibration duration priors. It must not replace the "
                "independent-window comparison against Julia."
            ),
            "",
            (
                "The auxiliary-blend prior-matched result remains window-wise after the "
                "bidirectional trial Transformer, but it matches the unlabeled evaluation-batch "
                "marginal to a calibration-derived class prior. It is therefore transductive "
                "Protocol 2+3 and is reported as exploratory."
            ),
            "",
            "The timing/template-only and auxiliary-only rows quantify how much of the structured gain is attributable to task priors versus learned neural auxiliary heads.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    validation = {
        "all_required_checks_pass": bool(
            len(common_targets) == 9
            and set(summary["method_label"]).issubset(set(METHOD_ORDER))
            and duration_rows["evaluation_labels_used_for_fitting"].eq(False).all()
        ),
        "common_targets": common_targets,
        "n_common_targets": len(common_targets),
        "seeds_averaged_within_subject_before_population_sem": True,
        "structured_and_transductive_results_separated_from_direct_comparison": True,
    }
    _atomic_json(output_dir / "validation.json", validation)
    _atomic_json(
        output_dir / "provenance.json",
        {
            "baseline_root": str(baseline_root),
            "duration_shards": [str(path) for path in duration_shards],
            "prior_match_root": str(Path(args.prior_match_root).expanduser().resolve()),
            "strict_composition_root": str(
                Path(args.strict_composition_root).expanduser().resolve()
            ),
            "control_root": str(Path(args.control_root).expanduser().resolve()),
            "common_targets": common_targets,
        },
    )
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--duration-shards", required=True)
    parser.add_argument("--prior-match-root", required=True)
    parser.add_argument("--strict-composition-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    build_report(build_arg_parser().parse_args(argv))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

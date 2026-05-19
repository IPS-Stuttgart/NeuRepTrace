from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class VariantSpec:
    """One all-subject NOD animate/inanimate benchmark variant."""

    name: str
    manifest: str
    output_dir: str
    temporal_smoothing: bool = False


@dataclass(frozen=True)
class SweepCommand:
    """A labelled command in the reproducible variant sweep."""

    label: str
    argv: tuple[str, ...]


DEFAULT_VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        name="baseline_logistic",
        manifest="nod_animate_all.csv",
        output_dir="nod_animate_all",
    ),
    VariantSpec(
        name="logistic_pca_whiten_tuned",
        manifest="nod_animate_logistic_tuned_pca_whiten_all.csv",
        output_dir="nod_animate_logistic_tuned_pca_whiten_all",
    ),
    VariantSpec(
        name="logistic_anova_select_tuned",
        manifest="nod_animate_logistic_tuned_anova_select_all.csv",
        output_dir="nod_animate_logistic_tuned_anova_select_all",
    ),
    VariantSpec(
        name="shrinkage_lda",
        manifest="nod_animate_shrinkage_lda_all.csv",
        output_dir="nod_animate_shrinkage_lda_all",
    ),
    VariantSpec(
        name="elastic_net_logistic",
        manifest="nod_animate_elastic_net_logistic_all.csv",
        output_dir="nod_animate_elastic_net_logistic_all",
    ),
    VariantSpec(
        name="logistic_temporal_train_window_tuned",
        manifest="nod_animate_logistic_tuned_temporal_ensemble_all.csv",
        output_dir="nod_animate_logistic_tuned_temporal_ensemble_all",
    ),
    VariantSpec(
        name="logistic_temporal_smoothing",
        manifest="nod_animate_logistic_temporal_smoothing_all.csv",
        output_dir="nod_animate_logistic_temporal_smoothing_all",
        temporal_smoothing=True,
    ),
)

DEFAULT_VARIANT_BY_NAME = {variant.name: variant for variant in DEFAULT_VARIANTS}


def select_variants(names: Sequence[str] | None) -> tuple[VariantSpec, ...]:
    """Resolve CLI variant names while preserving the requested order."""

    if not names:
        return DEFAULT_VARIANTS
    unknown = [name for name in names if name not in DEFAULT_VARIANT_BY_NAME]
    if unknown:
        known = ", ".join(DEFAULT_VARIANT_BY_NAME)
        raise ValueError(f"Unknown variant name(s): {', '.join(unknown)}. Known variants: {known}.")
    return tuple(DEFAULT_VARIANT_BY_NAME[name] for name in names)


def _command(label: str, argv: Sequence[str | Path | float | int]) -> SweepCommand:
    return SweepCommand(
        label=label,
        argv=tuple(part.as_posix() if isinstance(part, Path) else str(part) for part in argv),
    )


def build_commands(
    variants: Sequence[VariantSpec],
    *,
    benchmarks_dir: Path,
    results_root: Path,
    python_executable: str = sys.executable,
    chance: float = 0.5,
    n_permutations: int = 10000,
    cluster_alpha: float = 0.05,
    validate: bool = True,
    report: bool = True,
    inference: bool = True,
    calibration: bool = True,
    resume: bool = False,
) -> list[SweepCommand]:
    """Build the complete command list for the NOD all-subject variant sweep."""

    commands: list[SweepCommand] = []
    for variant in variants:
        manifest = benchmarks_dir / variant.manifest
        out_dir = results_root / variant.output_dir
        aggregate_csv = out_dir / "summary.csv"
        subject_glob = out_dir / "sub-*_time_decode.csv"
        calibration_dir = out_dir / "calibration"

        if validate:
            commands.append(
                _command(
                    f"validate:{variant.name}",
                    (
                        python_executable,
                        "-m",
                        "reptrace.validate_manifest",
                        manifest,
                        "--report-out",
                        out_dir / "manifest_validation.csv",
                    ),
                )
            )

        benchmark_argv: list[str | Path | float] = [
            python_executable,
            "-m",
            "reptrace.benchmark",
            manifest,
            "--out-dir",
            out_dir,
            "--aggregate-out",
            aggregate_csv,
            "--plot-out",
            out_dir / "summary.png",
            "--calibration-dir",
            calibration_dir,
            "--chance",
            chance,
        ]
        if variant.temporal_smoothing:
            benchmark_argv.extend(
                [
                    "--temporal-smoothing-dir",
                    out_dir / "temporal_smoothing",
                    "--temporal-smoothing-fit-window",
                    0.1,
                    0.8,
                ]
            )
        if resume:
            benchmark_argv.append("--resume")
        commands.append(_command(f"benchmark:{variant.name}", benchmark_argv))

        if report:
            commands.append(
                _command(
                    f"report:{variant.name}",
                    (
                        python_executable,
                        "-m",
                        "reptrace.report",
                        aggregate_csv,
                        subject_glob,
                        "--chance",
                        chance,
                        "--out",
                        out_dir / "report.md",
                    ),
                )
            )

        if inference:
            commands.append(
                _command(
                    f"inference:{variant.name}",
                    (
                        python_executable,
                        "-m",
                        "reptrace.inference",
                        subject_glob,
                        "--chance",
                        chance,
                        "--n-permutations",
                        n_permutations,
                        "--cluster-alpha",
                        cluster_alpha,
                        "--out-time",
                        out_dir / "inference_time.csv",
                        "--out-clusters",
                        out_dir / "inference_clusters.csv",
                    ),
                )
            )

        if calibration:
            commands.append(
                _command(
                    f"calibration:{variant.name}",
                    (
                        python_executable,
                        "-m",
                        "reptrace.calibration",
                        aggregate_csv,
                        calibration_dir / "*_calibration_bins.csv",
                        "--out-report",
                        out_dir / "calibration_report.md",
                        "--out-bins",
                        out_dir / "reliability_bins.csv",
                    ),
                )
            )

    return commands


def format_command(command: Sequence[str]) -> str:
    """Return a shell-copyable rendering of a command without executing it."""

    return shlex.join(command)


def run_commands(commands: Sequence[SweepCommand], *, dry_run: bool = False) -> None:
    """Print and optionally execute the sweep commands."""

    for command in commands:
        print(f"[{command.label}] {format_command(command.argv)}")
        if not dry_run:
            subprocess.run(command.argv, check=True)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the predefined all-subject NOD animate/inanimate benchmark variants "
            "with matched reporting, inference, and calibration outputs."
        )
    )
    parser.add_argument("--benchmarks-dir", type=Path, default=Path("benchmarks"), help="Directory containing benchmark manifest CSVs.")
    parser.add_argument("--results-root", type=Path, default=Path("results"), help="Directory under which variant output directories are created.")
    parser.add_argument("--variant", action="append", choices=tuple(DEFAULT_VARIANT_BY_NAME), help="Variant name to run. Repeat to run multiple variants; omit to run all.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used for subcommands.")
    parser.add_argument("--chance", type=float, default=0.5)
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--cluster-alpha", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true", help="Forward --resume to each benchmark run.")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands that would run, but do not execute them.")
    parser.add_argument("--skip-validation", action="store_true", help="Do not run manifest validation commands.")
    parser.add_argument("--skip-report", action="store_true", help="Do not generate per-variant Markdown reports.")
    parser.add_argument("--skip-inference", action="store_true", help="Do not run subject-level inference commands.")
    parser.add_argument("--skip-calibration", action="store_true", help="Do not generate calibration reports and reliability bins.")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _make_parser()
    args = parser.parse_args(argv)

    try:
        variants = select_variants(args.variant)
    except ValueError as exc:
        parser.error(str(exc))

    if not args.dry_run:
        for variant in variants:
            (args.results_root / variant.output_dir).mkdir(parents=True, exist_ok=True)

    commands = build_commands(
        variants,
        benchmarks_dir=args.benchmarks_dir,
        results_root=args.results_root,
        python_executable=args.python,
        chance=args.chance,
        n_permutations=args.n_permutations,
        cluster_alpha=args.cluster_alpha,
        validate=not args.skip_validation,
        report=not args.skip_report,
        inference=not args.skip_inference,
        calibration=not args.skip_calibration,
        resume=args.resume,
    )
    run_commands(commands, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

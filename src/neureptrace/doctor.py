"""Environment and configuration diagnostics for NeuRepTrace installations."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DoctorCheck:
    """One diagnostic result emitted by the doctor command."""

    name: str
    status: str
    details: str
    required: bool = True


CORE_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("scikit-learn", "sklearn"),
    ("mne", "mne"),
    ("matplotlib", "matplotlib"),
    ("PyYAML", "yaml"),
    ("joblib", "joblib"),
)

OPTIONAL_DEPENDENCIES: tuple[tuple[str, str], ...] = (
    ("xgboost", "xgboost"),
    ("torch", "torch"),
    ("pytorch-lightning", "pytorch_lightning"),
)


def _distribution_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _check_python_version() -> DoctorCheck:
    version = sys.version_info
    version_text = f"{version.major}.{version.minor}.{version.micro}"
    if (3, 11) <= (version.major, version.minor) < (3, 15):
        return DoctorCheck("python", "ok", f"Python {version_text} on {platform.platform()}")
    return DoctorCheck("python", "error", f"Python {version_text} is unsupported; NeuRepTrace requires >=3.11,<3.15.")


def _check_neureptrace_package() -> DoctorCheck:
    try:
        import neureptrace
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        return DoctorCheck("neureptrace", "error", f"Could not import neureptrace: {exc}")

    version = getattr(neureptrace, "__version__", None) or _distribution_version("neureptrace") or "unknown"
    return DoctorCheck("neureptrace", "ok", f"importable, version {version}")


def _check_dependency(distribution_name: str, module_name: str, *, required: bool) -> DoctorCheck:
    version = _distribution_version(distribution_name)
    module_available = _module_is_available(module_name)
    name = f"dependency:{distribution_name}"
    if version is not None and module_available:
        return DoctorCheck(name, "ok", f"{distribution_name} {version}", required=required)
    if module_available:
        return DoctorCheck(name, "ok", f"{module_name} is importable; distribution metadata unavailable", required=required)

    status = "error" if required else "warning"
    kind = "required" if required else "optional"
    return DoctorCheck(name, status, f"{kind} module '{module_name}' is not importable", required=required)


def _check_required_module(module_name: str) -> DoctorCheck:
    if _module_is_available(module_name):
        version = _distribution_version(module_name)
        details = f"{module_name} {version}" if version else f"{module_name} is importable"
        return DoctorCheck(f"module:{module_name}", "ok", details)
    return DoctorCheck(f"module:{module_name}", "error", f"Required module '{module_name}' is not importable")


def _check_dataset_config(path: Path, *, check_files: bool) -> DoctorCheck:
    try:
        from neureptrace.dataset_config import load_config, validate_dataset_config

        config = load_config(path)
        warnings = validate_dataset_config(config, base_dir=path.parent, check_files=check_files)
    except Exception as exc:
        return DoctorCheck(f"dataset-config:{path}", "error", str(exc))

    if warnings:
        return DoctorCheck(f"dataset-config:{path}", "warning", "valid; warnings: " + " | ".join(str(warning) for warning in warnings))
    return DoctorCheck(f"dataset-config:{path}", "ok", "valid")


def run_checks(
    *,
    include_optional: bool = True,
    required_modules: Iterable[str] = (),
    dataset_configs: Iterable[str | Path] = (),
    check_dataset_files: bool = False,
) -> list[DoctorCheck]:
    """Run environment and configuration diagnostics."""

    checks: list[DoctorCheck] = [
        _check_python_version(),
        _check_neureptrace_package(),
    ]

    checks.extend(_check_dependency(distribution_name, module_name, required=True) for distribution_name, module_name in CORE_DEPENDENCIES)

    if include_optional:
        checks.extend(_check_dependency(distribution_name, module_name, required=False) for distribution_name, module_name in OPTIONAL_DEPENDENCIES)

    seen_modules: set[str] = set()
    for module_name in required_modules:
        normalized = str(module_name).strip()
        if not normalized or normalized in seen_modules:
            continue
        seen_modules.add(normalized)
        checks.append(_check_required_module(normalized))

    for config_path in dataset_configs:
        checks.append(_check_dataset_config(Path(config_path), check_files=check_dataset_files))

    return checks


def checks_to_jsonable(checks: Sequence[DoctorCheck]) -> list[dict[str, Any]]:
    """Return a stable JSON-compatible representation of checks."""

    return [asdict(check) for check in checks]


def summarize_checks(checks: Sequence[DoctorCheck]) -> Mapping[str, int]:
    """Count checks by status."""

    summary = {"ok": 0, "warning": 0, "error": 0}
    for check in checks:
        summary[check.status] = summary.get(check.status, 0) + 1
    return summary


def _format_check(check: DoctorCheck) -> str:
    prefix = {"ok": "OK", "warning": "WARN", "error": "FAIL"}.get(check.status, check.status.upper())
    required = "required" if check.required else "optional"
    return f"[{prefix}] {check.name} ({required}): {check.details}"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for ``neureptrace doctor``."""

    parser = argparse.ArgumentParser(description="Check a NeuRepTrace installation and optional dataset configs.")
    parser.add_argument(
        "--dataset-config",
        action="append",
        default=[],
        metavar="PATH",
        help="Validate a NeuRepTrace YAML/JSON dataset config. Can be passed multiple times.",
    )
    parser.add_argument(
        "--check-dataset-files",
        action="store_true",
        help="Require files referenced by --dataset-config to exist.",
    )
    parser.add_argument(
        "--require-module",
        action="append",
        default=[],
        metavar="MODULE",
        help="Require an additional importable Python module. Can be passed multiple times.",
    )
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional ML dependency checks such as xgboost and torch.",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Return a non-zero exit status when warnings are present.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of text.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run NeuRepTrace environment diagnostics."""

    parser = build_parser()
    args = parser.parse_args(argv)

    checks = run_checks(
        include_optional=not args.skip_optional,
        required_modules=args.require_module,
        dataset_configs=args.dataset_config,
        check_dataset_files=args.check_dataset_files,
    )
    summary = summarize_checks(checks)

    if args.json:
        print(
            json.dumps(
                {
                    "summary": dict(summary),
                    "checks": checks_to_jsonable(checks),
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for check in checks:
            print(_format_check(check))
        print(f"Summary: {summary.get('ok', 0)} ok, {summary.get('warning', 0)} warning, {summary.get('error', 0)} error")

    if summary.get("error", 0):
        return 1
    if args.fail_on_warnings and summary.get("warning", 0):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

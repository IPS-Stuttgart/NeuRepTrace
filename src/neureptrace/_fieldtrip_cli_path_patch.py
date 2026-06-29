"""Runtime patch for complete FieldTrip-to-MNE CLI path options."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_NO_PATH_SENTINELS = {"", "none", "null", "false", "off", "-"}
_PATCH_MARKER = "_neureptrace_fieldtrip_cli_path_options_patched"


def _format_path_default(tokens: Sequence[Any] | None) -> str:
    if tokens is None:
        return "none"
    return ",".join(str(token) for token in tokens)


def _parse_optional_path_tokens(
    fieldtrip_mat: Any,
    value: str | Sequence[Any] | None,
    default: Sequence[Any] | None,
) -> tuple[Any, ...] | None:
    """Parse a CLI path option while allowing optional metadata paths to be disabled."""

    if value is None:
        return tuple(default) if default is not None else None
    if isinstance(value, str) and value.strip().lower() in _NO_PATH_SENTINELS:
        return None
    return fieldtrip_mat.parse_path_tokens(value, () if default is None else default)


def _parse_label_base(value: str | int | float | None) -> float | None:
    """Parse ``--label-base`` and allow ``none`` for already-normalized labels."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "null"}:
            return None
        value = text
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("label-base must be numeric or 'none'.") from exc


def _build_parser(fieldtrip_mat: Any, prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Convert FieldTrip-like MATLAB raw/trial data to MNE Epochs FIF plus metadata CSV.",
    )
    parser.add_argument("mat", type=Path)
    parser.add_argument("--epochs-out", type=Path, required=True)
    parser.add_argument("--metadata-out", type=Path)
    parser.add_argument(
        "--root-path",
        default=_format_path_default(fieldtrip_mat.DEFAULT_ROOT_PATH),
        help="Comma-separated field/index path to the FieldTrip root struct.",
    )
    parser.add_argument(
        "--trial-path",
        default=_format_path_default(fieldtrip_mat.DEFAULT_TRIAL_PATH),
        help="Comma-separated field/index path from root to the trial cell array.",
    )
    parser.add_argument(
        "--time-path",
        default=_format_path_default(fieldtrip_mat.DEFAULT_TIME_PATH),
        help="Comma-separated field/index path from root to the time cell array.",
    )
    parser.add_argument(
        "--label-path",
        default=_format_path_default(fieldtrip_mat.DEFAULT_LABEL_PATH),
        help="Comma-separated field/index path from root to channel labels.",
    )
    parser.add_argument(
        "--trialinfo-path",
        default=_format_path_default(fieldtrip_mat.DEFAULT_TRIALINFO_PATH),
        help="Comma-separated field/index path from root to trialinfo; use 'none' to disable trialinfo metadata.",
    )
    parser.add_argument(
        "--sampleinfo-path",
        default=_format_path_default(fieldtrip_mat.DEFAULT_SAMPLEINFO_PATH),
        help="Comma-separated field/index path from root to sampleinfo; use 'none' to disable sampleinfo metadata.",
    )
    parser.add_argument("--label-column", default="condition")
    parser.add_argument(
        "--label-base",
        type=_parse_label_base,
        default=1.0,
        help="Numeric offset subtracted from trialinfo labels; use 'none' for string/already-normalized labels.",
    )
    parser.add_argument("--trialinfo-column", type=int, default=0)
    parser.add_argument("--ch-type", default="grad")
    parser.add_argument(
        "--trial-axis-order",
        choices=("channel_time", "time_channel"),
        default="channel_time",
    )
    parser.add_argument("--no-trim-overlong-labels", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def install() -> None:
    """Install a CLI patch that exposes all loader path options."""

    import neureptrace.fieldtrip_mat as fieldtrip_mat

    if getattr(fieldtrip_mat.main, _PATCH_MARKER, False):
        return

    def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
        return _build_parser(fieldtrip_mat, prog=prog)

    def main(argv: Sequence[str] | None = None) -> int:
        parser = build_parser()
        args = parser.parse_args(argv)
        epochs_out, metadata_out = fieldtrip_mat.write_fieldtrip_raw_mat_epochs(
            args.mat,
            epochs_out=args.epochs_out,
            metadata_out=args.metadata_out,
            root_path=fieldtrip_mat.parse_path_tokens(
                args.root_path,
                fieldtrip_mat.DEFAULT_ROOT_PATH,
            ),
            trial_path=fieldtrip_mat.parse_path_tokens(
                args.trial_path,
                fieldtrip_mat.DEFAULT_TRIAL_PATH,
            ),
            time_path=fieldtrip_mat.parse_path_tokens(
                args.time_path,
                fieldtrip_mat.DEFAULT_TIME_PATH,
            ),
            label_path=fieldtrip_mat.parse_path_tokens(
                args.label_path,
                fieldtrip_mat.DEFAULT_LABEL_PATH,
            ),
            trialinfo_path=_parse_optional_path_tokens(
                fieldtrip_mat,
                args.trialinfo_path,
                fieldtrip_mat.DEFAULT_TRIALINFO_PATH,
            ),
            sampleinfo_path=_parse_optional_path_tokens(
                fieldtrip_mat,
                args.sampleinfo_path,
                fieldtrip_mat.DEFAULT_SAMPLEINFO_PATH,
            ),
            label_column=args.label_column,
            label_base=args.label_base,
            trialinfo_column=args.trialinfo_column,
            ch_type=args.ch_type,
            trial_axis_order=args.trial_axis_order,
            trim_overlong_labels=not args.no_trim_overlong_labels,
            overwrite=args.overwrite,
        )
        print(f"Wrote epochs: {epochs_out}")
        print(f"Wrote metadata: {metadata_out}")
        return 0

    setattr(main, _PATCH_MARKER, True)
    fieldtrip_mat.build_parser = build_parser
    fieldtrip_mat.main = main


__all__ = ["install"]

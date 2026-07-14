"""Runtime patch for complete FieldTrip-to-MNE CLI path options."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

_NO_PATH_SENTINELS = {"", "none", "null", "false", "off", "-"}
_PATCH_MARKER = "_neureptrace_fieldtrip_cli_path_options_patched"
_WRITER_PATCH_MARKER = "_neureptrace_fieldtrip_output_paths_patched"
_PARSE_PATH_PATCH_MARKER = "_neureptrace_fieldtrip_parse_path_bool_guard_patched"
_LABEL_CONFIG_PATCH_MARKER = "_neureptrace_fieldtrip_label_config_validation_patched"
_PATH_TOKEN_ERROR = "path tokens must be strings or integer indices, not boolean values."
_LABEL_BASE_ERROR = "label_base must be a finite numeric scalar or None, not a boolean value."
_LABEL_BASE_PARSE_ERROR = "label-base must be finite numeric or 'none'."
_TRIALINFO_COLUMN_ERROR = "trialinfo_column must be an integer column index, not a boolean value."
_OUTPUT_PATH_ERROR = "FieldTrip epochs and metadata output paths must be distinct."


def _format_path_default(tokens: Sequence[Any] | None) -> str:
    if tokens is None:
        return "none"
    return ",".join(str(token) for token in tokens)


def _path_tokens_contain_boolean(value: Any) -> bool:
    """Return whether a path-token object contains Python or NumPy booleans."""

    if isinstance(value, (bool, np.bool_)):
        return True
    if value is None or isinstance(value, (str, bytes)):
        return False
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_path_tokens_contain_boolean(item) for item in value.ravel(order="C"))
        return False
    try:
        iterator = iter(value)
    except TypeError:
        return False
    return any(_path_tokens_contain_boolean(item) for item in iterator)


def _materialize_path_tokens(value: Any) -> Any:
    """Materialize one-pass token iterables before validation consumes them."""

    if value is None or isinstance(value, (str, bytes, np.ndarray)):
        return value
    try:
        return tuple(value)
    except TypeError:
        return value


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


def _scalar_value_for_numeric_config(value: Any, *, message: str) -> Any:
    """Return a scalar config value while rejecting booleans and arrays."""

    if isinstance(value, np.ndarray):
        if value.ndim != 0:
            raise ValueError(message)
        value = value.item()
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(message)
    return value


def _coerce_label_base(value: Any) -> float | None:
    """Normalize FieldTrip label-base controls without bool-to-float coercion."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "null"}:
            return None
        value = text
    value = _scalar_value_for_numeric_config(value, message=_LABEL_BASE_ERROR)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_LABEL_BASE_ERROR) from exc
    if not np.isfinite(parsed):
        raise ValueError(_LABEL_BASE_ERROR)
    return parsed


def _coerce_trialinfo_column(value: Any) -> int:
    """Normalize a FieldTrip trialinfo column index without bool-as-int leakage."""

    value = _scalar_value_for_numeric_config(value, message=_TRIALINFO_COLUMN_ERROR)
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(_TRIALINFO_COLUMN_ERROR) from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0:
        raise ValueError(_TRIALINFO_COLUMN_ERROR)
    return int(parsed)


def _parse_label_base(value: str | int | float | None) -> float | None:
    """Parse ``--label-base`` and allow ``none`` for already-normalized labels."""

    try:
        return _coerce_label_base(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(_LABEL_BASE_PARSE_ERROR) from exc


def _writer_output_paths(
    epochs_out: Path | str,
    metadata_out: Path | str | None,
) -> tuple[Path, Path]:
    """Normalize writer paths and reject aliases that would overwrite epochs."""

    epochs_path = Path(epochs_out)
    metadata_path = (
        epochs_path.with_name(f"{epochs_path.stem}_metadata.csv")
        if metadata_out is None
        else Path(metadata_out)
    )
    if epochs_path.resolve(strict=False) == metadata_path.resolve(strict=False):
        raise ValueError(_OUTPUT_PATH_ERROR)
    return epochs_path, metadata_path


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


def _install_parse_path_tokens_patch(fieldtrip_mat: Any) -> None:
    """Reject boolean path tokens before Python treats them as integer indices."""

    if getattr(fieldtrip_mat.parse_path_tokens, _PARSE_PATH_PATCH_MARKER, False):
        return

    original_parse_path_tokens = fieldtrip_mat.parse_path_tokens

    def parse_path_tokens(value: Any, default: Sequence[Any]) -> tuple[Any, ...]:
        materialized = _materialize_path_tokens(value)
        if _path_tokens_contain_boolean(materialized):
            raise ValueError(_PATH_TOKEN_ERROR)
        return original_parse_path_tokens(materialized, default)

    setattr(parse_path_tokens, _PARSE_PATH_PATCH_MARKER, True)
    parse_path_tokens.__wrapped__ = original_parse_path_tokens
    fieldtrip_mat.parse_path_tokens = parse_path_tokens


def _install_label_config_patch(fieldtrip_mat: Any) -> None:
    """Reject boolean FieldTrip label controls before numeric coercion."""

    if getattr(fieldtrip_mat._metadata_from_trialinfo, _LABEL_CONFIG_PATCH_MARKER, False):
        return

    original_parse_label_base = fieldtrip_mat._parse_label_base
    original_metadata_from_trialinfo = fieldtrip_mat._metadata_from_trialinfo

    def _patched_parse_label_base(value: Any) -> float | None:
        try:
            return _coerce_label_base(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(_LABEL_BASE_PARSE_ERROR) from exc

    def _metadata_from_trialinfo(
        *,
        n_trials: int,
        trialinfo: np.ndarray | None,
        sampleinfo: np.ndarray | None,
        label_column: str,
        label_base: Any,
        trialinfo_column: Any,
    ) -> Any:
        return original_metadata_from_trialinfo(
            n_trials=n_trials,
            trialinfo=trialinfo,
            sampleinfo=sampleinfo,
            label_column=label_column,
            label_base=_coerce_label_base(label_base),
            trialinfo_column=_coerce_trialinfo_column(trialinfo_column),
        )

    setattr(_patched_parse_label_base, _LABEL_CONFIG_PATCH_MARKER, True)
    setattr(_metadata_from_trialinfo, _LABEL_CONFIG_PATCH_MARKER, True)
    _patched_parse_label_base.__wrapped__ = original_parse_label_base
    _metadata_from_trialinfo.__wrapped__ = original_metadata_from_trialinfo
    fieldtrip_mat._parse_label_base = _patched_parse_label_base
    fieldtrip_mat._metadata_from_trialinfo = _metadata_from_trialinfo


def _install_writer_path_patch(fieldtrip_mat: Any) -> None:
    """Normalize public writer paths and reject colliding outputs."""

    if getattr(fieldtrip_mat.write_fieldtrip_raw_mat_epochs, _WRITER_PATCH_MARKER, False):
        return

    original_writer = fieldtrip_mat.write_fieldtrip_raw_mat_epochs

    def write_fieldtrip_raw_mat_epochs(
        mat_path: Path | str,
        *,
        epochs_out: Path | str,
        metadata_out: Path | str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ):
        epochs_path, metadata_path = _writer_output_paths(epochs_out, metadata_out)
        return original_writer(
            mat_path,
            epochs_out=epochs_path,
            metadata_out=metadata_path,
            overwrite=overwrite,
            **kwargs,
        )

    setattr(write_fieldtrip_raw_mat_epochs, _WRITER_PATCH_MARKER, True)
    fieldtrip_mat.write_fieldtrip_raw_mat_epochs = write_fieldtrip_raw_mat_epochs


def install() -> None:
    """Install a CLI patch that exposes all loader path options."""

    import neureptrace.fieldtrip_mat as fieldtrip_mat

    _install_parse_path_tokens_patch(fieldtrip_mat)
    _install_label_config_patch(fieldtrip_mat)
    _install_writer_path_patch(fieldtrip_mat)

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

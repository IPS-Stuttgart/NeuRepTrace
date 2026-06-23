"""Protocol-4 oracle/debug LOSO alignment runner for BUSH-MEG.

This module is intentionally separated from benchmark-valid BUSH-MEG workflows.
It runs source-LOSO decoding with ``oracle_target_calibrated_alignment`` enabled,
which lets the alignment layer use scored held-out target labels/anchors.  The
outputs are therefore useful only as an upper-bound/debug analysis and must not
be mixed into zero-calibration Protocol-1/2 leaderboards.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from neureptrace.bushmeg_source_loso import run_bushmeg_source_loso
from neureptrace.decoding.source_alignment import ORACLE_TARGET_CALIBRATED_ALIGNMENT

ORACLE_ALIGNMENT_METHODS = ("procrustes", "hyperalignment", "mcca")
ORACLE_METHOD_ALIASES = {
    "ha": "hyperalignment",
    "hyper": "hyperalignment",
    "hyper_alignment": "hyperalignment",
    "multiway_cca": "mcca",
    "multiway-canonical-correlation": "mcca",
    "multiset_cca": "mcca",
    "multiset-canonical-correlation": "mcca",
    "sumcor_mcca": "mcca",
    "m_cca": "mcca",
    "m-cca": "mcca",
}
DEFAULT_PROTOCOL4_METHODS = ORACLE_ALIGNMENT_METHODS
PROTOCOL4_NAME = "oracle_target_calibrated_alignment"
PROTOCOL4_CATEGORY = 4
PROTOCOL4_NOTE = (
    "Protocol 4 oracle/debug upper bound: held-out target labels or label-derived "
    "anchors are available to the alignment transform. Do not report as "
    "zero-calibration or strict source-only benchmark performance."
)


def normalize_oracle_alignment_method(method: str) -> str:
    """Normalize public Protocol-4 oracle alignment method names."""

    normalized = str(method).strip().lower().replace("-", "_")
    normalized = ORACLE_METHOD_ALIASES.get(normalized, normalized)
    if normalized not in ORACLE_ALIGNMENT_METHODS:
        raise ValueError(
            f"Unknown Protocol-4 oracle alignment method {method!r}. "
            f"Available methods: {', '.join(ORACLE_ALIGNMENT_METHODS)}."
        )
    return normalized


def parse_oracle_methods(methods: str | Sequence[str] | None) -> tuple[str, ...]:
    """Return a deduplicated method tuple from CLI/config values."""

    if methods is None:
        raw = DEFAULT_PROTOCOL4_METHODS
    elif isinstance(methods, str):
        raw = [part.strip() for chunk in methods.split(",") for part in chunk.split() if part.strip()]
    else:
        raw = [str(item).strip() for item in methods if str(item).strip()]
    normalized = tuple(dict.fromkeys(normalize_oracle_alignment_method(item) for item in raw))
    if not normalized:
        raise ValueError("At least one Protocol-4 oracle method is required.")
    return normalized


def protocol4_oracle_overrides(
    method: str,
    *,
    anchor_mode: str = "class_mean",
    components: int | float | str = 64,
    repetition_cap: int | str | None = 16,
) -> list[str]:
    """Return source-LOSO config overrides for one oracle alignment method."""

    normalized = normalize_oracle_alignment_method(method)
    overrides = [
        f"source_loso.alignment_method={normalized}",
        f"source_loso.alignment_target_projection={ORACLE_TARGET_CALIBRATED_ALIGNMENT}",
        f"source_loso.alignment_anchor_mode={anchor_mode}",
        f"source_loso.alignment_components={components}",
    ]
    if repetition_cap is not None:
        overrides.append(f"source_loso.alignment_repetition_cap={repetition_cap}")
    return overrides


def protocol4_metadata(method: str) -> dict[str, Any]:
    """Return benchmark-hygiene metadata for Protocol-4 oracle rows."""

    normalized = normalize_oracle_alignment_method(method)
    return {
        "method": f"oracle_target_calibrated_{normalized}",
        "method_family": "oracle_target_calibrated_alignment",
        "alignment_method": normalized,
        "protocol_category": PROTOCOL4_CATEGORY,
        "protocol_name": PROTOCOL4_NAME,
        "protocol_note": PROTOCOL4_NOTE,
        "uses_source_data": True,
        "uses_source_labels": True,
        "uses_target_data": True,
        "uses_target_labels_for_fitting": True,
        "uses_target_labels_for_scoring_only": False,
        "target_data_use": "scored_heldout_target_trials",
        "target_label_use": "scored_heldout_target_labels_for_alignment_and_metrics",
        "calibration_rows_disjoint_from_evaluation": False,
        "valid_for_strict_source_only": False,
        "valid_for_zero_calibration": False,
        "valid_for_benchmark": False,
        "debug_upper_bound": True,
    }


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _enrich_frame(frame: pd.DataFrame | None, method: str) -> pd.DataFrame:
    metadata = protocol4_metadata(method)
    if frame is None or frame.empty:
        return pd.DataFrame(columns=list(metadata))
    enriched = frame.copy()
    for key, value in metadata.items():
        enriched[key] = value
    return enriched


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path)
    return pd.DataFrame()


def run_bushmeg_protocol4_oracle(
    config_path: str | Path,
    *,
    include_oracle: bool = False,
    methods: str | Sequence[str] | None = None,
    out_dir: str | Path = "results/bush_meg/protocol4_oracle",
    overrides: Sequence[str] | None = None,
    anchor_mode: str = "class_mean",
    components: int | float | str = 64,
    repetition_cap: int | str | None = 16,
) -> dict[str, pd.DataFrame]:
    """Run Protocol-4 BUSH-MEG oracle/debug alignment analyses.

    Parameters
    ----------
    include_oracle:
        Must be ``True``. This explicit gate prevents accidental inclusion of
        Protocol-4 target-label leakage in normal BUSH-MEG benchmarks.
    """

    if not include_oracle:
        raise ValueError(
            "Protocol-4 oracle alignment is a debug upper bound and requires "
            "include_oracle=True or the CLI flag --include-oracle."
        )

    config = Path(config_path)
    output_root = Path(out_dir)
    requested_methods = parse_oracle_methods(methods)
    user_overrides = list(overrides or [])

    summary_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    inner_frames: list[pd.DataFrame] = []
    metadata_rows: list[dict[str, Any]] = []

    for method in requested_methods:
        method_metadata = protocol4_metadata(method)
        method_name = str(method_metadata["method"])
        method_dir = output_root / "methods" / method_name
        method_dir.mkdir(parents=True, exist_ok=True)

        summary_path = method_dir / "summary.csv"
        inner_path = method_dir / "inner_cv.csv"
        predictions_path = method_dir / "predictions.csv"
        status_path = method_dir / "status.json"

        status_payload = {
            **method_metadata,
            "status": "running",
            "config_path": str(config),
            "summary_csv": str(summary_path),
            "inner_cv_csv": str(inner_path),
            "predictions_csv": str(predictions_path),
        }
        status_path.write_text(json.dumps(status_payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

        method_overrides = [
            *user_overrides,
            *protocol4_oracle_overrides(
                method,
                anchor_mode=anchor_mode,
                components=components,
                repetition_cap=repetition_cap,
            ),
        ]

        summary = run_bushmeg_source_loso(
            config,
            overrides=method_overrides,
            out_path=summary_path,
            inner_cv_out_path=inner_path,
            predictions_out_path=predictions_path,
        )

        enriched_summary = _enrich_frame(summary, method)
        enriched_predictions = _enrich_frame(_read_csv_if_exists(predictions_path), method)
        enriched_inner = _enrich_frame(_read_csv_if_exists(inner_path), method)

        _write_frame(summary_path, enriched_summary)
        if not enriched_predictions.empty:
            _write_frame(predictions_path, enriched_predictions)
        if not enriched_inner.empty:
            _write_frame(inner_path, enriched_inner)

        metadata_rows.append(
            {
                **method_metadata,
                "status": "evaluated",
                "skip_reason": "",
                "n_summary_rows": int(len(enriched_summary)),
                "n_prediction_rows": int(len(enriched_predictions)),
                "summary_csv": str(summary_path),
                "inner_cv_csv": str(inner_path),
                "predictions_csv": str(predictions_path),
            }
        )
        status_path.write_text(
            json.dumps({**status_payload, "status": "evaluated"}, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        summary_frames.append(enriched_summary)
        prediction_frames.append(enriched_predictions)
        inner_frames.append(enriched_inner)

    summary_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    predictions_all = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    inner_all = pd.concat(inner_frames, ignore_index=True) if inner_frames else pd.DataFrame()
    method_metadata = pd.DataFrame(metadata_rows)

    _write_frame(output_root / "summary.csv", summary_all)
    _write_frame(output_root / "predictions.csv", predictions_all)
    _write_frame(output_root / "inner_cv.csv", inner_all)
    _write_frame(output_root / "method_metadata.csv", method_metadata)
    (output_root / "provenance.json").write_text(
        json.dumps(
            {
                "config_path": str(config),
                "methods": list(requested_methods),
                "protocol_category": PROTOCOL4_CATEGORY,
                "protocol_name": PROTOCOL4_NAME,
                "protocol_note": PROTOCOL4_NOTE,
                "outputs": {
                    "summary": str(output_root / "summary.csv"),
                    "predictions": str(output_root / "predictions.csv"),
                    "inner_cv": str(output_root / "inner_cv.csv"),
                    "method_metadata": str(output_root / "method_metadata.csv"),
                },
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "summary": summary_all,
        "predictions": predictions_all,
        "inner_cv": inner_all,
        "method_metadata": method_metadata,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run BUSH-MEG Protocol-4 oracle/debug target-calibrated alignment. "
            "This is an upper bound and is not valid for zero-calibration benchmark claims."
        )
    )
    parser.add_argument("config", type=Path, help="BUSH-MEG source_loso config file.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/bush_meg/protocol4_oracle"))
    parser.add_argument("--methods", default=",".join(DEFAULT_PROTOCOL4_METHODS))
    parser.add_argument("--include-oracle", action="store_true", help="Required safety gate for Protocol-4 target-label leakage.")
    parser.add_argument("--alignment-anchor-mode", default="class_mean")
    parser.add_argument("--alignment-components", default="64")
    parser.add_argument("--alignment-repetition-cap", default="16")
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Extra dotted config override passed to source_loso.")
    args = parser.parse_args(argv)

    repetition_cap: str | None = args.alignment_repetition_cap
    if str(repetition_cap).strip().lower() in {"", "none", "null"}:
        repetition_cap = None

    outputs = run_bushmeg_protocol4_oracle(
        args.config,
        include_oracle=args.include_oracle,
        methods=args.methods,
        out_dir=args.out_dir,
        overrides=args.overrides,
        anchor_mode=args.alignment_anchor_mode,
        components=args.alignment_components,
        repetition_cap=repetition_cap,
    )
    summary = outputs["summary"]
    if summary.empty:
        print(f"Wrote empty Protocol-4 oracle summary to {args.out_dir / 'summary.csv'}")
        return 0
    if "balanced_accuracy" in summary.columns:
        print(
            "Protocol-4 oracle/debug mean balanced accuracy: "
            f"{float(pd.to_numeric(summary['balanced_accuracy'], errors='coerce').mean()):.6f}"
        )
    print(f"Wrote Protocol-4 oracle/debug outputs to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

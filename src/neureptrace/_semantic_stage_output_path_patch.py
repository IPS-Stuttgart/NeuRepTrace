"""Reject colliding semantic-stage output destinations."""

from __future__ import annotations

import importlib
from functools import wraps
from pathlib import Path
from typing import Any

_PATCH_MARKER = "_neureptrace_semantic_stage_output_path_patch_installed"


def _output_paths(
    out_time: Path | str | None,
    out_stages: Path | str | None,
    out_report: Path | str | None,
) -> tuple[Path | None, Path | None, Path | None]:
    """Normalize semantic-stage outputs and reject colliding destinations."""

    paths = (
        None if out_time is None else Path(out_time),
        None if out_stages is None else Path(out_stages),
        None if out_report is None else Path(out_report),
    )
    labels = ("time summary", "stage intervals", "report")
    destinations: dict[Path, str] = {}
    for label, path in zip(labels, paths, strict=True):
        if path is None:
            continue
        destination = path.resolve(strict=False)
        previous = destinations.get(destination)
        if previous is not None:
            raise ValueError(
                "Semantic-stage output paths must be distinct; "
                f"{previous} and {label} both resolve to {destination}."
            )
        destinations[destination] = label
    return paths


def install() -> None:
    """Patch semantic-stage analysis to reject output-path collisions."""

    semantic_stages = importlib.import_module("neureptrace.semantic_stages")
    original_analyze = semantic_stages.analyze_semantic_stages
    if getattr(original_analyze, _PATCH_MARKER, False):
        return

    @wraps(original_analyze)
    def analyze_semantic_stages(
        state_trace_csvs: list[Path],
        *,
        posterior_threshold: float = 0.6,
        match_threshold: float = 0.6,
        min_duration: float = 0.04,
        out_time: Path | str | None = None,
        out_stages: Path | str | None = None,
        out_report: Path | str | None = None,
    ) -> tuple[Any, Any, str | None]:
        out_time, out_stages, out_report = _output_paths(out_time, out_stages, out_report)
        return original_analyze(
            state_trace_csvs,
            posterior_threshold=posterior_threshold,
            match_threshold=match_threshold,
            min_duration=min_duration,
            out_time=out_time,
            out_stages=out_stages,
            out_report=out_report,
        )

    setattr(analyze_semantic_stages, _PATCH_MARKER, True)
    semantic_stages.analyze_semantic_stages = analyze_semantic_stages


__all__ = ["install"]

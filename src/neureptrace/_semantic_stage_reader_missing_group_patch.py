"""Preserve missing semantic-stage group identifiers while reading state traces."""

from __future__ import annotations

import importlib
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_semantic_stage_reader_missing_group_patch_installed"


def install() -> None:
    """Keep missing decoder and emission-mode values nullable after CSV loading."""

    semantic_stages = importlib.import_module("neureptrace.semantic_stages")
    if getattr(semantic_stages, _PATCH_MARKER, False):
        return

    original_read_state_traces = semantic_stages.read_state_traces

    @wraps(original_read_state_traces)
    def read_state_traces(csv_paths: list[Path]) -> pd.DataFrame:
        if not csv_paths:
            raise ValueError("At least one state trace CSV path is required.")

        frames: list[pd.DataFrame] = []
        for csv_path in csv_paths:
            frame = pd.read_csv(csv_path)
            missing = [
                column
                for column in ("time", "viterbi_class")
                if column not in frame.columns
            ]
            if missing:
                raise ValueError(f"{csv_path} is missing required columns: {missing}")
            semantic_stages._coerce_finite_numeric_column(
                frame,
                "time",
                source=csv_path,
            )
            columns = semantic_stages.posterior_columns(frame)
            semantic_stages._validate_posterior_frame(
                frame,
                columns,
                source=csv_path,
            )
            if "subject" not in frame.columns:
                frame["subject"] = csv_path.stem
            if "decoder" not in frame.columns:
                frame["decoder"] = "decoder"
            if "emission_mode" not in frame.columns:
                frame["emission_mode"] = "calibrated"
            if "sequence_id" not in frame.columns:
                if "sample_index" in frame.columns:
                    frame["sequence_id"] = frame["sample_index"]
                else:
                    frame["sequence_id"] = np.arange(len(frame))

            frame["subject"] = frame["subject"].astype(str)
            # Pandas' nullable string dtype preserves missing values as ``pd.NA``.
            # Plain ``astype(str)`` turns them into the ordinary identifier
            # ``"nan"``, which defeats the missing-group preservation patch.
            frame["decoder"] = frame["decoder"].astype("string")
            frame["emission_mode"] = frame["emission_mode"].astype("string")

            if "source_file" not in frame.columns:
                frame["source_file"] = csv_path.name
            else:
                frame["source_file"] = frame["source_file"].fillna(csv_path.name)
            if "source_path" not in frame.columns:
                frame["source_path"] = str(csv_path)
            else:
                frame["source_path"] = frame["source_path"].fillna(str(csv_path))
            frame["source_file"] = frame["source_file"].astype(str)
            frame["source_path"] = frame["source_path"].astype(str)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    semantic_stages.read_state_traces = read_state_traces
    setattr(semantic_stages, _PATCH_MARKER, True)


__all__ = ["install"]

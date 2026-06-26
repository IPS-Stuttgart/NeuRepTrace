"""Runtime compatibility patch for CSV group-column feature leakage.

CSV feature-matrix splits can declare label and group columns that are metadata,
not model features. The canonical loader already removes the label column before
selecting numeric feature columns, but numeric group/run IDs were left in the
feature matrix. This patch removes both declared metadata columns and preserves
inline metadata when no separate metadata CSV is present.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

_PATCH_MARKER = "_neureptrace_dataset_spec_csv_group_column_patch_installed"


def _unique_present_columns(columns: Iterable[str | None], frame: pd.DataFrame) -> list[str]:
    selected: list[str] = []
    for column in columns:
        if column is None or column not in frame.columns or column in selected:
            continue
        selected.append(column)
    return selected


def install() -> None:
    """Install CSV feature-matrix metadata-column exclusion."""

    from neureptrace import dataset_spec

    if getattr(dataset_spec, _PATCH_MARKER, False):
        return

    def _load_csv_feature_matrix(resolved: Any) -> Any:
        if not resolved.data_path.is_file():
            raise FileNotFoundError(f"Feature CSV not found: {resolved.data_path}")
        frame = pd.read_csv(resolved.data_path)
        metadata = None
        if resolved.metadata_path is not None and resolved.metadata_path.is_file():
            metadata = pd.read_csv(resolved.metadata_path)
        labels = None
        if resolved.label_column is not None and resolved.label_column in frame.columns:
            labels = frame[resolved.label_column].to_numpy()

        metadata_columns = _unique_present_columns(
            (resolved.label_column, resolved.group_column), frame
        )
        if metadata is None and metadata_columns:
            metadata = frame.loc[:, metadata_columns].copy()
        feature_frame = frame.drop(columns=metadata_columns) if metadata_columns else frame
        numeric = feature_frame.select_dtypes(include=[np.number])
        data = numeric.to_numpy(dtype=float)[:, :, np.newaxis]
        return dataset_spec.TrialDataset(
            data=data,
            times=np.array([0.0]),
            labels=labels,
            metadata=metadata,
            channels=tuple(numeric.columns),
            subject=resolved.subject,
            split=resolved.split,
            source_path=resolved.data_path,
        )

    _load_csv_feature_matrix.__doc__ = dataset_spec._load_csv_feature_matrix.__doc__
    dataset_spec._load_csv_feature_matrix = _load_csv_feature_matrix
    setattr(dataset_spec, _PATCH_MARKER, True)

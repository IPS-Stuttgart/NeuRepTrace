"""Prevent source-quantile bin-index overflow for large quantile grids."""

from __future__ import annotations

from functools import wraps

import numpy as np

_PATCH_MARKER = "_neureptrace_source_quantile_bin_dtype_patch_installed"


def _bin_index_dtype(max_index: int) -> np.dtype:
    if max_index <= np.iinfo(np.int16).max:
        return np.dtype(np.int16)
    if max_index <= np.iinfo(np.int32).max:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def install() -> None:
    """Install dynamic bin-index dtype selection for source-quantile bins."""

    from neureptrace.decoding import source_quantile

    original = source_quantile.apply_source_quantile_bins
    if getattr(original, _PATCH_MARKER, False):
        return

    @wraps(original)
    def apply_source_quantile_bins(features, *, bin_edges):
        matrix = source_quantile._matrix(features, name="features")
        edges = np.asarray(bin_edges, dtype=float)
        if edges.ndim != 2:
            raise ValueError("bin_edges must be a two-dimensional matrix.")
        if matrix.shape[1] != edges.shape[1]:
            raise ValueError("features width must match source quantile bin edges.")
        dtype = _bin_index_dtype(edges.shape[0])
        output = np.zeros(matrix.shape, dtype=dtype)
        for column in range(matrix.shape[1]):
            output[:, column] = np.searchsorted(edges[:, column], matrix[:, column], side="right").astype(dtype, copy=False)
        return output

    setattr(apply_source_quantile_bins, _PATCH_MARKER, True)
    source_quantile.apply_source_quantile_bins = apply_source_quantile_bins


__all__ = ["install"]

"""Sensor-name selection and deterministic MEG sensor-plane projections."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module

from neureptrace.meg.fieldtrip_struct import get_data_field, get_trial_signal, unwrap_singleton, value_to_string

DEFAULT_OCCIPITAL_PATTERN = r"^M[LRZ]O"
DEFAULT_PROJECTION_REFERENCE_PATTERN = r"^M"
DEFAULT_SENSOR_POSITION_UNIT = "auto"
DEFAULT_MIN_REFERENCE_AXIS_PROJECTION = 0.05
_SENSOR_POSITION_UNIT_SCALE_TO_MM = {"m": 1000.0, "cm": 10.0, "mm": 1.0}
_PROJECTION_EPSILON = 1e-12


@dataclass(frozen=True)
class SensorProjection:
    """Deterministic 2D projection basis for MEG sensor coordinates."""

    center: np.ndarray
    axes: np.ndarray
    normal: np.ndarray | None
    reference_projection_norms: tuple[float, ...]


def _get_struct_field(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        return value[field_name]
    if isinstance(value, np.void):
        return value[field_name]
    if isinstance(value, np.ndarray) and value.dtype.names:
        return value[field_name]
    raise TypeError(f"Cannot read field {field_name!r} from {type(value).__name__}.")


def _label_to_string(label: Any) -> str:
    return value_to_string(label)


def _normalize_sensor_position_unit(unit: Any) -> str:
    unit_text = value_to_string(unit).strip().lower()
    canonical = {
        "m": "m",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
        "cm": "cm",
        "centimeter": "cm",
        "centimeters": "cm",
        "centimetre": "cm",
        "centimetres": "cm",
        "mm": "mm",
        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",
    }
    if unit_text not in canonical:
        raise ValueError(f"sensor_position_unit must be 'auto', 'm', 'cm', 'mm', or a common spelled-out equivalent; got {unit!r}.")
    return canonical[unit_text]


def get_channel_names(data: Any, n_channels: int | None = None) -> list[str]:
    """Return channel names from a FieldTrip-like MATLAB structure."""

    labels = np.asarray(get_data_field(data, "label"), dtype=object).ravel()
    if n_channels is not None:
        labels = labels[:n_channels].ravel()
    return [_label_to_string(label) for label in labels]


def get_channel_positions(data: Any, n_channels: int | None = None) -> np.ndarray:
    """Return unscaled channel positions from ``data.grad.chanpos``."""

    grad = get_data_field(data, "grad")
    chanpos = unwrap_singleton(_get_struct_field(grad, "chanpos"))
    positions = np.asarray(chanpos, dtype=float)
    return positions if n_channels is None else positions[:n_channels]


def get_channel_position_unit(data: Any) -> str | None:
    """Return the unit stored in ``data.grad.unit``, or ``None`` if absent."""

    try:
        grad = get_data_field(data, "grad")
        unit = _get_struct_field(grad, "unit")
    except (KeyError, TypeError, ValueError):
        return None

    unit_text = value_to_string(unit).strip()
    if not unit_text:
        return None
    return _normalize_sensor_position_unit(unit_text)


def resolve_sensor_position_unit(data: Any, sensor_position_unit: str = DEFAULT_SENSOR_POSITION_UNIT) -> str:
    """Resolve the unit for ``data.grad.chanpos``."""

    if sensor_position_unit is None or value_to_string(sensor_position_unit).strip().lower() == "auto":
        return get_channel_position_unit(data) or "mm"
    return _normalize_sensor_position_unit(sensor_position_unit)


def get_channel_positions_mm(data: Any, n_channels: int | None = None, *, sensor_position_unit: str = DEFAULT_SENSOR_POSITION_UNIT) -> np.ndarray:
    """Return channel positions converted to millimetres."""

    positions = get_channel_positions(data, n_channels)
    unit = resolve_sensor_position_unit(data, sensor_position_unit)
    return positions * _SENSOR_POSITION_UNIT_SCALE_TO_MM[unit]


def select_channels(data: Any, location_pattern: str = DEFAULT_OCCIPITAL_PATTERN) -> list[int]:
    """Select channels whose labels match ``location_pattern``."""

    n_channels = get_trial_signal(data, 0).shape[0]
    pattern = re.compile(location_pattern)
    channel_names = get_channel_names(data, n_channels)
    return [index for index, channel_name in enumerate(channel_names) if pattern.search(channel_name)]


def _check_positions_2d_array(positions) -> np.ndarray:
    array = np.asarray(positions, dtype=float)
    if array.ndim != 2:
        raise ValueError("positions must be a 2D array with shape (n_sensors, n_coordinates).")
    if array.shape[0] < 3:
        raise ValueError("At least three sensor positions are required for 2D projection.")
    if array.shape[1] < 2:
        raise ValueError("Sensor positions must have at least two coordinate dimensions.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Sensor positions must be finite.")
    return array


def _pca_plane_normal(centered: np.ndarray) -> np.ndarray | None:
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    scale = max(float(singular_values[0]) if singular_values.size else 0.0, 1.0)
    if singular_values.size < 2 or singular_values[1] <= scale * _PROJECTION_EPSILON:
        raise ValueError("At least two non-collinear sensor positions are required for 2D projection.")
    if centered.shape[1] != 3 or vt.shape[0] < 3:
        return None
    return vt[2]


def _validated_min_reference_projection(min_reference_projection: float) -> float:
    value = float(min_reference_projection)
    if not np.isfinite(value) or value < 0.0 or value >= 1.0:
        raise ValueError("min_reference_axis_projection must be finite and in [0, 1).")
    return max(value, _PROJECTION_EPSILON)


def _anchored_plane_axes(normal: np.ndarray, *, min_reference_projection: float = DEFAULT_MIN_REFERENCE_AXIS_PROJECTION) -> np.ndarray:
    min_reference_projection = _validated_min_reference_projection(min_reference_projection)
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= _PROJECTION_EPSILON:
        raise ValueError("Could not determine a stable sensor projection plane normal.")
    normal = normal / normal_norm

    anchored_axes: list[np.ndarray] = []
    for reference in np.eye(3):
        candidate = reference - float(np.dot(reference, normal)) * normal
        for axis in anchored_axes:
            candidate = candidate - float(np.dot(candidate, axis)) * axis

        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm <= min_reference_projection:
            continue

        candidate = candidate / candidate_norm
        if float(np.dot(candidate, reference)) < 0.0:
            candidate = -candidate
        anchored_axes.append(candidate)
        if len(anchored_axes) == 2:
            return np.column_stack(anchored_axes)

    raise ValueError("Could not anchor two projected axes to the sensor coordinate frame.")


def _signed_pca_axes(centered: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axes = vt[:2].T
    for column_index in range(axes.shape[1]):
        axis = axes[:, column_index]
        largest_component = int(np.argmax(np.abs(axis)))
        if axis[largest_component] < 0:
            axes[:, column_index] = -axis
    return axes


def fit_sensor_projection(positions, *, min_reference_projection: float = DEFAULT_MIN_REFERENCE_AXIS_PROJECTION) -> SensorProjection:
    """Fit a reusable deterministic 2D projection basis for sensor positions."""

    positions = _check_positions_2d_array(positions)
    center = np.mean(positions, axis=0)
    centered = positions - center
    if centered.shape[1] == 2:
        return SensorProjection(center=center, axes=np.eye(2), normal=None, reference_projection_norms=(1.0, 1.0))

    normal = _pca_plane_normal(centered)
    if normal is None:
        axes = _signed_pca_axes(centered)
        return SensorProjection(center=center, axes=axes, normal=None, reference_projection_norms=tuple(float("nan") for _ in range(centered.shape[1])))

    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= _PROJECTION_EPSILON:
        raise ValueError("Could not determine a stable sensor projection plane normal.")
    normal = normal / normal_norm
    axes = _anchored_plane_axes(normal, min_reference_projection=min_reference_projection)
    reference_projection_norms = tuple(float(np.linalg.norm(reference - float(np.dot(reference, normal)) * normal)) for reference in np.eye(3))
    return SensorProjection(center=center, axes=axes, normal=normal, reference_projection_norms=reference_projection_norms)


def apply_sensor_projection(positions, projection: SensorProjection) -> np.ndarray:
    """Apply a fitted sensor projection to positions in the same coordinate frame."""

    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2:
        raise ValueError("positions must be a 2D array with shape (n_sensors, n_coordinates).")
    if positions.shape[1] != projection.center.shape[0]:
        raise ValueError("positions and projection center must have the same coordinate dimension.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Sensor positions must be finite.")
    return (positions - projection.center) @ projection.axes


def project_sensor_positions(positions) -> np.ndarray:
    """Project sensor positions to a deterministic coordinate-anchored 2D plane."""

    projection = fit_sensor_projection(positions)
    return apply_sensor_projection(positions, projection)


def _reference_positions_for_projection(data, all_positions, selected_positions, projection_reference_pattern):
    if projection_reference_pattern is None:
        return selected_positions

    reference_indices = select_channels(data, projection_reference_pattern)
    if not reference_indices:
        raise ValueError(f"No channels matched projection reference pattern: {projection_reference_pattern}")
    return np.take(all_positions, reference_indices, axis=0)


def project_channel_positions(
    data,
    channel_indices,
    *,
    sensor_position_unit: str = DEFAULT_SENSOR_POSITION_UNIT,
    projection_reference_pattern: str | None = DEFAULT_PROJECTION_REFERENCE_PATTERN,
    min_reference_axis_projection: float = DEFAULT_MIN_REFERENCE_AXIS_PROJECTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Return selected channel positions and their common-frame 2D projection."""

    n_channels = get_trial_signal(data, 0).shape[0]
    all_positions = get_channel_positions_mm(data, n_channels, sensor_position_unit=sensor_position_unit)
    channel_indices = np.asarray(channel_indices, dtype=int)
    selected_positions = np.take(all_positions, channel_indices, axis=0)
    reference_positions = _reference_positions_for_projection(data, all_positions, selected_positions, projection_reference_pattern)
    projection = fit_sensor_projection(reference_positions, min_reference_projection=min_reference_axis_projection)
    return selected_positions, apply_sensor_projection(selected_positions, projection)


def delaunay_edges(coords2d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Delaunay edge indices, vectors, and pseudoinverse for 2D sensor coords."""

    if len(coords2d) < 3:
        raise ValueError("At least three sensor positions are required.")

    triangulation = Delaunay(coords2d)
    edges = set()
    for simplex in triangulation.simplices:
        for first, second in ((0, 1), (1, 2), (2, 0)):
            edges.add(tuple(sorted((int(simplex[first]), int(simplex[second])))))

    edge_indices = np.array(sorted(edges), dtype=int)
    edge_vectors = coords2d[edge_indices[:, 1]] - coords2d[edge_indices[:, 0]]
    return edge_indices, edge_vectors, np.linalg.pinv(edge_vectors)

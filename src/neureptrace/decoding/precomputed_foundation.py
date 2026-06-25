"""Precomputed foundation-feature tables for frozen M/EEG probes.

This module covers the dependency-light foundation-model path: external encoders
such as BENDR, LaBraM, EEGPT, CBraMod, or project-local models can export a table
of trial features, and NeuRepTrace can align those rows to fold-local metadata and
train ordinary source-label probes without importing the upstream model package.

The loader/probe API intentionally does not accept held-out target labels.  The
``feature_fit_scope`` metadata records how the external features were produced so
strict source-only, unlabeled target-adaptive, and calibrated feature extractors
remain distinguishable in reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression

FEATURE_FIT_SCOPES = (
    "external_frozen",
    "source_only",
    "source_plus_unlabeled_target",
    "target_calibrated",
    "oracle_target_included",
)
FEATURE_FIT_SCOPE_PROTOCOL = {
    "external_frozen": "strict_source_only_frozen_external_features",
    "source_only": "strict_source_only_precomputed_features",
    "source_plus_unlabeled_target": "unlabeled_target_adaptive_precomputed_features",
    "target_calibrated": "target_calibrated_precomputed_features",
    "oracle_target_included": "oracle_target_included_precomputed_features",
}
FEATURE_FIT_SCOPE_CATEGORY = {
    "external_frozen": 1,
    "source_only": 1,
    "source_plus_unlabeled_target": 2,
    "target_calibrated": 3,
    "oracle_target_included": 4,
}
FEATURE_FIT_SCOPE_USES_TARGET_FEATURES = {
    "external_frozen": False,
    "source_only": False,
    "source_plus_unlabeled_target": True,
    "target_calibrated": True,
    "oracle_target_included": True,
}
FEATURE_FIT_SCOPE_USES_TARGET_LABELS = {
    "external_frozen": False,
    "source_only": False,
    "source_plus_unlabeled_target": False,
    "target_calibrated": True,
    "oracle_target_included": True,
}
CSV_EXTENSIONS = {".csv", ".tsv", ".txt"}
NUMPY_EXTENSIONS = {".npy", ".npz"}
DEFAULT_ROW_ID_COLUMN = "row_id"
DEFAULT_FEATURES_KEY = "features"
DEFAULT_ROW_ID_KEY = "row_ids"


@dataclass(frozen=True, slots=True)
class PrecomputedFoundationFeatureTable:
    """Immutable feature table keyed by row id."""

    features: np.ndarray
    row_ids: tuple[Any, ...]
    feature_names: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        matrix = _feature_matrix(self.features, name="features")
        row_ids = tuple(self.row_ids)
        if matrix.shape[0] != len(row_ids):
            raise ValueError(f"features and row_ids must have the same number of rows: {matrix.shape[0]} != {len(row_ids)}.")
        feature_names = tuple(self.feature_names)
        if matrix.shape[1] != len(feature_names):
            raise ValueError(f"features and feature_names must have the same number of columns: {matrix.shape[1]} != {len(feature_names)}.")
        _validate_unique_row_ids(row_ids)
        object.__setattr__(self, "features", matrix.astype(np.float32, copy=False))
        object.__setattr__(self, "row_ids", row_ids)
        object.__setattr__(self, "feature_names", feature_names)

    @property
    def n_rows(self) -> int:
        """Number of rows in the feature table."""

        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        """Number of feature columns in the table."""

        return int(self.features.shape[1])

    def row_index(self) -> dict[Any, int]:
        """Return a row-id to row-index mapping."""

        return {row_id: index for index, row_id in enumerate(self.row_ids)}


@dataclass(frozen=True, slots=True)
class PrecomputedFoundationProbeResult:
    """Source-label probe fitted on precomputed foundation features."""

    train_features: np.ndarray
    test_features: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray | None
    classes: np.ndarray
    classifier: BaseEstimator
    metadata: dict[str, Any] = field(default_factory=dict)


# pylint: disable-next=too-many-arguments,too-many-locals

def load_precomputed_foundation_features(
    path: str | Path,
    *,
    features_key: str = DEFAULT_FEATURES_KEY,
    row_id_key: str = DEFAULT_ROW_ID_KEY,
    row_id_column: str = DEFAULT_ROW_ID_COLUMN,
    feature_columns: Sequence[str] | str | None = None,
    feature_prefix: str | None = None,
    allow_pickle: bool = False,
    delimiter: str | None = None,
    feature_fit_scope: str | None = "external_frozen",
    source_model: str = "external",
) -> PrecomputedFoundationFeatureTable:
    """Load precomputed foundation features from ``.npy``, ``.npz``, or CSV/TSV.

    Parameters
    ----------
    path:
        Feature table path.  ``.npz`` files should contain a feature matrix and,
        preferably, row ids.  ``.npy`` files contain only a matrix and receive
        sequential integer row ids.  CSV/TSV files use ``row_id_column`` and
        numeric feature columns.
    features_key:
        Key for the feature matrix in ``.npz`` files.
    row_id_key:
        Key for row ids in ``.npz`` files.  If absent, sequential row ids are used.
    row_id_column:
        Row-id column for CSV/TSV files.  If absent, sequential row ids are used.
    feature_columns:
        Explicit CSV/TSV feature columns.  A comma-separated string is accepted.
    feature_prefix:
        Optional CSV/TSV prefix used to select feature columns, for example
        ``"feat_"``.  Ignored when ``feature_columns`` is provided.
    allow_pickle:
        Forwarded to NumPy loading for projects that intentionally store object
        arrays.  The default is false.
    delimiter:
        Explicit delimiter for text tables.  Defaults to tab for ``.tsv`` and
        comma otherwise.
    feature_fit_scope:
        Declaration of how the external feature extractor was fit.  This controls
        protocol metadata but does not change the loaded matrix.
    source_model:
        Human-readable model/source name, for example ``"BENDR"`` or ``"LaBraM"``.
    """

    table_path = Path(path)
    suffix = table_path.suffix.lower()
    scope = normalize_feature_fit_scope(feature_fit_scope)
    if suffix == ".npz":
        features, row_ids, feature_names = _load_npz_features(table_path, features_key=features_key, row_id_key=row_id_key, allow_pickle=allow_pickle)
    elif suffix == ".npy":
        features = np.load(table_path, allow_pickle=allow_pickle)
        features = _feature_matrix(features, name="features")
        row_ids = tuple(range(features.shape[0]))
        feature_names = tuple(f"foundation_{index}" for index in range(features.shape[1]))
    elif suffix in CSV_EXTENSIONS:
        features, row_ids, feature_names = _load_text_features(
            table_path,
            row_id_column=row_id_column,
            feature_columns=feature_columns,
            feature_prefix=feature_prefix,
            delimiter=delimiter,
        )
    else:
        raise ValueError(f"Unsupported feature table extension {suffix!r}; expected one of {sorted(NUMPY_EXTENSIONS | CSV_EXTENSIONS)}.")

    metadata = _feature_table_metadata(
        path=table_path,
        source_model=source_model,
        feature_fit_scope=scope,
        n_rows=features.shape[0],
        n_features=features.shape[1],
    )
    return PrecomputedFoundationFeatureTable(features=features, row_ids=tuple(row_ids), feature_names=tuple(feature_names), metadata=metadata)


def make_precomputed_foundation_feature_table(
    features: Sequence[Sequence[float]] | np.ndarray,
    row_ids: Sequence[Any] | np.ndarray | None = None,
    *,
    feature_names: Sequence[str] | None = None,
    feature_fit_scope: str | None = "external_frozen",
    source_model: str = "external",
) -> PrecomputedFoundationFeatureTable:
    """Create a feature table directly from in-memory arrays."""

    matrix = _feature_matrix(features, name="features")
    ids = tuple(range(matrix.shape[0])) if row_ids is None else tuple(np.asarray(row_ids, dtype=object).reshape(-1).tolist())
    if len(ids) != matrix.shape[0]:
        raise ValueError(f"row_ids must contain one value per feature row: {len(ids)} != {matrix.shape[0]}.")
    names = tuple(f"foundation_{index}" for index in range(matrix.shape[1])) if feature_names is None else tuple(str(name) for name in feature_names)
    metadata = _feature_table_metadata(
        path=None,
        source_model=source_model,
        feature_fit_scope=normalize_feature_fit_scope(feature_fit_scope),
        n_rows=matrix.shape[0],
        n_features=matrix.shape[1],
    )
    return PrecomputedFoundationFeatureTable(features=matrix, row_ids=ids, feature_names=names, metadata=metadata)


def align_precomputed_foundation_features(
    table: PrecomputedFoundationFeatureTable,
    row_ids: Sequence[Any] | np.ndarray,
) -> np.ndarray:
    """Return table rows in the requested row-id order."""

    requested = tuple(np.asarray(row_ids, dtype=object).reshape(-1).tolist())
    index = table.row_index()
    missing = [row_id for row_id in requested if row_id not in index]
    if missing:
        preview = ", ".join(repr(row_id) for row_id in missing[:5])
        raise KeyError(f"Precomputed feature table is missing {len(missing)} requested row id(s): {preview}.")
    return table.features[[index[row_id] for row_id in requested]].astype(np.float32, copy=False)


# pylint: disable-next=too-many-arguments,too-many-locals

def fit_precomputed_foundation_probe(
    *,
    feature_table: PrecomputedFoundationFeatureTable,
    train_row_ids: Sequence[Any] | np.ndarray,
    train_labels: Sequence[Any] | np.ndarray,
    test_row_ids: Sequence[Any] | np.ndarray,
    classifier: BaseEstimator | None = None,
    classifier_C: float = 1.0,
    classifier_max_iter: int = 1000,
    classifier_class_weight: str | Mapping[Any, float] | None = "balanced",
    sample_weight: Sequence[float] | np.ndarray | None = None,
) -> PrecomputedFoundationProbeResult:
    """Train a source-label probe on precomputed foundation features.

    The probe uses row ids to align source training rows and held-out target rows
    to the precomputed feature table.  It trains only from ``train_labels`` and
    intentionally has no ``target_labels`` argument.
    """

    train_ids = tuple(np.asarray(train_row_ids, dtype=object).reshape(-1).tolist())
    test_ids = tuple(np.asarray(test_row_ids, dtype=object).reshape(-1).tolist())
    labels = _label_vector(train_labels)
    if labels.shape[0] != len(train_ids):
        raise ValueError(f"train_labels must contain one value per train row id: {labels.shape[0]} != {len(train_ids)}.")
    if labels.shape[0] < 1 or np.unique(labels).shape[0] < 2:
        raise ValueError("train_labels must contain at least two classes.")
    train_features = align_precomputed_foundation_features(feature_table, train_ids)
    test_features = align_precomputed_foundation_features(feature_table, test_ids)
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    if weights is not None:
        if weights.shape[0] != labels.shape[0]:
            raise ValueError(f"sample_weight must contain one value per train row: {weights.shape[0]} != {labels.shape[0]}.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("sample_weight must contain finite non-negative values.")

    model = clone(classifier) if classifier is not None else LogisticRegression(
        C=_positive_float(classifier_C, name="classifier_C"),
        max_iter=_positive_int(classifier_max_iter, name="classifier_max_iter"),
        class_weight=classifier_class_weight,
        random_state=13,
    )
    fit_kwargs = {} if weights is None else {"sample_weight": weights}
    model.fit(train_features, labels, **fit_kwargs)
    predictions = np.asarray(model.predict(test_features))
    probabilities = _predict_probabilities_or_none(model, test_features)
    classes = np.asarray(getattr(model, "classes_", np.unique(labels)))
    metadata = _probe_metadata(feature_table, n_train_rows=len(train_ids), n_test_rows=len(test_ids), classifier=model)
    return PrecomputedFoundationProbeResult(
        train_features=train_features,
        test_features=test_features,
        predictions=predictions,
        probabilities=probabilities,
        classes=classes,
        classifier=model,
        metadata=metadata,
    )


def normalize_feature_fit_scope(value: str | None) -> str:
    """Normalize feature-extractor fit-scope aliases to protocol categories."""

    normalized = "external_frozen" if value is None else str(value).strip().lower().replace("-", "_")
    normalized = {
        "external": "external_frozen",
        "external_pretrained": "external_frozen",
        "frozen_external": "external_frozen",
        "frozen_pretrained": "external_frozen",
        "protocol_1": "source_only",
        "category_1": "source_only",
        "strict_source_only": "source_only",
        "source": "source_only",
        "train_only": "source_only",
        "source_target": "source_plus_unlabeled_target",
        "source_plus_target": "source_plus_unlabeled_target",
        "unlabeled_target": "source_plus_unlabeled_target",
        "target_adaptive": "source_plus_unlabeled_target",
        "protocol_2": "source_plus_unlabeled_target",
        "category_2": "source_plus_unlabeled_target",
        "few_shot": "target_calibrated",
        "calibrated": "target_calibrated",
        "target_calibration": "target_calibrated",
        "protocol_3": "target_calibrated",
        "category_3": "target_calibrated",
        "oracle": "oracle_target_included",
        "target_included": "oracle_target_included",
        "protocol_4": "oracle_target_included",
        "category_4": "oracle_target_included",
    }.get(normalized, normalized)
    if normalized not in FEATURE_FIT_SCOPES:
        raise ValueError(f"Unknown feature_fit_scope {value!r}. Available scopes: {', '.join(FEATURE_FIT_SCOPES)}.")
    return normalized


def _label_vector(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    try:
        items = list(values)
    except TypeError:
        items = [values]
    if any(isinstance(item, tuple) for item in items):
        vector = np.empty(len(items), dtype=object)
        vector[:] = items
        return vector
    return np.asarray(items).reshape(-1)


def _load_npz_features(path: Path, *, features_key: str, row_id_key: str, allow_pickle: bool) -> tuple[np.ndarray, tuple[Any, ...], tuple[str, ...]]:
    with np.load(path, allow_pickle=allow_pickle) as payload:
        if features_key in payload:
            features = np.asarray(payload[features_key])
        else:
            matrix_keys = [key for key in payload.files if np.asarray(payload[key]).ndim == 2]
            if len(matrix_keys) != 1:
                raise ValueError(f"NPZ file must contain key {features_key!r} or exactly one two-dimensional array; found {matrix_keys}.")
            features = np.asarray(payload[matrix_keys[0]])
        features = _feature_matrix(features, name="features")
        if row_id_key in payload:
            row_ids = tuple(np.asarray(payload[row_id_key], dtype=object).reshape(-1).tolist())
        elif "row_id" in payload:
            row_ids = tuple(np.asarray(payload["row_id"], dtype=object).reshape(-1).tolist())
        else:
            row_ids = tuple(range(features.shape[0]))
        if len(row_ids) != features.shape[0]:
            raise ValueError(f"row ids in NPZ must match feature rows: {len(row_ids)} != {features.shape[0]}.")
        if "feature_names" in payload:
            feature_names = tuple(str(value) for value in np.asarray(payload["feature_names"], dtype=object).reshape(-1).tolist())
        else:
            feature_names = tuple(f"foundation_{index}" for index in range(features.shape[1]))
    return features, row_ids, feature_names


def _load_text_features(
    path: Path,
    *,
    row_id_column: str,
    feature_columns: Sequence[str] | str | None,
    feature_prefix: str | None,
    delimiter: str | None,
) -> tuple[np.ndarray, tuple[Any, ...], tuple[str, ...]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a core dependency today
        raise ImportError("Loading CSV/TSV precomputed foundation features requires pandas.") from exc
    sep = delimiter if delimiter is not None else ("\t" if path.suffix.lower() == ".tsv" else ",")
    frame = pd.read_csv(path, sep=sep)
    if frame.empty:
        raise ValueError("Feature table must contain at least one row.")
    if row_id_column in frame.columns:
        row_ids = tuple(frame[row_id_column].astype(object).tolist())
        candidate_frame = frame.drop(columns=[row_id_column])
    else:
        row_ids = tuple(range(frame.shape[0]))
        candidate_frame = frame
    columns = _resolve_feature_columns(candidate_frame, feature_columns=feature_columns, feature_prefix=feature_prefix)
    features = candidate_frame.loc[:, list(columns)].to_numpy(dtype=float)
    return _feature_matrix(features, name="features"), row_ids, tuple(str(column) for column in columns)


def _resolve_feature_columns(frame, *, feature_columns: Sequence[str] | str | None, feature_prefix: str | None) -> tuple[str, ...]:
    if feature_columns is not None:
        if isinstance(feature_columns, str):
            columns = tuple(part.strip() for part in feature_columns.replace(";", ",").split(",") if part.strip())
        else:
            columns = tuple(str(column) for column in feature_columns)
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"Requested feature column(s) not found: {missing}.")
        return columns
    if feature_prefix:
        columns = tuple(str(column) for column in frame.columns if str(column).startswith(feature_prefix))
        if not columns:
            raise ValueError(f"No feature columns start with prefix {feature_prefix!r}.")
        return columns
    numeric_columns = tuple(str(column) for column in frame.select_dtypes(include=[np.number]).columns)
    if not numeric_columns:
        raise ValueError("No numeric feature columns found. Provide feature_columns or feature_prefix.")
    return numeric_columns


def _feature_table_metadata(*, path: Path | None, source_model: str, feature_fit_scope: str, n_rows: int, n_features: int) -> dict[str, Any]:
    return {
        "precomputed_foundation_features": True,
        "precomputed_foundation_source_model": str(source_model),
        "precomputed_foundation_path": "" if path is None else str(path),
        "precomputed_foundation_fit_scope": feature_fit_scope,
        "precomputed_foundation_protocol": FEATURE_FIT_SCOPE_PROTOCOL[feature_fit_scope],
        "precomputed_foundation_protocol_category": FEATURE_FIT_SCOPE_CATEGORY[feature_fit_scope],
        "precomputed_foundation_uses_target_features_for_feature_fit": FEATURE_FIT_SCOPE_USES_TARGET_FEATURES[feature_fit_scope],
        "precomputed_foundation_uses_target_labels_for_feature_fit": FEATURE_FIT_SCOPE_USES_TARGET_LABELS[feature_fit_scope],
        "precomputed_foundation_valid_for_strict_source_only": FEATURE_FIT_SCOPE_CATEGORY[feature_fit_scope] == 1,
        "precomputed_foundation_valid_for_unlabeled_target_adaptation": FEATURE_FIT_SCOPE_CATEGORY[feature_fit_scope] in {1, 2},
        "precomputed_foundation_debug_upper_bound": FEATURE_FIT_SCOPE_CATEGORY[feature_fit_scope] == 4,
        "precomputed_foundation_n_rows": int(n_rows),
        "precomputed_foundation_n_features": int(n_features),
    }


def _probe_metadata(feature_table: PrecomputedFoundationFeatureTable, *, n_train_rows: int, n_test_rows: int, classifier: BaseEstimator) -> dict[str, Any]:
    return {
        **feature_table.metadata,
        "precomputed_foundation_probe": True,
        "precomputed_foundation_probe_classifier": type(classifier).__name__,
        "precomputed_foundation_probe_uses_train_labels": True,
        "precomputed_foundation_probe_uses_target_labels": False,
        "precomputed_foundation_probe_n_train_rows": int(n_train_rows),
        "precomputed_foundation_probe_n_test_rows": int(n_test_rows),
    }


def _predict_probabilities_or_none(model: BaseEstimator, features: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        return _normalize_probability_rows(probabilities)
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
        return _normalize_probability_rows(exp_scores)
    return None


def _normalize_probability_rows(probabilities: np.ndarray) -> np.ndarray:
    matrix = np.asarray(probabilities, dtype=float)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("Predicted probabilities must be a finite two-dimensional array.")
    matrix = np.maximum(matrix, 0.0)
    row_sums = np.sum(matrix, axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("Predicted probability rows must have positive mass.")
    return matrix / row_sums


def _feature_matrix(values: Sequence[Sequence[float]] | np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional feature matrix.")
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one row and one feature column.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _validate_unique_row_ids(row_ids: Sequence[Any]) -> None:
    seen = set()
    duplicates = []
    for row_id in row_ids:
        try:
            hash(row_id)
        except TypeError as exc:
            raise ValueError(f"row_ids must be hashable; got {row_id!r}.") from exc
        if row_id in seen:
            duplicates.append(row_id)
        seen.add(row_id)
    if duplicates:
        preview = ", ".join(repr(row_id) for row_id in duplicates[:5])
        raise ValueError(f"row_ids must be unique; duplicate row id(s): {preview}.")


def _positive_int(value: int | str, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if not np.isfinite(parsed) or parsed % 1.0 != 0.0 or parsed < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(parsed)


def _positive_float(value: float | str, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite.")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return parsed

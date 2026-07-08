from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

import neureptrace


def _load_core_precomputed_module():
    """Load the source module under a fresh name, bypassing import-time monkeypatches."""

    module_path = Path(neureptrace.__file__).parent / "decoding" / "precomputed_foundation.py"
    spec = importlib.util.spec_from_file_location("_neureptrace_precomputed_foundation_core_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_core_loader_preserves_matrix_encoded_tuple_row_ids(tmp_path) -> None:
    module = _load_core_precomputed_module()
    path = tmp_path / "features.npz"
    features = np.asarray([[4.0, 0.0], [0.0, 4.0], [5.0, 0.0], [0.0, 5.0]], dtype=float)
    row_ids = np.asarray([[1, 0], [1, 1], [2, 0], [2, 1]], dtype=int)
    np.savez(path, features=features, row_ids=row_ids)

    table = module.load_precomputed_foundation_features(path)
    result = module.fit_precomputed_foundation_probe(
        feature_table=table,
        train_row_ids=np.asarray([[1, 0], [1, 1]], dtype=int),
        train_labels=[0, 1],
        test_row_ids=np.asarray([[2, 0], [2, 1]], dtype=int),
        classifier_C=1000.0,
    )

    assert table.row_ids == ((1, 0), (1, 1), (2, 0), (2, 1))
    assert np.allclose(result.train_features, np.asarray([[4.0, 0.0], [0.0, 4.0]], dtype=np.float32))
    assert np.allclose(result.test_features, np.asarray([[5.0, 0.0], [0.0, 5.0]], dtype=np.float32))


def test_core_probe_preserves_composite_train_labels() -> None:
    module = _load_core_precomputed_module()
    table = module.make_precomputed_foundation_feature_table(
        [[-2.0, 0.0], [-1.5, 0.2], [2.0, 0.0], [1.6, -0.2], [-1.8, 0.1], [1.8, -0.1]],
        row_ids=["s0", "s1", "s2", "s3", "t0", "t1"],
    )
    train_labels = np.asarray(
        [
            ["left", "early"],
            ["left", "early"],
            ["right", "late"],
            ["right", "late"],
        ],
        dtype=object,
    )

    result = module.fit_precomputed_foundation_probe(
        feature_table=table,
        train_row_ids=["s0", "s1", "s2", "s3"],
        train_labels=train_labels,
        test_row_ids=["t0", "t1"],
    )

    assert result.predictions.tolist() == [("left", "early"), ("right", "late")]
    assert result.classes.tolist() == [("left", "early"), ("right", "late")]
    assert result.classifier.__class__.__name__ == "DecodedLabelClassifier"

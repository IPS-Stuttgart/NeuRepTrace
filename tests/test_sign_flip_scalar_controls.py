from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from neureptrace.inference import sign_flip_time_inference
from neureptrace.paired_stats import sign_flip_p_value


def _write_time_csv(path: Path, subject: str) -> None:
    pd.DataFrame(
        {
            "subject": [subject, subject],
            "fold": [0, 0],
            "time": [0.1, 0.2],
            "accuracy": [0.55, 0.56],
            "log_loss": [0.7, 0.7],
            "brier": [0.5, 0.5],
            "ece": [0.1, 0.1],
        }
    ).to_csv(path, index=False)


def _time_csvs(tmp_path: Path) -> list[Path]:
    paths = []
    for idx in range(3):
        path = tmp_path / f"sub-{idx + 1:02d}_time_decode.csv"
        _write_time_csv(path, f"sub-{idx + 1:02d}")
        paths.append(path)
    return paths


def test_sign_flip_time_inference_rejects_array_valued_scalar_controls(tmp_path: Path):
    csv_paths = _time_csvs(tmp_path)
    cases = [
        ({"n_permutations": np.asarray(True)}, "n_permutations must be a positive integer"),
        ({"n_permutations": np.array([8])}, "n_permutations must be a positive integer"),
        ({"random_state": True}, "random_state must be a non-negative integer seed"),
        ({"random_state": np.asarray(True)}, "random_state must be a non-negative integer seed"),
        ({"random_state": np.array([7])}, "random_state must be a non-negative integer seed"),
        ({"cluster_alpha": np.asarray(True)}, "cluster_alpha must be between 0 and 1"),
        ({"cluster_alpha": np.array([0.05])}, "cluster_alpha must be between 0 and 1"),
    ]
    for kwargs, match in cases:
        params = {"n_permutations": 8, "random_state": 7, "cluster_alpha": 0.05}
        params.update(kwargs)
        with pytest.raises(ValueError, match=match):
            sign_flip_time_inference(csv_paths, **params)


def test_paired_sign_flip_rejects_array_valued_scalar_controls():
    differences = np.array([1.0, -0.5, 0.25])
    cases = [
        ({"n_permutations": np.asarray(True)}, "n_permutations must be a positive integer"),
        ({"n_permutations": np.array([8])}, "n_permutations must be a positive integer"),
        ({"random_state": True}, "random_state must be a non-negative integer seed"),
        ({"random_state": np.asarray(True)}, "random_state must be a non-negative integer seed"),
        ({"random_state": np.array([7])}, "random_state must be a non-negative integer seed"),
    ]
    for kwargs, match in cases:
        params = {"n_permutations": 8, "random_state": 7}
        params.update(kwargs)
        with pytest.raises(ValueError, match=match):
            sign_flip_p_value(differences, **params)

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neureptrace.openneuro_decode_diagnostics import aggregate_workflow_outputs


def _write_minimal_shard(root: Path, name: str, *, label_shuffle_control: object) -> Path:
    output = root / name
    decode = output / "decode"
    decode.mkdir(parents=True)

    manifest = {
        "dataset": "ds006629",
        "mode": "smoke",
        "artifact_name": f"openneuro-meg-ds006629-smoke-{name}",
        "outer_test_groups": name,
        "label_shuffle_control": label_shuffle_control,
        "diagnostics_best_time": "0.184",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    pd.DataFrame(
        {
            "dataset_id": ["ds006629"],
            "subject": [name],
            "epochs_path": [f"{name}_epo.fif"],
            "n_trials": [3],
            "labels": ["a|b"],
            "runs": ["01"],
        }
    ).to_csv(output / "stage_summary.csv", index=False)
    pd.DataFrame({"time": [0.184], "accuracy": [0.5], "balanced_accuracy": [0.5]}).to_csv(
        decode / "time_decode_summary.csv",
        index=False,
    )
    return output


def test_openneuro_aggregate_accepts_numeric_false_manifest_bool(tmp_path: Path) -> None:
    root = tmp_path / "shards"
    shard_false = _write_minimal_shard(root, "sub-01", label_shuffle_control="false")
    shard_zero = _write_minimal_shard(root, "sub-02", label_shuffle_control="0")

    aggregate_dir = tmp_path / "aggregate"
    aggregate_workflow_outputs([shard_false, shard_zero], out_dir=aggregate_dir)

    manifest = json.loads((aggregate_dir / "run_manifest.json").read_text(encoding="utf-8"))
    quality = pd.read_csv(aggregate_dir / "workflow_quality_summary.csv")
    assert manifest["label_shuffle_control"] == "false"
    assert bool(quality.loc[0, "label_shuffle_control"]) is False
    assert quality.loc[0, "shard_count"] == 2

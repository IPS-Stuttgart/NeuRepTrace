from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import neureptrace  # noqa: F401  # installs runtime compatibility patches
from neureptrace.openneuro_decode_diagnostics import workflow_quality_summary


def test_workflow_quality_summary_accepts_list_manifest_provenance(tmp_path: Path) -> None:
    output_dir = tmp_path / "openneuro-run"
    output_dir.mkdir()
    manifest = {
        "dataset": "ds-test",
        "mode": "smoke",
        "artifact_name": "openneuro-test",
        "label_shuffle_control": False,
        "ensemble_weights": [0.25, 0.75],
        "ensemble_source_decoders": ["logistic", "linear_svm"],
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    diagnostics = {
        "decode_summary": {"exists": True},
        "stage_summary": {"n_subjects": 2, "total_trials": 12},
    }
    best_rows = pd.DataFrame(
        {
            "selection_metric": ["balanced_accuracy"],
            "time": [0.184],
            "balanced_accuracy": [0.5],
            "selection_value": [0.5],
        }
    )

    quality = workflow_quality_summary(output_dir, diagnostics, best_rows)

    assert quality.loc[0, "ensemble_weights"] == [0.25, 0.75]
    assert quality.loc[0, "ensemble_source_decoders"] == ["logistic", "linear_svm"]

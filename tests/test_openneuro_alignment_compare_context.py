from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neureptrace.openneuro_alignment_compare import build_raw_alignment_comparison, build_variant_summary


def _write_artifact(
    root: Path,
    name: str,
    *,
    method: str,
    value: float,
    subjects: str,
    label_shuffle_control: bool = False,
) -> Path:
    output = root / name / "openneuro_ds000117_smoke"
    decode = output / "decode"
    decode.mkdir(parents=True)
    (output / "run_manifest.json").write_text(
        json.dumps(
            {
                "artifact_name": name,
                "dataset": "ds000117",
                "mode": "smoke",
                "subjects": subjects,
                "runs": "01,02",
                "n_subjects": len(subjects.split(",")) if "," in subjects else subjects,
                "label_shuffle_control": str(label_shuffle_control).lower(),
                "label_shuffle_seed": 13,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "time": [0.184],
            "balanced_accuracy": [value],
            "alignment_method": [method],
            "alignment_anchor_mode": ["class_repetition" if method != "none" else "class_mean"],
            "alignment_anchor_column": [""],
            "alignment_target_projection": ["group_projection"],
        }
    ).to_csv(decode / "time_decode_summary.csv", index=False)
    return output


def test_raw_alignment_comparison_does_not_mix_subject_sets_or_shuffle_runs(tmp_path: Path):
    root = tmp_path / "artifacts"
    raw_six = _write_artifact(root, "raw-six", method="none", value=0.50, subjects="1-6")
    aligned_three = _write_artifact(root, "aligned-three", method="mcca", value=0.80, subjects="1-3")
    aligned_shuffle_six = _write_artifact(
        root,
        "aligned-shuffle-six",
        method="mcca",
        value=0.90,
        subjects="1-6",
        label_shuffle_control=True,
    )
    aligned_real_six = _write_artifact(root, "aligned-real-six", method="mcca", value=0.55, subjects="1-6")

    variants = build_variant_summary(
        [raw_six, aligned_three, aligned_shuffle_six, aligned_real_six],
        fixed_time=0.184,
    )
    comparison = build_raw_alignment_comparison(variants)

    assert comparison["best_alignment_artifact"].tolist() == ["aligned-real-six"]
    assert comparison.loc[0, "raw_artifact"] == "raw-six"
    assert comparison.loc[0, "subjects"] == "1-6"
    assert comparison.loc[0, "label_shuffle_control"] is False
    assert float(comparison.loc[0, "score_delta_alignment_minus_raw"]) == 0.05

"""Regression tests for source-alignment import-hook composition."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_contrastive_and_oracle_source_alignment_patches_compose_in_fresh_interpreter():
    repo_src = Path(__file__).resolve().parents[1] / "src"
    env = os.environ.copy()
    pythonpath = str(repo_src)
    if env.get("PYTHONPATH"):
        pythonpath = pythonpath + os.pathsep + env["PYTHONPATH"]
    env["PYTHONPATH"] = pythonpath

    script = textwrap.dedent(
        """
        import importlib
        import neureptrace

        sa = importlib.import_module("neureptrace.decoding.source_alignment")
        assert sa.normalize_source_alignment_method("contrastive-subject-alignment") == "contrastive"

        cfg = sa.source_alignment_config(
            method="contrastive",
            target_projection=sa.ORACLE_TARGET_CALIBRATED_ALIGNMENT,
        )
        meta = cfg.static_metadata()
        assert meta["alignment_method"] == "contrastive"
        assert meta["alignment_oracle_target_calibrated"] is True
        assert meta["alignment_oracle_target_projection_source"] == "scored_heldout_target_rows"
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, env=env)

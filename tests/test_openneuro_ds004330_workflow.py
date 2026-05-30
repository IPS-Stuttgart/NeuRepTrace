from __future__ import annotations

import json
import re
from pathlib import Path

from neureptrace.openneuro_meg import DATASET_SPECS, subject_label


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ds004330_sharded_dispatch_workflow() -> str:
    return (REPO_ROOT / ".github" / "workflows" / "openneuro-ds004330-sharded-dispatch.yml").read_text(encoding="utf-8")


def test_ds004330_sharded_dispatch_default_matches_configured_subjects() -> None:
    workflow = _ds004330_sharded_dispatch_workflow()
    match = re.search(r"shard_groups_json:\n(?:.*\n){0,8}\s+default: '([^']+)'", workflow)

    assert match is not None
    assert tuple(json.loads(match.group(1))) == tuple(
        subject_label(DATASET_SPECS["ds004330"], subject) for subject in DATASET_SPECS["ds004330"].default_subjects
    )


def test_ds004330_sharded_dispatch_normalizes_validated_shards_before_dispatch() -> None:
    workflow = _ds004330_sharded_dispatch_workflow()

    assert "id: shards" in workflow
    assert "normalized_shards" in workflow
    assert 'handle.write(f"shard_groups_json={compact}\\n")' in workflow
    assert "SHARD_GROUPS_JSON: ${{ steps.shards.outputs.shard_groups_json }}" in workflow


def test_ds004330_sharded_dispatch_rejects_ambiguous_shards() -> None:
    workflow = _ds004330_sharded_dispatch_workflow()

    assert "Duplicate held-out shard" in workflow
    assert "contains duplicate held-out groups" in workflow
    assert "invalid group(s)" in workflow
    assert "^[A-Za-z0-9_.-]+$" in workflow

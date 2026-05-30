from __future__ import annotations

import json
import re
from pathlib import Path

from neureptrace.openneuro_meg import DATASET_SPECS, subject_label


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_ds004330_sharded_dispatch_default_matches_configured_subjects() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "openneuro-ds004330-sharded-dispatch.yml").read_text(encoding="utf-8")
    match = re.search(r"shard_groups_json:\n(?:.*\n){0,8}\s+default: '([^']+)'", workflow)

    assert match is not None
    assert tuple(json.loads(match.group(1))) == tuple(
        subject_label(DATASET_SPECS["ds004330"], subject) for subject in DATASET_SPECS["ds004330"].default_subjects
    )

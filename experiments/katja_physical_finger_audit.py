"""Audit global physical-finger semantics in the Katja event-feature cache.

The Julia-comparable benchmark maps each participant's four variable physical
finger codes to participant-local classes 0--3.  This audit checks whether that
mapping is globally consistent across participants and infers the fixed first
finger as the complement of the five-code global physical-finger universe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sort_codes(values: np.ndarray) -> tuple[Any, ...]:
    items = np.unique(values).tolist()
    try:
        return tuple(sorted(items))
    except TypeError:
        return tuple(sorted(items, key=str))


def audit_cache(cache_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = Path(cache_path)
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "subjects",
            "press_positions",
            "finger_codes",
            "correct_order",
            "metadata_json",
        }
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Cache is missing required arrays: {missing}")
        subjects = archive["subjects"].astype(str)
        press_positions = archive["press_positions"].astype(int)
        finger_codes = np.asarray(archive["finger_codes"])
        correct_order = archive["correct_order"].astype(bool)
        metadata = json.loads(str(archive["metadata_json"].item()))

    included = correct_order & np.isin(press_positions, np.asarray((2, 3, 4, 5)))
    if not np.any(included):
        raise ValueError("No retained variable-finger rows were found.")

    global_codes = _sort_codes(finger_codes[included])
    if len(global_codes) != 5:
        raise ValueError(
            "The physical-finger audit requires exactly five global codes; "
            f"found {global_codes!r}."
        )

    participants = tuple(dict.fromkeys(subjects[included].tolist()))
    rows: list[dict[str, Any]] = []
    mappings_by_local_class: dict[int, list[Any]] = {index: [] for index in range(4)}
    for subject in participants:
        mask = included & (subjects == subject)
        variable_codes = _sort_codes(finger_codes[mask])
        if len(variable_codes) != 4:
            raise ValueError(
                f"Participant {subject!r} has {len(variable_codes)} variable codes: "
                f"{variable_codes!r}."
            )
        fixed_codes = tuple(code for code in global_codes if code not in variable_codes)
        if len(fixed_codes) != 1:
            raise ValueError(
                f"Participant {subject!r} has ambiguous fixed code complement: "
                f"{fixed_codes!r}."
            )
        for local_class, physical_code in enumerate(variable_codes):
            mappings_by_local_class[local_class].append(physical_code)
        rows.append(
            {
                "participant": subject,
                "fixed_physical_code": fixed_codes[0],
                "variable_physical_codes": ",".join(str(code) for code in variable_codes),
                "local_0_physical": variable_codes[0],
                "local_1_physical": variable_codes[1],
                "local_2_physical": variable_codes[2],
                "local_3_physical": variable_codes[3],
                "n_event_rows": int(np.count_nonzero(mask)),
                "n_trials": int(np.count_nonzero(mask) // 4),
            }
        )

    frame = pd.DataFrame(rows).sort_values("participant").reset_index(drop=True)
    local_class_code_sets = {
        str(local_class): [str(code) for code in _sort_codes(np.asarray(values))]
        for local_class, values in mappings_by_local_class.items()
    }
    mapping_consistent = all(
        len(code_set) == 1 for code_set in local_class_code_sets.values()
    )
    fixed_counts = (
        frame["fixed_physical_code"].astype(str).value_counts().sort_index().to_dict()
    )
    report = {
        "cache_path": str(path),
        "cache_format": metadata.get("format"),
        "global_physical_codes": [str(code) for code in global_codes],
        "n_participants": len(participants),
        "participant_local_mapping_globally_consistent": mapping_consistent,
        "physical_codes_by_local_class": local_class_code_sets,
        "fixed_physical_code_counts": {
            str(key): int(value) for key, value in fixed_counts.items()
        },
        "n_distinct_fixed_codes": int(frame["fixed_physical_code"].nunique()),
        "structural_head_is_nontrivial": bool(
            not mapping_consistent or frame["fixed_physical_code"].nunique() > 1
        ),
    }
    return frame, report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame, report = audit_cache(args.feature_cache)
    frame.to_csv(output / "katja_physical_finger_mapping.csv", index=False)
    (output / "katja_physical_finger_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print("\n" + json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

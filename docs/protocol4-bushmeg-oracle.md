# BUSH-MEG Protocol-4 oracle alignment

`neureptrace-bushmeg-protocol4-oracle` runs BUSH-MEG source-LOSO decoding with
`oracle_target_calibrated_alignment` enabled.  This is a **debug upper bound**:
the held-out target subject's scored labels or label-derived anchors are
available to the alignment transform.

Do not mix these rows into Protocol-1 strict source-only or Protocol-2
zero-label target-adaptive leaderboards.

## Example

```bash
export BUSH_MEG_DATA_DIR=/path/to/Bush_MEG-Data/MEG-Data

poetry run neureptrace-bushmeg-protocol4-oracle \
  configs/bush_meg/protocol4_oracle_response_window_c.yml \
  --include-oracle \
  --methods procrustes,hyperalignment,mcca \
  --out-dir results/bush_meg/protocol4_oracle_response_window_c
```

The command writes one directory per oracle method plus aggregate files:

- `summary.csv`
- `predictions.csv`
- `inner_cv.csv`
- `method_metadata.csv`
- `provenance.json`

Every aggregate row is tagged with:

- `protocol_category = 4`
- `protocol_name = oracle_target_calibrated_alignment`
- `uses_target_labels_for_fitting = true`
- `valid_for_zero_calibration = false`
- `valid_for_strict_source_only = false`
- `debug_upper_bound = true`

The required `--include-oracle` flag is intentional. It prevents accidental
target-label leakage in normal BUSH-MEG benchmark scripts.

# OpenNeuro Alignment Debug Findings

This note records current artifact-level evidence for the source-alignment debug
path. It is not a benchmark claim: oracle target-calibrated runs use held-out
target labels or anchors and are diagnostic upper bounds only.

## ds000117 Event-Code Alignment

Comparator output:

```bash
python -m neureptrace.openneuro_alignment_compare \
  outputs/ds000117_raw_all_events_20260609/openneuro-meg-ds000117-full-shard-aggregate \
  outputs/ds000117_event_code_followup_20260609/27200779414/openneuro-meg-ds000117-full-shard-aggregate \
  outputs/ds000117_event_code_followup_20260609/27200777169/openneuro-meg-ds000117-full-shard-aggregate \
  outputs/ds000117_event_code_oracle_20260609/27200264477/openneuro-meg-ds000117-full-shard-aggregate \
  outputs/ds000117_event_code_oracle_20260609/27200273151/openneuro-meg-ds000117-full-shard-aggregate \
  outputs/ds000117_event_code_oracle_20260609/27200282180/openneuro-meg-ds000117-full-shard-aggregate \
  --out-dir results/ds000117-alignment-debug-full-aggregate-20260609 \
  --fixed-time 0.184 \
  --min-delta 0.01
```

At fixed time 0.184 s:

| run | protocol | method | anchor | target projection | balanced accuracy |
| --- | --- | --- | --- | --- | ---: |
| 27201369939 | benchmark-valid raw | none | class_mean | group_projection | 0.5532 |
| 27200779414 | benchmark-valid strict alignment | mcca | event_code_mean | group_projection | 0.4952 |
| 27200264477 | oracle upper bound | procrustes | event_code_mean | oracle_target_calibrated_alignment | 0.7247 |
| 27200273151 | oracle upper bound | hyperalignment | event_code_mean | oracle_target_calibrated_alignment | 0.7276 |
| 27200777169 / 27200282180 | oracle upper bound | mcca | event_code_mean | oracle_target_calibrated_alignment | 0.7333 |

Interpretation:

- Strict event-code MCCA does not improve the benchmark-valid ds000117 full
  run; it trails raw by -0.0581 balanced accuracy at 0.184 s.
- Oracle target-calibrated event-code MCCA improves over strict event-code MCCA
  by +0.2381 balanced accuracy at the same fixed time.
- Therefore the alignment machinery and event-code anchors can produce a strong
  upper-bound effect, but the source-only/group-projection target transform is
  the current bottleneck.
- This is a debug result, not a publishable alignment gain. The oracle rows are
  explicitly invalid for benchmark reporting.

Current claim status:

> Oracle target-calibrated alignment shows that ds000117 event-code anchors can
> support a strong cross-subject common-space upper bound, but strict source-only
> event-code alignment does not beat the raw decoder. The result localizes the
> failure to target projection/calibration rather than proving a benchmark-valid
> cross-subject alignment improvement.

Remaining evidence needed before any stronger claim:

- Re-run the full ds000117 strict and oracle alignment variants with
  `alignment_diagnostics.csv` enabled, so actual components, alignment rows,
  channel-projection collapse, and anchor correlations are present in the full
  artifact.
- Add matched `class_repetition` full runs at the same six-subject/all-event
  scope if the specific anchor-semantics question remains important.
- A publishable positive alignment claim requires a benchmark-valid strict run
  to beat raw, not just an oracle upper bound.

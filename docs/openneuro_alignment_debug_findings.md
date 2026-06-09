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
| 27216491916 | benchmark-valid disjoint target calibration | mcca | event_code_mean | target_calibrated_alignment | 0.4120 |

Interpretation:

- Strict event-code MCCA does not improve the benchmark-valid ds000117 full
  run; it trails raw by -0.0581 balanced accuracy at 0.184 s.
- Oracle target-calibrated event-code MCCA improves over strict event-code MCCA
  by +0.2381 balanced accuracy at the same fixed time.
- Benchmark-valid disjoint target calibration does not rescue event-code MCCA in
  the current setup. It trails strict event-code MCCA by -0.0831 and raw by
  -0.1412 balanced accuracy at 0.184 s.
- Therefore the alignment machinery and event-code anchors can produce a strong
  upper-bound effect, but the source-only/group-projection target transform is
  the current bottleneck. The valid sparse-calibration variant tested so far is
  not sufficient.
- This is a debug result, not a publishable alignment gain. The oracle rows are
  explicitly invalid for benchmark reporting.

The 27216491916 target-calibrated run emitted `alignment_diagnostics.csv` for
the aggregate and all six subject shards. Its aggregate diagnostics show why this
variant is not a positive alignment result:

| field | value |
| --- | ---: |
| n_alignment_rows | 9 |
| actual_components | 9 |
| feature_dim | 7650 |
| decode_feature_dim | 9 |
| alignment_dimensionality_reduction | true |
| anchor_row_correlation_before | 0.7797 |
| anchor_row_correlation_after | 0.8884 |
| source_inner_decoding_before_alignment | 0.4987 |
| source_inner_decoding_after_alignment | 0.4645 |

The target transform was
`target_calibrated_template_ridge_least_squares`. Anchor correlation improved,
but source-inner decoding worsened and the held-out target score fell. The
projection also reduced the high-dimensional channel-time feature space to a
9-dimensional aligned decode space, so this diagnostic is consistent with a
sparse/low-rank calibration bottleneck rather than a robust benchmark-valid
alignment gain. Subsequent diagnostics separate this ordinary aligned-space
dimensionality reduction from the cross-window channel-projection-collapse
fallback.

Current claim status:

> Oracle target-calibrated alignment shows that ds000117 event-code anchors can
> support a strong cross-subject common-space upper bound, but strict source-only
> and sparse disjoint target-calibrated event-code alignment do not beat the raw
> decoder. The result localizes the failure to valid target projection and
> calibration, rather than proving a benchmark-valid cross-subject alignment
> improvement.

Remaining evidence needed before any stronger claim:

- Re-run the full ds000117 strict and oracle alignment variants with
  `alignment_diagnostics.csv` enabled if exact dimensionality is needed for
  those older artifacts. The target-calibrated 27216491916 artifact already has
  aggregate and per-shard diagnostics.
- Add matched `class_repetition` full runs at the same six-subject/all-event
  scope if the specific anchor-semantics question remains important.
- A publishable positive alignment claim requires a benchmark-valid strict run
  to beat raw, not just an oracle upper bound.

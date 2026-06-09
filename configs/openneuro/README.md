# OpenNeuro MEG LOSO Recipes

These configs target leave-one-subject-out MEG decoding after raw BIDS files
have been staged into per-subject MNE Epochs files with:

```bash
python -m neureptrace.openneuro_meg stage --dataset ds006629 --bids-root data/ds006629 --staged-dir data/openneuro-staged
```

Set `NEUREPTRACE_OPENNEURO_STAGED_DIR` to the staging directory before running
`neureptrace-decode-from-config`.

Supported recipes:

- `ds000117_face_recognition.yml`: classic multisubject face-recognition MEG,
  default label is `stim_type` (`Famous`, `Unfamiliar`, `Scrambled`).
- `ds004276_words.yml`: auditory words, default label is binary word length
  (`short` vs `long`) derived from the behavior file.
- `ds006629_singsing.yml`: SINGSING auditory oddball, default label is
  `trial_type` (`Stand`, `Large dev`, `Inter dev`).
- `ds004330_object_drawing.yml`: object drawing dynamics, default label is the
  visual form derived from `trial_type` (`Drawing`, `Sketch`, `Photo`).

The `OpenNeuro MEG LOSO` GitHub Actions workflow can stage and decode these
recipes directly. On self-hosted runners it stores raw BIDS files and staged
Epochs under `/home/github-runner/.cache/datasets/openneuro`; on GitHub-hosted
runners it uses `actions/cache`. If `OPENNEURO_API_KEY` is configured as a
repository secret, the workflow logs in to `openneuro-py` before downloading
missing files.

For matched null controls, pass `--set decoding.label_shuffle_control=true`
to `neureptrace-decode-from-config` or enable `label_shuffle_control` in the
workflow dispatch form. This shuffles only training labels inside each outer
fold; held-out labels and group splits stay unchanged. Per-subject, confusion,
class-count, and time-course diagnostics can be generated from any observation
CSV with `neureptrace-loso-observation-diagnostics`. For matched real versus
shuffle comparisons, set the workflow `diagnostics_best_time` input to the
predeclared real-run peak, for example `0.184`, so the null is not selected at
its own best time.

For `logistic-svm-ensemble` runs, the config and workflow can also tune the
probability ensemble without code changes:

```bash
neureptrace-decode-from-config configs/openneuro/ds006629_singsing.yml \
  --set decoding.ensemble_weights='[0.7,0.3]' \
  --set decoding.ensemble_baseline_window='[-0.35,-0.05]'
```

Use `decoding.ensemble_baseline_window=null` to disable baseline debiasing. The
same controls are available in the GitHub Actions dispatch form as
comma-separated values, which makes logistic-heavy versus SVM-heavy follow-up
runs directly comparable to the current ds006629 result.
For class-imbalanced source folds, test a balanced logistic source while keeping
the same ensemble protocol:

```bash
neureptrace-decode-from-config configs/openneuro/ds006629_singsing.yml \
  --set decoding.ensemble_source_decoders='[multinomial-logistic-weighted,linear_svm]'
```

The workflow exposes the same source override as `ensemble_source_decoders`.
For a more diverse follow-up, add a third shrinkage-LDA source. If no explicit
weights are provided, non-default source sets use equal weights:

```bash
neureptrace-decode-from-config configs/openneuro/ds006629_singsing.yml \
  --set decoding.ensemble_source_decoders='[multinomial-logistic-weighted,linear_svm,shrinkage_lda]'
```

The workflow dispatch form also exposes a `config_overrides` field for
semicolon- or newline-separated `--set` overrides. A compact ds006629 peak
follow-up can therefore stay in the same LOSO protocol while testing, for
example:

```text
preprocessing.window_size=0.075;
preprocessing.window_step=0.024;
preprocessing.decode_window=[0.120,0.248];
decoding.temporal_train_window=[0.120,0.248];
decoding.temporal_train_mode=window_ensemble;
decoding.tune_hyperparameters=true;
decoding.tuning_scoring=balanced_accuracy;
decoding.tuning_c_grid=0.03,0.1,0.3,1,3
```

For the same dispatch, set `diagnostics_best_time=0.184` as a workflow input
when you want the diagnostic tables to report the predeclared ds006629 peak.
If source-subject class counts are imbalanced, test
`decoding.class_prior_correction=train_uniform` as a balanced-accuracy-oriented
variant; it divides fold-held-out posterior probabilities by the training-fold
class priors and renormalizes them before scoring. The workflow dispatch form
exposes the same setting as `class_prior_correction`, and writes the selected
mode to `run_manifest.json` for real-versus-null comparisons.

Oracle target-calibrated alignment upper bound:

Use `decoding.alignment_target_projection=oracle_target_calibrated_alignment`
only as a debug upper bound. It fits a held-out-subject projection from held-out
target labels or target anchors, so every output row is marked with
`alignment_oracle_target_calibrated=true`,
`alignment_debug_upper_bound=true`, and
`alignment_valid_for_benchmark=false`. Run it as a paired comparison against the
same strict source-only alignment with `decoding.alignment_target_projection`
left as `group_projection`.

Recommended first ds000117 six-subject oracle run:

```bash
gh workflow run openneuro-meg-loso.yml \
  --ref main \
  -f dataset=ds000117 \
  -f mode=full \
  -f subjects="1-6" \
  -f runs="01,02" \
  -f min_successful_subjects=6 \
  -f config_overrides='decoding.n_splits=6;decoding.alignment_method=mcca;decoding.alignment_anchor_mode=stimulus_id_mean;decoding.alignment_target_projection=oracle_target_calibrated_alignment;workflow.outer_test_group_shards_json=["sub-01","sub-02","sub-03","sub-04","sub-05","sub-06"]'
```

For the paired strict comparison, rerun the same command with
`decoding.alignment_target_projection=group_projection`. On ds000117,
newly staged epochs expose canonical `stimulus_id` and `event_code` metadata
columns derived from the public `stim_file` and `trigger` event columns.
Previously staged epochs with only `stim_file` and `trigger` still work through
the same resolver. In either case, `stimulus_id_mean` avoids treating same-class
repetition offsets as if they were guaranteed shared images across subjects.

For the ds000117 anchor-semantics diagnostic, keep every other setting fixed and
compare these `decoding.alignment_anchor_mode` values:

- `class_repetition`: legacy within-class offset anchors.
- `stimulus_id_mean`: actual image/stimulus identity means.
- `stimulus_id_repetition`: actual image/stimulus identity with repetition rows.
- `event_code_mean`: trigger/event-code identity means.
- `run_event_index_within_stimulus`: run-local repeated presentations of the
  same stimulus.

If a true identity anchor improves over `class_repetition`, the class-repetition
row correspondences are the likely failure mode. If true identity anchors still
hurt, focus next on target projection, sensor-space alignment, and dimensionality
rather than anchor semantics.

Between strict source-only and oracle debugging, run a disjoint
target-calibrated diagnostic:

```bash
gh workflow run openneuro-meg-loso.yml \
  --ref main \
  -f dataset=ds000117 \
  -f mode=full \
  -f subjects="1-6" \
  -f runs="01,02" \
  -f min_successful_subjects=6 \
  -f config_overrides='decoding.n_splits=6;decoding.alignment_method=mcca;decoding.alignment_anchor_mode=event_code_mean;decoding.alignment_target_projection=target_calibrated_alignment;decoding.alignment_target_calibration_per_anchor=1;decoding.alignment_target_calibration_seed=13;workflow.outer_test_group_shards_json=["sub-01","sub-02","sub-03","sub-04","sub-05","sub-06"]'
```

This reserves calibration rows from each held-out target subject by class,
stimulus, or event-code anchor and scores only the remaining disjoint rows.
Artifacts should report `alignment_target_calibrated=true`,
`alignment_debug_upper_bound=false`, and
`alignment_protocol_note=uses disjoint target calibration rows; not strict
source-only`. If this helps while strict source-only does not, the publishable
question becomes whether a small target-calibration protocol is acceptable for
the benchmark, not whether the alignment implementation can work.

Recommended first ds006629 six-subject oracle run:

```bash
gh workflow run openneuro-meg-loso.yml \
  --ref main \
  -f dataset=ds006629 \
  -f mode=full \
  -f subjects="1,2,4,5,7,8" \
  -f runs="0" \
  -f min_successful_subjects=6 \
  -f config_overrides='decoding.n_splits=6;decoding.alignment_method=mcca;decoding.alignment_anchor_mode=class_mean;decoding.alignment_target_projection=oracle_target_calibrated_alignment;workflow.outer_test_group_shards_json=["sub-01","sub-02","sub-04","sub-05","sub-07","sub-08"]'
```

Inspect `time_decode_summary.csv`, `observations.csv`,
`alignment_anchor_availability.csv`, and `alignment_diagnostics.csv` from both
artifacts. `alignment_anchor_availability.csv` is written before each alignment
fit, including folds that later fail, and reports common-anchor counts, retained
anchor rows, target-anchor coverage, estimated alignment rows, and prefit failure
reasons. `alignment_diagnostics.csv` is post-fit and reports the actual aligned
dimensionality, anchor correlation before/after, source-inner decoding, target
transform type, and channel-projection-collapse status. If the oracle run
improves while the paired strict run does not, the alignment machinery can work
and the source-only/no-target-calibration protocol is the likely bottleneck. If
the oracle run still hurts, treat feature construction, anchor construction, or
the alignment implementation as the next debugging target.

After downloading several alignment artifacts into one directory, summarize the
debug sweep with:

```bash
neureptrace-openneuro-alignment-compare \
  outputs/same_window_alignment_smoke_20260609 \
  --out-dir results/openneuro-alignment-debug \
  --fixed-time 0.184 \
  --min-delta 0.01
```

The comparator writes `alignment_variant_summary.csv`,
`alignment_vs_raw_comparison.csv`, `alignment_anchor_comparison.csv`,
`alignment_oracle_comparison.csv`, and `alignment_debug_summary.md`. Use the raw
comparison to see whether any benchmark-valid alignment condition helps at all;
use the anchor comparison to test whether true stimulus/event identity beats
`class_repetition`; use the oracle comparison to test whether target-calibrated
projection beats strict source-only projection.

The current ds000117 full artifact readout is recorded in
`docs/openneuro_alignment_debug_findings.md`: oracle event-code alignment gives a
large diagnostic upper-bound gain, but strict source-only event-code alignment
still trails raw and is not a benchmark-valid improvement.

Dataset-specific staging hardening:

- `ds004276` event files include probe rows that do not correspond to auditory
  word sounds. Staging filters to auditory word event rows before joining the
  behavior table.
- All recipes drop events whose requested epoch window would fall outside the
  raw file bounds before applying per-label trial caps.
- Integer PCA component requests are capped inside each training fold, so small
  smoke runs and inner calibration folds keep the requested provenance while
  avoiding infeasible `n_components` failures.

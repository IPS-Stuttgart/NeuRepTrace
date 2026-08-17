# Katja online-window evaluation preparation

## Endpoint boundary

The published NeuRepTrace Katja numbers and Julia's reported numbers currently
measure different tasks.

NeuRepTrace's existing endpoint is **event conditioned**:

- recorded press times are already known;
- presses 2--5 produce four scored events per retained sequence trial;
- the fixed first press and null/background periods are not scored;
- independent finger accuracy is reported separately from one-to-one trial
  assignment.

Julia's endpoint is **online sliding-window decoding**:

- 500-ms windows move through the 0--6 s execution period with a 40-ms stride;
- the press interval is currently described as `[-400,+100]` ms around each
  press;
- windows predict finger identity, sequence identity, overlap ratio, and serial
  position;
- the first press and null/background windows can contribute.

The event-conditioned 65.79% independent result and 71.01% one-to-one result
must therefore not be described as directly outperforming Julia's reported
62.5--64.5% online-window range. They remain evidence for a related, easier
endpoint.

## Two distinct time references

The Katja files use two relevant clocks:

1. **MEG/fractal-cue clock**

   MEG epoch time zero is the onset of the fractal cue. The behavioral press
   timestamp on this clock is

   ```text
   (cueDur[t] + timing[t, p]) / 1000
   ```

2. **Execution/go-cue clock**

   The 0--6 s online execution interval is naturally expressed relative to the
   end of the fractal cue. The corresponding behavioral press timestamp is

   ```text
   timing[t, p] / 1000
   ```

`UPPT002` records the press pulse on the MEG acquisition clock. After matching
its pulses, conversion to execution time is therefore

```text
trigger_execution = trigger_meg - cueDur / 1000
```

NeuRepTrace retains both clocks explicitly. The preparation workflow uses the
execution-relative timestamps for the provisional 0--6 s window geometry.
Whether Julia anchors her implementation exactly this way remains a point for
confirmation, so the MEG-relative values are retained as well.

## Press-timing audit

`neureptrace.katja_press_timing` compares behavioral timestamps against
`UPPT002` for all five presses. It records:

- behavioral MEG-clock timestamp;
- measured trigger timestamp;
- trigger-minus-behavior lag;
- matching residual relative to the configured expected lag;
- whether the trigger was matched;
- an explicit measured/fallback timestamp source.

Run the raw audit with:

```bash
python -m neureptrace.katja_press_timing \
  --dataset-root "/path/to/Katja Button Press Data" \
  --output-dir results/katja_press_timing
```

The audit includes the first press and does not require a decoding model.

Add the dual MEG/execution references with:

```bash
python -m neureptrace.katja_execution_time \
  --press-timing results/katja_press_timing/katja_press_timing_per_press.csv.gz \
  --dataset-root "/path/to/Katja Button Press Data" \
  --participants "s05,s06,..." \
  --output results/katja_press_timing/katja_press_timing_dual_reference.csv.gz
```

The preferred provisional online timestamp column is
`recommended_time_execution_seconds`. It uses the matched trigger when
available and otherwise the timing audit's documented fallback before
converting to the execution clock.

## Label-agnostic sliding-window manifest

`neureptrace.katja_sliding_window_manifest` constructs the candidate window grid
and stores the raw intersection duration between every decoding window and each
of the five press intervals:

```bash
python -m neureptrace.katja_sliding_window_manifest \
  --press-timing results/katja_press_timing/katja_press_timing_dual_reference.csv.gz \
  --press-time-column recommended_time_execution_seconds \
  --output results/katja_online/katja_window_intersections.csv.gz \
  --execution-start-seconds 0 \
  --execution-stop-seconds 6 \
  --window-width-ms 500 \
  --stride-ms 40 \
  --press-before-ms 400 \
  --press-after-ms 100
```

The output deliberately defines neither finger labels nor null labels. It stores
both intersection/window and intersection/press fractions, overlap ties, and the
maximum-overlap candidate. Julia's exact function can therefore be applied later
without rereading the multi-gigabyte SPM files.

The details still requiring the reference function are:

- whether the grid values denote starts, centers, or another anchor;
- the exact overlap-ratio denominator;
- the threshold and rule for finger versus null labels;
- handling of windows overlapping multiple presses;
- handling of windows before press 1 that overlap press 0;
- the exact sequence and serial-position target representations.

## Candidate trial splits

Until the exact split function is supplied, the repository supports three
explicit candidate conventions:

- `nested_rest`: one per-sequence permutation per seed; the first `k` trials are
  calibration and all remaining trials are evaluation;
- `independent_rest`: redraw separately for each `k`;
- `fixed_max_complement`: reserve the largest pool first and use one common
  evaluation complement, matching the earlier NeuRepTrace event analysis.

For example:

```bash
python -m neureptrace.katja_online_protocol splits \
  --trials results/katja_online/katja_trial_registry.csv \
  --output results/katja_online/katja_splits_nested_rest.csv.gz \
  --calibration-counts 1,3,5,10,15,20 \
  --seeds 0,1,2,3,4 \
  --mode nested_rest
```

These are protocol candidates, not assertions about Julia's implementation.
Her exact ten-subject registry and split function remain required.

## Reporting shell

After the reference function has produced labels and a model has produced
predictions, the reporting shell emits both uncertainty conventions:

```bash
python -m neureptrace.katja_online_protocol report \
  --predictions results/katja_online/window_predictions.csv \
  --output-dir results/katja_online/report \
  --expected-subjects "...exact ten IDs..." \
  --expected-seeds 0,1,2,3,4 \
  --expected-k 1,3,5,10,15,20
```

Outputs include:

- one score row per subject, seed, and k;
- Julia-style mean and sample SD over subject-by-seed folds;
- seed-averaged subject scores;
- NeuRepTrace-style population mean and SEM across subjects.

The report can additionally score sequence identity, serial position, and
overlap-ratio regression when the corresponding true/predicted columns exist.

## Supplied cache and matched endpoint

The supplied `meg_windows_0to6_100hz_stride4.npz` cache now fixes the ten
participants, 500 ms windows, 40 ms stride, raw six-class finger/rest labels,
press order, overlap fraction, per-finger occupancy, sequence ID, and complete
trial grouping. Julia provided this cache and the accompanying task
documentation for the matched comparison.
`neureptrace.katja_julia_window_benchmark` reproduces the validated NeuRepTrace
reference at 53.36% for `k=10` over ten targets and 56.03%
for `k=20` over the feasible common-nine cohort.

The shared cache and documented `tau=0.2` relabel rule define the common
endpoint; they do not define the NeuRepTrace method. Julia did not supply model
architecture, weights, training/adaptation code, or a split function. The
hierarchical TCN, source/target adapters, calibration-only selector,
trial-context Transformer, probability ensemble, and structured decoder are
independent NeuRepTrace implementations.

The comparison remains task- and data-matched rather than model-identical. The
collaborator's unpublished architecture, exact split implementation, and fold
predictions were not supplied. NeuRepTrace therefore uses deterministic nested
per-sequence calibration draws and reports the collaborator's 62.5--64.5% range
as descriptive context, not as a paired significance comparison.

## Sliding-window accuracy push

`neureptrace.katja_window_accuracy_push` evaluates the hierarchical temporal
model, causal and bidirectional trial-context models, five-model probability
ensembles, and constrained trial decoding. It requires a validated reference
run and reserves the maximum calibration pool before fitting, so every `k` is
scored on identical evaluation trials within a target and split seed:

```bash
neureptrace-katja-window-accuracy-push \
  --cache "/path/to/meg_windows_0to6_100hz_stride4.npz" \
  --raw-window-cache "/path/to/meg_windows_raw.npy" \
  --baseline-results results/katja_button/julia_window_fair_matched \
  --out-dir results/katja_button/window_accuracy_push \
  --targets s05,s06,s08,s09,s10,s11,s15,s16,s17,s18 \
  --k-values 1,3,5,10,15,20 \
  --split-seeds 0,1,2,3,4 \
  --model-seeds 0,1,2,3,4 \
  --context-modes offline,causal \
  --resume
```

The classifier factorizes the endpoint into `P(press)` and
`P(finger | press)`, using `press_ratios[:, 1:6]` as soft conditional targets.
Source epochs and adapter architecture are selected with held-out source
participants. Adaptation settings use calibration-only inner validation.
Target normalization, the two finger templates, and state-duration estimates
use calibration rows only; evaluation labels are never available during
fitting or decoding.

At evaluation time, the runner uses only information present in Julia's task:
the supplied MEG windows, held-out participant identity, trial membership, and
chronological window order. It does not consume target evaluation finger,
sequence, press-order, overlap, or press-ratio annotations, nor external
ground-truth press timestamps or cue durations. Structured decoding uses
calibration-observed templates and model-predicted auxiliary heads, never the
evaluation trial's true template or event labels.

Structured decoding uses finger, order, overlap, and template predictions in a
monotonic rest/press Viterbi model. The fixed auxiliary log-emission weights are
0.35 for order, 0.15 for overlap, and 0.15 for template identity. Bidirectional
Transformer and offline Viterbi rows are labeled as offline results; causal
Transformer and prefix-decoding rows are reported separately.

Because the collaborator's unpublished classifier code does not establish
whether trial grouping is consumed at inference, only source-only and
hierarchical-TCN independent-window rows are treated as the primary direct
comparison. Trial Transformer, hybrid, causal-prefix, and constrained-decoder
rows are supplementary. The report writes this classification explicitly to
`method_comparison_scope.csv` and produces a separate
`katja_window_accuracy_push_direct_comparison.png` figure.

Primary outputs are `fold_results.csv`, `summary_subject_sem.csv`,
`summary_julia_fold_sd.csv`, `paired_method_improvements.csv`, per-window NPZ
prediction bundles, `katja_window_accuracy_push.png`, and `report.md`.

Production shard aggregation uses `--require-full-design`. This independently
requires all ten targets, all 12 reported methods, the exact 6,960 result
identities, and the predeclared `k=20` common-nine cohort. It also inspects all
5,800 persisted prediction bundles, compares their actual evaluation-row
indices across every `k`, model seed, and model family, and requires
source-only probabilities to remain identical across `k`.

## Explicit-duration and auxiliary-head follow-up

The fixed v2 follow-up replaces the geometric press placement heuristic with
an explicit-duration 11-state HSMM: alternating rest and ordered press states
for the five presses in a trial. Duration and transition priors are estimated
from source labels plus the held-out participant's calibration trials only.
The decoder consumes predicted finger, order, overlap, and sequence-template
probabilities, and never target evaluation labels. The maximum calibration
pool is reserved first, nested subsets are used for smaller `k`, and every `k`
is scored on the same remaining evaluation trials within a target and seed.

On the common nine-target `k=20` cohort, after averaging five split seeds
within each target and then computing SEM across targets, the main results are:

| Method | Evaluation regime | Six-class accuracy |
|---|---|---:|
| Source-only | Protocol 1 | 29.46 +/- 1.47% |
| Direct Transformer ensemble | Protocol 3, independent window | 61.59 +/- 1.44% |
| Direct + auxiliary blend | Protocol 3, calibration-only | 62.20 +/- 1.47% |
| Direct + auxiliary blend + prior match | Protocol 2+3, transductive | 65.63 +/- 1.63% |
| Geometric structured Transformer | Protocol 3, offline structured | 65.16 +/- 1.51% |
| Explicit-duration Transformer | Protocol 3, offline structured | 72.22 +/- 1.80% |

The explicit-duration result improves on the direct Transformer by
10.63 +/- 0.61 percentage points in paired target-level differences. It is a
structured task-decoding result, not unconstrained per-window finger
classification, and must be reported separately from the direct comparison.
Two controls quantify how much task structure contributes at `k=20`:
timing/template priors with all neural probabilities made uniform achieve
57.70 +/- 1.89%, while predicted auxiliary heads plus duration priors with
uniform finger probabilities achieve 66.30 +/- 2.18%. Full finger
probabilities therefore add about 5.9 points beyond the auxiliary-only control.

Probability-marginal matching is also reported separately. It does not use
evaluation labels, but it estimates batch marginals from unlabeled evaluation
predictions. It is consequently a Protocol 2+3 transductive analysis rather
than strict calibration-only Protocol 3. The fixed equal auxiliary blend
without this step is the corresponding strict Protocol 3 control.

The validated compact report is produced by
`neureptrace-katja-window-accuracy-push-v2-report`. Its validation requires the
fixed common-target cohort, seed-within-target aggregation, complete method-by-k
curves, and the structured/transductive scope labels. Julia's reported
62.5--64.5% range remains descriptive because her per-fold predictions and
model code are unavailable for paired testing.

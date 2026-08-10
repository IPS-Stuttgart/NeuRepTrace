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
must therefore not be described as directly outperforming Julia's 59.4%
online-window result. They remain evidence for a related, easier endpoint.

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

## Information still required

A like-for-like run needs:

1. the exact ten participant IDs;
2. exact window-grid anchoring and time reference;
3. overlap and null-label function;
4. multiple-press and tie handling;
5. k-trial split implementation;
6. final averaging order and window/trial weighting.

A one-trial fixture containing cue duration, all five press times, window
starts/stops, and all four target labels is sufficient to convert these choices
into exact regression tests.

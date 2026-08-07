# Katja online-window evaluation preparation

## Endpoint boundary

The published NeuRepTrace Katja numbers and Julia's reported numbers currently
measure different tasks.

NeuRepTrace's existing endpoint is **event conditioned**:

- the recorded press times are already known;
- presses 2--5 produce four scored events per retained sequence trial;
- the fixed first press and null/background periods are not scored;
- independent finger accuracy is reported separately from one-to-one trial
  assignment.

Julia's endpoint is **online sliding-window decoding**:

- 500-ms windows move through the 0--6 s execution period with a 40-ms stride;
- the press interval is currently described as `[-400, +100]` ms around each
  press;
- windows predict finger identity, sequence identity, overlap ratio, and serial
  position;
- the first press and null/background windows can contribute.

Therefore, the event-conditioned 65.79% independent result and 71.01%
permutation-constrained result must not be described as directly outperforming
Julia's 59.4% online-window result. They remain useful evidence for a related,
easier endpoint.

## Press-time sources

The behavioral clock gives press `p` in trial `t` as

```text
(cueDur[t] + timing[t, p]) / 1000
```

relative to the MEG epoch's fractal-cue onset. The `UPPT002` channel records the
press pulse on the MEG acquisition clock and is expected to lag the behavioral
log slightly. `neureptrace.katja_press_timing` matches the five pulses to the
five behavioral presses and retains all of the following:

- behavioral timestamp;
- measured `UPPT002` timestamp;
- measured trigger-minus-behavior lag;
- match residual relative to the configured expected lag;
- a recommended timestamp that prefers the measured trigger and has an explicit
  fallback source.

Run the audit with:

```bash
python -m neureptrace.katja_press_timing \
  --dataset-root "/path/to/Katja Button Press Data" \
  --output-dir results/katja_press_timing
```

The audit includes the first press and does not require a finger-decoding model.

## Label-agnostic online-window manifest

`neureptrace.katja_sliding_window_manifest` constructs the window grid and
stores the raw intersection duration between every decoding window and each of
the five press intervals:

```bash
python -m neureptrace.katja_sliding_window_manifest \
  --press-timing results/katja_press_timing/katja_press_timing_per_press.csv.gz \
  --output results/katja_online/katja_window_intersections.csv.gz \
  --execution-start-seconds 0 \
  --execution-stop-seconds 6 \
  --window-width-ms 500 \
  --stride-ms 40 \
  --press-before-ms 400 \
  --press-after-ms 100
```

The output deliberately defines neither finger labels nor null labels. It records
both intersection/window and intersection/press fractions, overlap ties, and the
maximum-overlap candidate. Julia's exact function can therefore be applied later
without reading the multi-gigabyte SPM files again.

The remaining windowing details that require the reference function are:

- whether 0--6 s refers to window starts, centers, or another anchor;
- the exact overlap-ratio denominator;
- the threshold and rule for finger versus null labels;
- handling of windows overlapping multiple presses;
- handling of windows before press 1 that overlap press 0;
- the exact target representation for serial position and sequence identity.

## Candidate trial splits

Until the exact split function is supplied, the repository supports three
explicit candidate conventions:

- `nested_rest`: one per-sequence permutation per seed; first `k` trials are
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

These files are protocol candidates, not a claim that Julia used one of them.
Her exact ten-subject registry and split implementation remain required.

## Reporting shell

After the reference function has produced window labels and a model has produced
predictions, the reporting shell writes both uncertainty conventions:

```bash
python -m neureptrace.katja_online_protocol report \
  --predictions results/katja_online/window_predictions.csv \
  --output-dir results/katja_online/report \
  --expected-subjects "...exact ten IDs..." \
  --expected-seeds 0,1,2,3,4 \
  --expected-k 1,3,5,10,15,20
```

Outputs include:

- fold scores for each subject, seed, and k;
- Julia-style mean and sample SD over subject-by-seed folds;
- seed-averaged subject scores;
- NeuRepTrace-style population mean and SEM across subjects.

The report can additionally score sequence identity, serial position, and
overlap-ratio regression when the corresponding true/predicted columns exist.

## Information still required

A like-for-like run needs the following from the reference implementation:

1. exact ten participant IDs;
2. exact window-grid construction;
3. overlap and null-label function;
4. multiple-press/tie handling;
5. k-trial split implementation;
6. final averaging order and any window/trial weighting.

A one-trial fixture containing press times, window starts/stops, and all four
output targets is sufficient to turn these items into regression tests.

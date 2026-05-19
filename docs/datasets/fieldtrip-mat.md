# FieldTrip MAT Datasets

This page documents the NeuRepTrace convention for MATLAB v5 `.mat` files that
contain FieldTrip-like raw/trial data. It is intended for datasets that were
previously handled by PyMEGDec-specific loaders but can be represented as a
generic M/EEG decoding input.

## Scope

Use the `fieldtrip-mat` input format for files whose MATLAB payload is a struct
with FieldTrip raw-style fields such as:

- `label`
- `trial`
- `time`
- `trialinfo`
- `sampleinfo`
- `grad`

The expected semantic structure is:

```text
data.trial{trial_index}  -> channels x samples matrix
data.time{trial_index}   -> samples vector in seconds
data.trialinfo           -> per-trial labels or metadata
data.sampleinfo          -> original start/stop sample indices
data.grad                -> optional MEG sensor geometry
```

The `.mat` container alone is not sufficient to identify the dataset layout.
NeuRepTrace therefore separates the physical container from the internal data
structure:

```yaml
loader:
  input_format: fieldtrip-mat
  root_path: [data, 0]
```

`input_format: fieldtrip-mat` means that NeuRepTrace should parse a MATLAB file
and interpret the object at `root_path` as a FieldTrip raw/trial struct.

## Default paths

The PyMEGDec-style defaults are:

```yaml
loader:
  input_format: fieldtrip-mat
  root_path: [data, 0]
  trial_path: [trial, 0]
  time_path: [time, 0]
  label_path: [label, 0]
  trialinfo_path: [trialinfo, 0]
  sampleinfo_path: [sampleinfo, 0]
  grad_path: [grad, 0]
```

These defaults correspond to MATLAB files where the top-level variable is named
`data` and the loaded struct stores MATLAB cell arrays for `trial` and `time`.

If a dataset was saved with different `scipy.io.loadmat` options or different
MATLAB nesting, override the paths explicitly in the dataset YAML rather than
changing the loader code.

## Channel labels and trimming

FieldTrip raw data should have one channel label for each row of each trial
matrix. Some legacy files contain additional labels for reference,
head-localization, or status channels that are not present in `data.trial`.

NeuRepTrace should treat the trial matrix as authoritative:

```text
n_channels = data.trial{1}.shape[0]
```

If `data.label` is longer than `n_channels`, the loader trims it to the trial
channel count and emits a `RuntimeWarning`.

If `data.label` is shorter than `n_channels`, the loader raises an error because
the channel mapping is under-specified.

The same rule applies to channel-level `grad` fields when they can be recognized
unambiguously:

- `grad.label`
- `grad.chantype`
- `grad.chanunit`
- `grad.chanpos`
- `grad.chanori`

Coil-level fields such as `grad.coilpos`, `grad.coilori`, and `grad.tra` should
not be blindly trimmed because their first dimension may describe coils rather
than data channels.

## Metadata convention

The first `trialinfo` column is used as the decoding label by default:

```yaml
metadata:
  label_column: condition
  trialinfo_column: 0
  label_base: 1
```

`label_base: 1` means that labels are MATLAB-style one-based class identifiers.
For downstream Python decoding, the loader may expose a zero-based `condition`
column while preserving the raw value as `trialinfo` or `trialinfo_0`.

`sampleinfo` should be copied to metadata as:

```text
sample_start
sample_stop
```

These columns make it possible to trace each decoded trial back to the original
recording sample range.

## Direct decoding command

Once FieldTrip MAT support is enabled in `neureptrace-mne-time-decode`, decode a
single participant file directly:

```bash
neureptrace-mne-time-decode \
  --epochs "D:/Uni-Data/Bush_MEG-Data/MEG-Data/Part10Data.mat" \
  --input-format fieldtrip-mat \
  --fieldtrip-root-path data,0 \
  --fieldtrip-label-base 1 \
  --fieldtrip-ch-type grad \
  --label-column condition \
  --out results/part10_time_decode.csv \
  --observations-out results/part10_observations.csv \
  --subject Part10
```

The command should warn if channel-level labels or `grad` fields are trimmed.
Warnings are expected for legacy files whose `data.label` contains reference or
status channels beyond the rows present in `data.trial`.

## Dataset YAML prototype

For dataset-level runs, use a manifest or YAML file to describe the files and
experimental split semantics:

```yaml
schema_version: neureptrace.dataset.v1

dataset:
  id: bush_meg
  root: "D:/Uni-Data/Bush_MEG-Data/MEG-Data"

loader:
  input_format: fieldtrip-mat
  root_path: [data, 0]
  label_base: 1
  trim_overlong_labels: true
  ch_type: grad

participants:
  include: [10]

recordings:
  main:
    pattern: "Part{participant}Data.mat"
  cue:
    pattern: "Part{participant}CueData.mat"

metadata:
  label_column: condition
  trialinfo_column: 0

analyses:
  - name: stimulus_main_to_cue
    task: transfer_decode
    train_recording: main
    test_recording: cue
    window_ms: 100
    step_ms: 50
    decoder: multiclass-svm
    observations_out: "results/part{participant}_main_to_cue_observations.csv"
    summary_out: "results/part{participant}_main_to_cue.csv"
```

The YAML should describe dataset conventions only. Loader code remains
responsible for interpreting MATLAB structs, validating trial/time shapes,
building MNE `EpochsArray` objects, and producing NeuRepTrace-compatible
metadata.

## Parity checks

Before deleting a PyMEGDec loader, verify the following for at least one real
participant file:

- number of trials matches the legacy loader;
- number of channels equals the trial matrix row count;
- `data.label` trimming emits a warning when expected;
- time vectors are identical across trials;
- sampling frequency matches the legacy estimate;
- class counts match the legacy `trialinfo` interpretation;
- a fixed time window produces the same flattened feature matrix as PyMEGDec;
- direct decoding writes NeuRepTrace probability observations.

For the Bush-style MEG files, a typical validation target is:

```text
720 trials
273 trial channels
1201 samples per trial
16 classes with 45 trials each
```

These are dataset checks, not assumptions that should be hard-coded in the
loader.

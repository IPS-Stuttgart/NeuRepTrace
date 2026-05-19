# PyMEGDec Migration

This page describes how to phase out PyMEGDec as a hard-coded dataset-specific
loader while preserving its scientific outputs through NeuRepTrace.

## Goal

The migration target is:

```text
PyMEGDec-specific file conventions -> dataset YAML or manifest
PyMEGDec MATLAB loading             -> NeuRepTrace fieldtrip-mat loader
PyMEGDec decoding workflows         -> NeuRepTrace decoding workflows
PyMEGDec summary tables             -> derived from probability observations
PyMEGDec alpha/paper helpers        -> examples or optional MEG extensions
```

Do not replace all of PyMEGDec with YAML alone. YAML should describe file names,
participant sets, labels, split semantics, and analysis defaults. Python loader
code should still implement MATLAB parsing, validation, FieldTrip semantics,
and MNE conversion.

## Responsibility split

Move these responsibilities to NeuRepTrace:

- FieldTrip-style MATLAB raw/trial loading;
- MNE `EpochsArray` conversion;
- time-resolved decoding;
- classifier calibration;
- probability-observation export;
- cross-validation summaries;
- transfer decoding between two recordings;
- plotting and aggregation that are not dataset-specific.

Keep these responsibilities out of NeuRepTrace core unless they become reusable:

- Bush-MEG-specific paper export scripts;
- one-off reaction-time joins;
- dataset-download helpers for private data shares;
- highly specific alpha-band figures;
- project-specific defaults that should live in an example YAML file.

Reusable MEG helpers can live under a small optional namespace such as
`neureptrace.meg`, but only after they have more than one dataset or paper use
case.

## Migration stages

### Stage 1: Loader parity

Add or enable the `fieldtrip-mat` loader and validate it against one known
participant file.

Required checks:

- trial count matches PyMEGDec;
- trial matrices have the same shape and orientation;
- labels are derived from the same `trialinfo` column;
- overlong `data.label` is trimmed with a warning;
- channel order is unchanged;
- time windows select the same sample indices;
- flattened feature matrices are numerically identical for a fixed window.

### Stage 2: Direct NeuRepTrace decoding

Run the participant file directly through NeuRepTrace:

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

The canonical intermediate output should be the NeuRepTrace probability
observation CSV. Legacy summary tables should be regenerated from observations
where possible instead of being treated as primary outputs.

### Stage 3: Dataset YAML

Create a dataset YAML that describes PyMEGDec's previous implicit conventions:

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

analyses:
  - name: stimulus_main_to_cue
    task: transfer_decode
    train_recording: main
    test_recording: cue
    label_column: condition
    window_ms: 100
    step_ms: 50
    decoder: multiclass-svm
```

The YAML runner should expand participant and recording patterns, load each
recording through the configured loader, then call the corresponding NeuRepTrace
workflow.

### Stage 4: Transfer decoding

Implement a generic transfer-decoding function that accepts separate training
and test recordings:

```python
run_time_resolved_transfer_decode(
    train_epochs,
    train_metadata,
    test_epochs,
    test_metadata,
    *,
    label_column="condition",
    window_ms=100,
    step_ms=50,
)
```

This replaces PyMEGDec workflows such as main-to-cue and cue-to-main. The core
model construction, probability alignment, observation schema, and plotting
should be shared with within-recording time-resolved decoding.

### Stage 5: Compatibility wrappers

Once NeuRepTrace can reproduce the decoding outputs, make PyMEGDec commands thin
wrappers that call NeuRepTrace and emit deprecation warnings.

Example wrapper behavior:

```python
warnings.warn(
    "PyMEGDec is deprecated for decoding workflows. Use NeuRepTrace dataset YAML instead.",
    DeprecationWarning,
    stacklevel=2,
)
```

Keep wrappers for one transition release so existing scripts continue to work.

## Old-to-new workflow mapping

| PyMEGDec workflow | NeuRepTrace target |
| --- | --- |
| Load `Part*Data.mat` | `input_format: fieldtrip-mat` |
| Load `Part*CueData.mat` | second recording entry in dataset YAML |
| Resolve data folder | `dataset.root` in YAML |
| Select participants | `participants.include` in YAML |
| Same-recording cross-validation | `neureptrace-mne-time-decode` or YAML `cross_validate` |
| Main-to-cue decoding | YAML `transfer_decode` with `train_recording: main`, `test_recording: cue` |
| Cue-to-main decoding | YAML `transfer_decode` with `train_recording: cue`, `test_recording: main` |
| Stimulus prediction CSV | NeuRepTrace probability observations |
| Accuracy/time summary | derived from NeuRepTrace observations and result CSVs |
| Onset scan | NeuRepTrace onset/stimulus detection from observations |
| Alpha metrics | `examples/bush_meg` or optional `neureptrace.meg` extension |
| Paper exports | project-specific example scripts |

## Command examples

### Single-file decoding

Old style:

```bash
pymegdec stimulus decoding --data-folder D:/Uni-Data/Bush_MEG-Data/MEG-Data --participants 10
```

NeuRepTrace direct decoding:

```bash
neureptrace-mne-time-decode \
  --epochs "D:/Uni-Data/Bush_MEG-Data/MEG-Data/Part10Data.mat" \
  --input-format fieldtrip-mat \
  --fieldtrip-root-path data,0 \
  --fieldtrip-label-base 1 \
  --label-column condition \
  --out results/part10_time_decode.csv \
  --observations-out results/part10_observations.csv \
  --subject Part10
```

### Dataset-level decoding

Target style:

```bash
neureptrace dataset run configs/bush_meg.yml \
  --participant 10 \
  --analysis stimulus_main_to_cue
```

If the dataset runner is not yet implemented, use the direct command above for
same-recording decoding and a temporary script for transfer decoding.

## Archive criteria

PyMEGDec can be reduced to wrappers or archived only after these checks pass:

```text
[ ] NeuRepTrace can load FieldTrip MAT files directly.
[ ] NeuRepTrace can decode a single FieldTrip MAT recording.
[ ] NeuRepTrace can run main-to-cue transfer decoding.
[ ] NeuRepTrace can run cue-to-main transfer decoding.
[ ] Dataset YAML replaces data-folder and participant naming conventions.
[ ] Part10 loader parity test passes.
[ ] Fixed-window feature parity test passes.
[ ] One full stimulus-decoding parity test passes.
[ ] Probability observations replace legacy primary output tables.
[ ] Alpha workflows are either ported or moved to examples.
[ ] Old PyMEGDec commands warn and delegate to NeuRepTrace.
[ ] Documentation contains old-to-new command mappings.
```

If any item is not needed scientifically, mark it explicitly as abandoned in the
migration issue or release notes rather than leaving an undocumented behavior
gap.

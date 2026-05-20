# Transfer from config

`transfer-from-config` trains a decoder on one configured subset of an epoched
dataset and evaluates it on another. It is the first config-driven replacement
for PyMEGDec-style main-to-cue transfer workflows.

```bash
neureptrace transfer-from-config configs/bush_meg/main_to_cue_transfer.yml
```

The dataset can be any config-supported epoched source. For the BUSH-MEG MATLAB
layout, attach a `split` column while declaring files:

```yaml
dataset:
  type: fieldtrip_mat
  root: ${BUSH_MEG_DATA_DIR}
  files:
    - path: "Part2Data.mat"
      participant: 2
      split: main
    - path: "Part2CueData.mat"
      participant: 2
      split: cue
```

Then define the transfer subsets:

```yaml
transfer:
  label_column: stimulus_class
  train_filter:
    split: main
  test_filter:
    split: cue
  classifier: multiclass-svm
  emission_mode: calibrated
```

Relative output paths follow the same path policy as `decode-from-config`:

```yaml
paths:
  base: cwd

outputs:
  base_dir: results/bush_meg
  summary_csv: main_to_cue_transfer_summary.csv
  observations_csv: main_to_cue_transfer_observations.csv
```

The command writes summary rows per time window. If `observations_csv` is set, it
also writes held-out probability observations suitable for downstream temporal
state and onset workflows.

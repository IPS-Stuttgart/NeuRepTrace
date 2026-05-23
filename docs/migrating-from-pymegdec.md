# Migrating from PyMEGDec

The migration goal is to retire PyMEGDec as a separate installed package while
preserving the old MEG dataset conventions as explicit NeuRepTrace configs.

## Command mapping

| PyMEGDec command | NeuRepTrace equivalent |
| --- | --- |
| `pymegdec data write-neureptrace-spec --out configs/bushmeg.yml` | `neureptrace dataset write-pymegdec-bushmeg --out configs/bushmeg.yml` |
| `pymegdec stimulus-decoding --data-dir DATA --participants 2` | `neureptrace decode-from-config configs/bush_meg/stimulus_decoding.yml --set dataset.root=DATA --set participants.ids=2` |
| `pymegdec cross-validate --data-dir DATA --participant 2` | Add a cross-validation config using the same `fieldtrip_mat` dataset recipe and run `neureptrace decode-from-config ...` |
| `pymegdec transfer --data-dir DATA --participant 2` | Represent main and cue files with `neureptrace dataset write-pymegdec-bushmeg`, then expand the `stimulus_transfer` or `stimulus_transfer_reverse` workflow with `neureptrace dataset manifest ...` |
| `pymegdec-make-synthetic-data --out DIR` | `neureptrace synthetic-fieldtrip --out DIR` |

## Dataset-spec migration

NeuRepTrace owns the maintained dataset-spec template for the historical
PyMEGDec/Bush-MEG `Part*Data.mat` and `Part*CueData.mat` convention:

```bash
neureptrace dataset write-pymegdec-bushmeg --out configs/bushmeg.yml
neureptrace dataset validate configs/bushmeg.yml
neureptrace dataset manifest configs/bushmeg.yml \
  --workflow stimulus_transfer \
  --out results/bushmeg_main_to_cue_manifest.csv
```

For compatibility with older migration notes, the dataset CLI also accepts
`write-pymegdec-spec` as an alias. The generated spec includes participant
ranges, main/cue file templates, FieldTrip-style MATLAB keys, 16-class chance
defaults, and both main-to-cue and cue-to-main transfer workflow metadata. Keep
only paper-specific alpha, CTF-geometry, and historical export scripts in
PyMEGDec.

## Python API mapping

Old code should move from PyMEGDec modules to config-driven NeuRepTrace entry
points:

```python
from neureptrace.decode_from_config import run_decode_from_config

run_decode_from_config(
    "configs/bush_meg/stimulus_decoding.yml",
    overrides=["dataset.root=/path/to/MEG-Data", "participants.ids=2"],
)
```

For private-data-free FieldTrip-style smoke fixtures:

```python
from neureptrace.io.synthetic_fieldtrip import SyntheticFieldTripConfig, write_synthetic_fieldtrip_dataset

write_synthetic_fieldtrip_dataset(
    "demo-data",
    SyntheticFieldTripConfig(participant_id=2, n_classes=16),
)
```

For lower-level loading:

```python
from neureptrace.dataset_config import load_epoch_dataset_from_config

dataset = load_epoch_dataset_from_config(
    "configs/bush_meg/stimulus_decoding.yml",
    overrides=["participants.ids=2"],
)
```

Generic reaction-time joins that used to be embedded in PyMEGDec's alpha/RT
workflow can be expressed without alpha-specific dependencies:

```python
from neureptrace.behavior.reaction_time import ReactionTimeCsvConfig, join_reaction_times, load_reaction_time_csv

reaction_times = load_reaction_time_csv("rt.csv", ReactionTimeCsvConfig(trial_index_base=1))
joined_rows = join_reaction_times(metric_rows, reaction_times)
```

## Suggested deprecation path

1. Add golden-output regression tests for the old PyMEGDec workflows.
2. Reproduce those outputs with NeuRepTrace configs.
3. Convert PyMEGDec commands into thin wrappers that call NeuRepTrace.
4. Emit deprecation warnings in PyMEGDec.
5. Archive PyMEGDec once the maintained configs and tests are stable.

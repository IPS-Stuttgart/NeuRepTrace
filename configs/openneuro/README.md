# OpenNeuro MEG LOSO Recipes

These configs target leave-one-subject-out MEG decoding after raw BIDS files
have been staged into per-subject MNE Epochs files with:

```bash
python -m neureptrace.openneuro_meg stage --dataset ds006629 --bids-root data/ds006629 --staged-dir data/openneuro-staged
```

Set `NEUREPTRACE_OPENNEURO_STAGED_DIR` to the staging directory before running
`neureptrace-decode-from-config`.

Supported recipes:

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

Dataset-specific staging hardening:

- `ds004276` event files include probe rows that do not correspond to auditory
  word sounds. Staging filters to auditory word event rows before joining the
  behavior table.
- All recipes drop events whose requested epoch window would fall outside the
  raw file bounds before applying per-label trial caps.
- Integer PCA component requests are capped inside each training fold, so small
  smoke runs and inner calibration folds keep the requested provenance while
  avoiding infeasible `n_components` failures.

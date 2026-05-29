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
CSV with `neureptrace-loso-observation-diagnostics`.

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

The workflow dispatch form also exposes the main follow-up sweep controls:
window size/step, decode-window range, normalization, baseline window, feature
preprocessor, PCA/components, temporal train-window range/mode, and nested
regularization tuning. A compact ds006629 peak follow-up can therefore stay in
the same LOSO protocol while testing, for example:

```text
window_size=0.075
window_step=0.024
decode_window=0.120,0.248
temporal_train_window=0.120,0.248
temporal_train_mode=window_ensemble
tune_hyperparameters=true
tuning_scoring=balanced_accuracy
tuning_c_grid=0.03,0.1,0.3,1,3
```

Dataset-specific staging hardening:

- `ds004276` event files include probe rows that do not correspond to auditory
  word sounds. Staging filters to auditory word event rows before joining the
  behavior table.
- All recipes drop events whose requested epoch window would fall outside the
  raw file bounds before applying per-label trial caps.
- Integer PCA component requests are capped inside each training fold, so small
  smoke runs and inner calibration folds keep the requested provenance while
  avoiding infeasible `n_components` failures.

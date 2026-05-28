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

Dataset-specific staging hardening:

- `ds004276` event files include probe rows that do not correspond to auditory
  word sounds. Staging filters to auditory word event rows before joining the
  behavior table.
- All recipes drop events whose requested epoch window would fall outside the
  raw file bounds before applying per-label trial caps.
- Integer PCA component requests are capped inside each training fold, so small
  smoke runs and inner calibration folds keep the requested provenance while
  avoiding infeasible `n_components` failures.

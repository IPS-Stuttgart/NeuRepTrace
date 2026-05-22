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

# BUSH-MEG Source-Only Pseudo-Trial Training

This workflow option adds source-only pseudo-trial training to the BUSH-MEG
nested LOSO workflow. It is intended for the regime where cue-based alignment is
above chance but consistently worse than source-only decoding.

The feature is target-free: for each outer or inner LOSO fit it averages only
source-subject training rows within each `(subject, class)` group. Held-out
subject trials remain untouched single trials, and cue files are not read or
used as classifier examples.

Two modes are available:

- `replace`: fit the classifier only on averaged source pseudo-trials.
- `augment`: append averaged source pseudo-trials to the original source trials.

The candidate grid key is `source_loso.candidate_grid.pseudotrials_per_class`.
A value of `0` disables pseudo-trials, so it can be included in the same nested
selection grid as a leakage-safe baseline.

Recommended focused run:

```bash
export BUSH_MEG_DATA_DIR=/path/to/PartData/files
neureptrace-bushmeg-source-loso configs/bush_meg/source_loso_pseudotrial_decoding.yml
```

Useful overrides for quick smoke checks:

```bash
neureptrace-bushmeg-source-loso configs/bush_meg/source_loso_pseudotrial_decoding.yml \
  --set source_loso.candidate_grid.pseudotrials_per_class='[4]' \
  --set source_loso.candidate_grid.feature_kinds='[evoked_dct]' \
  --set source_loso.candidate_grid.window_sets='[{name: single_175ms, centers: [0.175], window_size: 0.100}]'
```

Why this might help: the best BUSH-MEG numbers so far are source-only and only
modestly above chance, which is consistent with high single-trial noise and
cross-subject variance. Subject/class pseudo-trials trade sample count for
higher SNR while preserving subject-local structure and strict held-out-subject
semantics.

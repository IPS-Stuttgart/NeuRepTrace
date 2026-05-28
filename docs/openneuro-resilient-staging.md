# Resilient OpenNeuro staging

Large OpenNeuro MEG runs may contain a small number of corrupt or unreadable subject FIF files. The standard `neureptrace.openneuro_meg stage` command is intentionally strict and aborts on the first failed subject. For exploratory full-cohort screening, use the resilient wrapper instead:

```bash
python -m neureptrace.openneuro_resilient stage \
  --dataset ds006629 \
  --bids-root "$OPENNEURO_DATA_ROOT" \
  --staged-dir "$NEUREPTRACE_OPENNEURO_STAGED_DIR" \
  --subjects all \
  --runs all \
  --summary-out outputs/openneuro_ds006629_full/stage_summary.csv \
  --failure-summary-out outputs/openneuro_ds006629_full/stage_failures.csv \
  --successful-subjects-out outputs/openneuro_ds006629_full/successful_subjects.txt \
  --skip-failed-subjects \
  --min-successful-subjects 2 \
  --on-mismatch warn \
  --overwrite
```

The wrapper keeps the same staging options as the strict command, but records failed subjects in `stage_failures.csv` and writes the successful cohort as a compact `participants.ids` override. For example, if `sub-06` fails in `ds006629`, `successful_subjects.txt` contains the remaining numeric participant ids so the decode config can be run only on the subjects that were staged successfully.

In GitHub Actions, pass `--github-env "$GITHUB_ENV"` to append updated `OPENNEURO_SUBJECTS`, `OPENNEURO_N_SPLITS`, and `OPENNEURO_FAILED_SUBJECTS` values for subsequent decode steps.

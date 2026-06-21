# Category-2 unlabeled calibration-run alignment

This protocol implements the deployment-style hyperalignment/SRM setup where the
held-out target subject contributes a separate unlabeled calibration recording,
such as a movie, stimulus stream, or resting/task block with shared event anchors.
The target subject's decoding labels are never used.

## Information use

| Data source | Used? | Purpose |
| --- | --- | --- |
| Source decoding features `X_s` | Yes | downstream classifier training after projection |
| Source decoding labels `y_s` | Yes | downstream classifier training only |
| Source calibration features | Yes | fit source-subject projections and common space |
| Source calibration anchors | Yes | row-align calibration events across source subjects |
| Target calibration features `X_t^calib` | Yes | fit the held-out target projection |
| Target calibration anchors | Yes | row-align the target calibration recording to the source common-space template |
| Target decoding labels `y_t` | No | hidden until evaluation |

This is therefore **Protocol 2: unlabeled target-adaptive**. It is not strict
source-only because target calibration features are used, and it is not
supervised/calibrated target alignment because target task labels or target class
prototypes are not used.

## API

Use `align_train_test_with_unlabeled_calibration` from
`neureptrace.decoding.unlabeled_calibration_alignment`:

```python
from neureptrace.decoding.unlabeled_calibration_alignment import (
    align_train_test_with_unlabeled_calibration,
    unlabeled_calibration_alignment_config,
)

result = align_train_test_with_unlabeled_calibration(
    train_features=X_source_task,
    train_labels=y_source_task,
    train_subject_ids=source_subject_ids,
    test_features=X_target_task,
    source_calibration_features=X_source_movie,
    source_calibration_subject_ids=source_movie_subject_ids,
    source_calibration_anchor_values=source_movie_frame_ids,
    target_calibration_features=X_target_movie,
    target_calibration_anchor_values=target_movie_frame_ids,
    config=unlabeled_calibration_alignment_config(
        method="hyperalignment",  # also supports "procrustes" and "mcca"
        anchor_mode="stimulus_id_mean",
        components=64,
    ),
)

X_source_aligned = result.train_features
X_target_aligned = result.test_features
```

Downstream classifiers should be trained only on `X_source_aligned` and
`y_source_task`, then scored on `X_target_aligned`.

## Guardrails

The implementation keeps source decoding rows separate from source calibration
rows and target calibration rows. Every source decoding subject must have a
source calibration projection. Target calibration anchors must be label-free
shared calibration anchors, not target task labels or class prototypes.

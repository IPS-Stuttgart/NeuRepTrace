# Unlabeled anchor alignment

`neureptrace.decoding.unlabeled_anchor_alignment` implements a clean **Category 2 / unlabeled target-adaptive** alignment protocol for shared calibration anchors.

Use this when source subjects and the held-out target subject share a label-free calibration axis, for example:

- movie time bins,
- stimulus identifiers,
- event identifiers,
- resting-state segment ids,
- other external anchors that are not the decoded target labels.

The target projection is fitted from target calibration features and their anchor ids, then applied to separate target test rows. The API intentionally has no `target_labels` argument. Passing decoded class labels as anchors would no longer be an ordinary Category-2 claim.

::: neureptrace.decoding.unlabeled_anchor_alignment
    options:
      members:
        - AnchorProjection
        - UnlabeledAnchorAlignmentResult
        - fit_unlabeled_anchor_alignment
        - anchor_template
        - fit_anchor_projection
        - transform_with_anchor_projection

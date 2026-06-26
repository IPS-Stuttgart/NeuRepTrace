# Subspace alignment

`neureptrace.decoding.subspace_alignment` implements PCA subspace alignment for cross-subject transfer.

The method fits a source PCA basis and an unlabeled target PCA basis, rotates the source coordinates toward the target coordinates, and then allows a source-label classifier to be trained in the aligned space.

Protocol boundary:

- uses source features,
- uses source labels only for the optional classifier helper,
- uses unlabeled target features to fit the target subspace,
- does not accept target labels.

::: neureptrace.decoding.subspace_alignment
    options:
      members:
        - SubspaceAlignmentModel
        - SubspaceAlignmentResult
        - SubspaceAlignedClassificationResult
        - fit_subspace_alignment
        - fit_subspace_aligned_classifier
        - normalize_standardization_scope

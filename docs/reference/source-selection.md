# Source-domain selection

`neureptrace.decoding.source_selection` provides a generic source-domain selection and weighting utility for cross-subject transfer experiments.

The protocol is **Category 2 / unlabeled target-adaptive**. It uses source features, source domain identifiers, optional source labels for class-balanced weighting, and unlabeled target features. It does **not** accept target labels.

::: neureptrace.decoding.source_selection
    options:
      members:
        - SourceDomainSelectionResult
        - select_source_domains_by_target_similarity
        - selected_source_subset
        - normalize_source_selection_metric

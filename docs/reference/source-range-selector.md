# Source range selector

`neureptrace.decoding.source_range_selector` implements source-only feature selection by source feature range.

The protocol is **Category 1 / strict source-only**. Feature ranges and the selected feature mask are estimated from source rows only. Evaluation rows are transformed with the fitted mask but are not used to fit it.

::: neureptrace.decoding.source_range_selector
    options:
      members:
        - SourceRangeSelectorResult
        - fit_source_range_selector
        - select_source_range_features

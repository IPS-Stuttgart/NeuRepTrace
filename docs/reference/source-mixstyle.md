# Source MixStyle

`neureptrace.decoding.source_mixstyle` implements a source-only feature-space MixStyle augmentation for cross-subject domain generalization.

The method estimates per-source-domain feature means and scales, then creates synthetic source rows by mixing each row's domain statistics with another source domain's statistics. Labels are copied from the source rows. Held-out target features and target labels are not part of the API.

This is a **Protocol 1 / strict source-only** augmentation:

- uses `X_s`, `y_s`, and source-domain ids,
- does not use `X_t`,
- does not use `y_t`.

::: neureptrace.decoding.source_mixstyle
    options:
      members:
        - SourceMixStyleConfig
        - SourceMixStyleResult
        - source_mixstyle_config
        - augment_source_domains_mixstyle
        - mixstyle_row

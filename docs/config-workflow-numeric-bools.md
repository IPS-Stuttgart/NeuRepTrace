# Legacy config workflow numeric booleans

The legacy `config_workflow` parser accepts real booleans, quoted boolean strings, and numeric `0`/`1` flags for boolean configuration fields.

This keeps JSON/YAML workflow files compatible with common scheduler- and spreadsheet-generated configs where boolean values are serialized as `0` or `1`.

Ambiguous numeric values such as `2`, `-1`, `0.5`, or `NaN` remain invalid.

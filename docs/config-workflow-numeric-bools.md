# Legacy config workflow numeric booleans

The legacy config workflow parser accepts real booleans,
quoted boolean strings, and numeric zero/one flags for boolean
configuration fields.

This keeps JSON/YAML workflow files compatible with scheduler and
spreadsheet generated configs where boolean values are serialized
numerically.

Other numeric values remain invalid.

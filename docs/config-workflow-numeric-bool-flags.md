# Legacy config workflow numeric booleans

The legacy config workflow parser accepts real booleans, quoted
boolean strings, and numeric zero or one flags for boolean
configuration fields.

This keeps JSON and YAML workflow files compatible with scheduler
generated configs where boolean values are serialized numerically.
Other numeric values remain invalid.
